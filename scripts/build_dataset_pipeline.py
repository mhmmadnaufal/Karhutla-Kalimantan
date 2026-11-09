"""
Skrip CLI Pipeline Utama:
1. Membaca berkas CSV/SHP mentah NASA FIRMS VIIRS 375 m S-NPP (2017-2025).
2. Menyaring confidence ('nominal'/'high') & membuang sumber panas persisten (tambang/industri).
3. Menjalankan klasterisasi spatiotemporal event (d_max = 5 km, g_max = 2 hari) beserta diagnostik d_max = 10 km.
4. Menarik prediktor Lite-17 & FireMask VNP14A1 dari GEE via `ee.data.computePixels`
   pada grid UTM dinamis 256x256 @ 375 m, lalu menyimpan berkas `.npz` per event-hari.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd

sys.path.insert(0, r"c:\Neural-CA-Kalimantan")

from src.config import ClusteringConfig, GridConfig
from src.data.event_clustering import (
    filter_viirs_detections,
    cluster_spatiotemporal_events,
    build_patch_roi_grid,
    rasterize_viirs_footprints_256,
    compute_ca_initial_state_and_targets,
)
from src.data.gee_extractor import fetch_gee_patch_numpy, assemble_lite17_sample_dict


def main():
    parser = argparse.ArgumentParser(description="Build Custom VIIRS 375m Kalimantan Dataset (Lite-17)")
    parser.add_argument("--firms-csv", type=str, required=True, help="Path ke CSV/SHP mentah NASA FIRMS VIIRS S-NPP")
    parser.add_argument("--roi-shp", type=str, default=r"c:\Neural-CA-Kalimantan\GEE_Karhutla_SHP\ROI_Pulau_Kalimantan.shp")
    parser.add_argument("--out-dir", type=str, default=r"c:\Neural-CA-Kalimantan\data\processed")
    parser.add_argument("--d-max-km", type=float, default=5.0, help="Ambang jarak spasial DBSCAN (default: 5.0 km)")
    parser.add_argument("--g-max-days", type=int, default=2, help="Toleransi celah hari tertutup awan (default: 2 hari)")
    parser.add_argument("--fetch-gee", action="store_true", help="Tarik band prediktor dari GEE via computePixels")
    parser.add_argument("--max-events", type=int, default=None, help="Batas jumlah event yang diekspor (opsional)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    roi_gdf = gpd.read_file(args.roi_shp) if os.path.exists(args.roi_shp) else None

    print(f"[1/4] Membaca data deteksi VIIRS dari: {args.firms_csv}")
    if args.firms_csv.endswith(".shp"):
        raw_gdf = gpd.read_file(args.firms_csv)
        raw_gdf["longitude"] = raw_gdf.geometry.x
        raw_gdf["latitude"] = raw_gdf.geometry.y
        raw_df = pd.DataFrame(raw_gdf.drop(columns=["geometry"]))
    else:
        raw_df = pd.read_csv(args.firms_csv)

    cfg = ClusteringConfig(d_max_km=args.d_max_km, g_max_days=args.g_max_days)
    filtered_df = filter_viirs_detections(raw_df, cfg=cfg, roi_gdf=roi_gdf)
    print(f"      Deteksi valid setelah filter confidence & sumber panas statis: {len(filtered_df):,}")

    print(f"[2/4] Menjalankan klasterisasi spatiotemporal (d_max={args.d_max_km} km, g_max={args.g_max_days} hari)...")
    clustered_df, summary_df = cluster_spatiotemporal_events(
        filtered_df, d_max_km=args.d_max_km, g_max_days=args.g_max_days
    )
    summary_csv = os.path.join(args.out_dir, f"event_summary_d{int(args.d_max_km)}km_g{args.g_max_days}d.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"      Terbentuk {len(summary_df):,} event. Ringkasan disimpan di: {summary_csv}")

    if not args.fetch_gee:
        print("[Info] Mode --fetch-gee tidak diaktifkan. Klasterisasi event selesai.")
        return

    import ee
    ee.Initialize()

    events_to_run = summary_df["event_id"].tolist()
    if args.max_events is not None:
        events_to_run = events_to_run[: args.max_events]

    print(f"[3/4] Mengekspor tensor Lite-17 (256x256 @ 375 m) via GEE computePixels untuk {len(events_to_run)} event...")
    for eid in events_to_run:
        ev_info = summary_df[summary_df["event_id"] == eid].iloc[0]
        ev_pts = clustered_df[clustered_df["event_id"] == eid]
        patch_spec = build_patch_roi_grid(
            center_lon=float(ev_info["center_lon"]),
            center_lat=float(ev_info["center_lat"]),
        )

        # Rasterisasi mask harian sepanjang umur event
        daily_masks = {}
        for d_str, d_grp in ev_pts.groupby("acq_date"):
            daily_masks[d_str] = rasterize_viirs_footprints_256(d_grp, patch_spec, use_scan_track_footprint=True)

        active_dates = sorted(daily_masks.keys())
        for date_t in active_dates:
            gee_bands = fetch_gee_patch_numpy(patch_spec=patch_spec, date_t=date_t)
            valid_masks = {
                date_t: gee_bands["valid_t"].astype(np.uint8),
                (pd.to_datetime(date_t) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"): gee_bands["valid_next"].astype(np.uint8),
            }
            fire_hist = compute_ca_initial_state_and_targets(
                daily_masks=daily_masks,
                valid_masks=valid_masks,
                date_t=date_t,
                event_start_date=str(ev_info["start_date"]),
            )
            sample_dict = assemble_lite17_sample_dict(
                fire_history_dict=fire_hist,
                fire_freq_prev_years=np.zeros((256, 256), dtype=np.float32),
                gee_bands_dict=gee_bands,
                event_id=eid,
                date_t=date_t,
                patch_spec=patch_spec,
            )
            out_file = os.path.join(args.out_dir, f"{eid}_{date_t.replace('-', '')}.npz")
            np.savez_compressed(out_file, **sample_dict)
            print(f"      Saved: {out_file}")


if __name__ == "__main__":
    main()

