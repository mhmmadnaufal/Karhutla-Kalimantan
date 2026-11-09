"""
Modul Klasterisasi Spatiotemporal Event, Penyaringan VIIRS S-NPP 375 m,
Penentuan ROI Dinamis Lintas Zona UTM, & Rasterisasi Footprint 375 m.
Sesuai PRD v2 Bagian 1.6, 1.8, 1.9, 1.10 dan Keputusan Terkunci.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Point
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
import networkx as nx

from src.config import GridConfig, ClusteringConfig


def filter_viirs_detections(
    df: pd.DataFrame,
    cfg: ClusteringConfig = ClusteringConfig(),
    roi_gdf: Optional[gpd.GeoDataFrame] = None,
) -> pd.DataFrame:
    """
    Menyaring deteksi mentah NASA FIRMS VIIRS 375 m S-NPP sesuai PRD Bagian 1.6.2:
    1. Filter batas wilayah Kalimantan & rentang tahun (2017-2025).
    2. Simpan confidence 'nominal' dan 'high' (buang 'low').
    3. Jika kolom 'type' tersedia, simpan type == 0 (vegetation fire).
    4. Deteksi dan buang sumber panas statis/persisten (industri, coal seam fire, flare)
       yang aktif pada >= persistent_min_days_per_year atau >= persistent_min_years.
    """
    data = df.copy()
    # Normalisasi nama kolom standar FIRMS
    col_map = {c.lower(): c for c in data.columns}
    lon_col = col_map.get("longitude", col_map.get("lon", "longitude"))
    lat_col = col_map.get("latitude", col_map.get("lat", "latitude"))
    date_col = col_map.get("acq_date", col_map.get("date_t", "acq_date"))

    data["longitude"] = pd.to_numeric(data[lon_col], errors="coerce")
    data["latitude"] = pd.to_numeric(data[lat_col], errors="coerce")
    data = data.dropna(subset=["longitude", "latitude", date_col]).copy()

    # Filter bounding box Kalimantan
    min_lon, min_lat, max_lon, max_lat = cfg.kalimantan_bbox
    mask_bbox = (
        (data["longitude"] >= min_lon) & (data["longitude"] <= max_lon) &
        (data["latitude"] >= min_lat) & (data["latitude"] <= max_lat)
    )
    data = data[mask_bbox].copy()

    # Konversi tanggal UTC (PRD 1.6.1)
    data["acq_date"] = pd.to_datetime(data[date_col]).dt.strftime("%Y-%m-%d")
    data["year"] = pd.to_datetime(data["acq_date"]).dt.year
    data = data[(data["year"] >= cfg.start_year) & (data["year"] <= cfg.end_year)].copy()

    # 1. Filter Confidence (buang 'low' / 'l')
    if "confidence" in col_map:
        conf_series = data[col_map["confidence"]].astype(str).str.lower().str.strip()
        data = data[conf_series.isin(cfg.valid_confidences)].copy()

    # 2. Filter Type == 0 bila tersedia di arsip standar
    if "type" in col_map:
        type_vals = pd.to_numeric(data[col_map["type"]], errors="coerce")
        if type_vals.notna().any():
            data = data[type_vals == 0].copy()

    # 3. Filter Sumber Panas Persisten (PRD 1.6.2 & 1.7)
    data["grid_lon_stat"] = (data["longitude"] / cfg.persistent_grid_deg).round() * cfg.persistent_grid_deg
    data["grid_lat_stat"] = (data["latitude"] / cfg.persistent_grid_deg).round() * cfg.persistent_grid_deg

    cell_yearly = data.groupby(["grid_lon_stat", "grid_lat_stat", "year"])["acq_date"].nunique().reset_index()
    cell_yearly.rename(columns={"acq_date": "n_days_active"}, inplace=True)

    cell_summary = cell_yearly.groupby(["grid_lon_stat", "grid_lat_stat"]).agg(
        max_days_in_year=("n_days_active", "max"),
        years_active=("year", "nunique"),
    ).reset_index()

    persistent_cells = cell_summary[
        (cell_summary["max_days_in_year"] >= cfg.persistent_min_days_per_year) |
        (cell_summary["years_active"] >= cfg.persistent_min_years)
    ][["grid_lon_stat", "grid_lat_stat"]].copy()
    persistent_cells["is_persistent"] = True

    data = data.merge(persistent_cells, on=["grid_lon_stat", "grid_lat_stat"], how="left")
    n_before = len(data)
    data = data[data["is_persistent"].isna()].drop(columns=["grid_lon_stat", "grid_lat_stat", "is_persistent"]).copy()
    data["n_removed_persistent"] = n_before - len(data)

    # 4. Clip dengan poligon daratan Kalimantan jika disediakan
    if roi_gdf is not None and len(data) > 0:
        pts_gdf = gpd.GeoDataFrame(
            data,
            geometry=gpd.points_from_xy(data["longitude"], data["latitude"]),
            crs="EPSG:4326",
        )
        roi_4326 = roi_gdf.to_crs("EPSG:4326")
        pts_clipped = gpd.sjoin(pts_gdf, roi_4326[["geometry"]], how="inner", predicate="intersects")
        data = pd.DataFrame(pts_clipped.drop(columns=["geometry", "index_right"], errors="ignore"))

    return data.reset_index(drop=True)


def cluster_spatiotemporal_events(
    df: pd.DataFrame,
    d_max_km: float = 5.0,
    g_max_days: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Klasterisasi spatiotemporal dua tahap sesuai PRD Bagian 1.8 & 1.9:
    - Tahap 1 (Spasial per hari): DBSCAN dengan eps = d_max_km (dalam meter, proyeksi metrik).
    - Tahap 2 (Temporal lintas hari): Graph connected components yang menghubungkan klaster
      pada hari t dan t + delta_t (1 <= delta_t <= g_max_days + 1) jika jarak minimum
      antar titik kedua klaster <= d_max_km.
    Mengembalikan:
      (df_with_event_id, event_summary_df)
    """
    if len(df) == 0:
        raise ValueError("DataFrame deteksi kosong.")

    data = df.copy()
    data["date_dt"] = pd.to_datetime(data["acq_date"])

    # Proyeksi ke EPSG:32749 / metrik ekuivalen untuk perhitungan jarak Euclidean akurat
    gdf = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data["longitude"], data["latitude"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:32749")
    data["x_m"] = gdf.geometry.x.values
    data["y_m"] = gdf.geometry.y.values

    eps_m = d_max_km * 1000.0
    daily_clusters = []

    # Tahap 1: Klasterisasi spasial per hari
    for date_val, group in data.groupby("date_dt", sort=True):
        coords = np.column_stack([group["x_m"].values, group["y_m"].values])
        db = DBSCAN(eps=eps_m, min_samples=1).fit(coords)
        labels = db.labels_
        for lbl in np.unique(labels):
            idx = group.index[labels == lbl]
            node_id = f"{date_val.strftime('%Y-%m-%d')}_c{lbl}"
            data.loc[idx, "daily_cluster_id"] = node_id
            daily_clusters.append({
                "node_id": node_id,
                "date_dt": date_val,
                "coords": coords[labels == lbl],
                "indices": idx,
            })

    # Tahap 2: Asosiasi temporal dengan Graph Connected Components (toleransi celah g_max_days)
    G = nx.Graph()
    for dc in daily_clusters:
        G.add_node(dc["node_id"], date_dt=dc["date_dt"])

    # Kelompokkan berdasarkan tanggal untuk mempercepat pencarian pasangan hari
    by_date: Dict[pd.Timestamp, List[dict]] = {}
    for dc in daily_clusters:
        by_date.setdefault(dc["date_dt"], []).append(dc)

    sorted_dates = sorted(by_date.keys())
    max_day_step = g_max_days + 1  # g_max=2 hari tanpa deteksi berarti t dapat terhubung ke t+1, t+2, t+3

    for i, d_curr in enumerate(sorted_dates):
        for j in range(i + 1, len(sorted_dates)):
            d_next = sorted_dates[j]
            day_diff = (d_next - d_curr).days
            if day_diff > max_day_step:
                break
            for c1 in by_date[d_curr]:
                tree1 = cKDTree(c1["coords"])
                for c2 in by_date[d_next]:
                    dists, _ = tree1.query(c2["coords"], k=1)
                    if np.min(dists) <= eps_m:
                        G.add_edge(c1["node_id"], c2["node_id"])

    # Beri label event_id berdasarkan urutan waktu mulai dan ukuran
    components = list(nx.connected_components(G))
    node_to_event = {}
    for comp_idx, comp_nodes in enumerate(components):
        eid = f"EVT_{comp_idx:05d}"
        for nid in comp_nodes:
            node_to_event[nid] = eid

    data["event_id"] = data["daily_cluster_id"].map(node_to_event)

    # Ringkasan diagnostik per event (PRD Bagian 1.9)
    summary_rows = []
    for eid, grp in data.groupby("event_id"):
        d_min = grp["date_dt"].min()
        d_max = grp["date_dt"].max()
        span_days = (d_max - d_min).days + 1
        active_days = grp["acq_date"].nunique()
        center_lon = float(grp["longitude"].mean())
        center_lat = float(grp["latitude"].mean())
        utm_epsg = GridConfig.get_utm_epsg(center_lon, center_lat)

        # Hitung envelope dalam CRS UTM spesifik patch
        grp_gdf = gpd.GeoDataFrame(
            grp,
            geometry=gpd.points_from_xy(grp["longitude"], grp["latitude"]),
            crs="EPSG:4326",
        ).to_crs(epsg=utm_epsg)
        minx, miny, maxx, maxy = grp_gdf.total_bounds
        env_width_km = (maxx - minx) / 1000.0
        env_height_km = (maxy - miny) / 1000.0

        summary_rows.append({
            "event_id": eid,
            "year": int(d_min.year),
            "start_date": d_min.strftime("%Y-%m-%d"),
            "end_date": d_max.strftime("%Y-%m-%d"),
            "span_days": int(span_days),
            "active_days": int(active_days),
            "n_detections": int(len(grp)),
            "center_lon": round(center_lon, 5),
            "center_lat": round(center_lat, 5),
            "utm_epsg": int(utm_epsg),
            "envelope_width_km": round(env_width_km, 2),
            "envelope_height_km": round(env_height_km, 2),
            "exceeds_single_patch": bool(env_width_km > 96.0 or env_height_km > 96.0),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(["start_date", "n_detections"], ascending=[True, False]).reset_index(drop=True)
    return data, summary_df


def build_patch_roi_grid(
    center_lon: float,
    center_lat: float,
    grid_cfg: GridConfig = GridConfig(),
) -> Dict[str, object]:
    """
    Membentuk spesifikasi grid metrik presisi 256 x 256 (375 m x 375 m = 96.0 x 96.0 km)
    pada zona UTM lokal sesuai lokasi centroid event (PRD Bagian 1.2 & 1.10).
    """
    utm_epsg = grid_cfg.get_utm_epsg(center_lon, center_lat)
    pt_4326 = gpd.GeoSeries([Point(center_lon, center_lat)], crs="EPSG:4326")
    pt_utm = pt_4326.to_crs(epsg=utm_epsg).iloc[0]

    half_ext = grid_cfg.patch_extent_m / 2.0  # 48,000 m
    # Bulatkan origin ke kelipatan 375 m terdekat untuk konsistensi sub-piksel
    cx = round(pt_utm.x / grid_cfg.pixel_size_m) * grid_cfg.pixel_size_m
    cy = round(pt_utm.y / grid_cfg.pixel_size_m) * grid_cfg.pixel_size_m

    minx = cx - half_ext
    maxx = cx + half_ext
    miny = cy - half_ext
    maxy = cy + half_ext

    # Affine transform standar north-up: [scale_x, shear_x, trans_x, shear_y, scale_y, trans_y]
    # origin di pojok kiri-atas (minx, maxy)
    affine_tuple = (
        grid_cfg.pixel_size_m, 0.0, minx,
        0.0, -grid_cfg.pixel_size_m, maxy,
    )

    poly_utm = box(minx, miny, maxx, maxy)
    poly_4326 = gpd.GeoSeries([poly_utm], crs=f"EPSG:{utm_epsg}").to_crs("EPSG:4326").iloc[0]

    return {
        "utm_epsg": utm_epsg,
        "crs": f"EPSG:{utm_epsg}",
        "center_lon": center_lon,
        "center_lat": center_lat,
        "center_x_utm": cx,
        "center_y_utm": cy,
        "bounds_utm": (minx, miny, maxx, maxy),
        "bounds_4326": poly_4326.bounds,
        "width_px": grid_cfg.patch_size,
        "height_px": grid_cfg.patch_size,
        "pixel_size_m": grid_cfg.pixel_size_m,
        "affine_transform": affine_tuple,
        "polygon_utm": poly_utm,
        "polygon_4326": poly_4326,
    }


def rasterize_viirs_footprints_256(
    detections_df: pd.DataFrame,
    patch_spec: Dict[str, object],
    use_scan_track_footprint: bool = True,
) -> np.ndarray:
    """
    Merasterisasi titik deteksi VIIRS S-NPP 375 m ke matriks biner 256 x 256 (uint8).
    Mendukung mode titik pusat maupun mode footprint aktual `scan` x `track` (PRD Bagian 1.2).
    """
    H = patch_spec["height_px"]
    W = patch_spec["width_px"]
    mask = np.zeros((H, W), dtype=np.uint8)
    if len(detections_df) == 0:
        return mask

    utm_epsg = patch_spec["utm_epsg"]
    minx, miny, maxx, maxy = patch_spec["bounds_utm"]
    px_size = patch_spec["pixel_size_m"]

    gdf = gpd.GeoDataFrame(
        detections_df,
        geometry=gpd.points_from_xy(detections_df["longitude"], detections_df["latitude"]),
        crs="EPSG:4326",
    ).to_crs(epsg=utm_epsg)

    xs = gdf.geometry.x.values
    ys = gdf.geometry.y.values

    # Filter di dalam kotak patch
    in_box = (xs >= minx) & (xs < maxx) & (ys > miny) & (ys <= maxy)
    if not np.any(in_box):
        return mask

    xs = xs[in_box]
    ys = ys[in_box]
    sub_df = detections_df.iloc[np.where(in_box)[0]]

    # Indeks baris ( dari atas maxy ke bawah ) dan kolom ( dari kiri minx ke kanan )
    cols = np.floor((xs - minx) / px_size).astype(int)
    rows = np.floor((maxy - ys) / px_size).astype(int)
    cols = np.clip(cols, 0, W - 1)
    rows = np.clip(rows, 0, H - 1)

    if not use_scan_track_footprint or ("scan" not in sub_df.columns or "track" not in sub_df.columns):
        mask[rows, cols] = 1
        return mask

    # Jika atribut scan dan track (dalam meter atau km, standar FIRMS adalah km, misal 0.38 - 0.80 km) tersedia
    scans = pd.to_numeric(sub_df["scan"], errors="coerce").fillna(0.375).values
    tracks = pd.to_numeric(sub_df["track"], errors="coerce").fillna(0.375).values
    # Konversi km ke meter bila nilainya < 10
    scans_m = np.where(scans < 10.0, scans * 1000.0, scans)
    tracks_m = np.where(tracks < 10.0, tracks * 1000.0, tracks)

    half_c = np.floor((scans_m / 2.0) / px_size).astype(int)
    half_r = np.floor((tracks_m / 2.0) / px_size).astype(int)

    for r, c, hr, hc in zip(rows, cols, half_r, half_c):
        r0, r1 = max(0, r - hr), min(H, r + hr + 1)
        c0, c1 = max(0, c - hc), min(W, c + hc + 1)
        mask[r0:r1, c0:c1] = 1

    return mask


def compute_ca_initial_state_and_targets(
    daily_masks: Dict[str, np.ndarray],
    valid_masks: Dict[str, np.ndarray],
    date_t: str,
    event_start_date: str,
) -> Dict[str, np.ndarray]:
    """
    Menghitung state awal CA 3-keadaan [P_U, P_F, P_B] pada hari t,
    kanal riwayat api (fire_t, fire_t-1, fire_t-2, count_14d),
    serta target kumulatif C_{t+24} dan target pertumbuhan baru Y_{t+24}
    sesuai PRD Bagian 1.6.3 & 1.6.4.
    """
    dt = pd.to_datetime(date_t)
    dt_start = pd.to_datetime(event_start_date)
    H, W = next(iter(daily_masks.values())).shape

    def get_mask(d: pd.Timestamp) -> np.ndarray:
        return daily_masks.get(d.strftime("%Y-%m-%d"), np.zeros((H, W), dtype=np.uint8))

    fire_t = get_mask(dt).astype(np.float32)
    fire_tm1 = get_mask(dt - pd.Timedelta(days=1)).astype(np.float32)
    fire_tm2 = get_mask(dt - pd.Timedelta(days=2)).astype(np.float32)

    # count_14d: jumlah hari deteksi dalam [t-13, t]
    count_14d = np.zeros((H, W), dtype=np.float32)
    for k in range(14):
        count_14d += get_mask(dt - pd.Timedelta(days=k)).astype(np.float32)

    # Akumulasi C_t sejak awal event (t_0 .. t)
    C_t = np.zeros((H, W), dtype=np.uint8)
    curr_d = dt_start
    while curr_d <= dt:
        C_t = np.maximum(C_t, get_mask(curr_d))
        curr_d += pd.Timedelta(days=1)

    # State awal CA [P_U, P_F, P_B] (PRD 1.6.4):
    # - Terdeteksi pada hari t -> P_F = 1
    # - Terdeteksi pada hari sebelumnya dalam event (C_t == 1) dan tidak pada t -> P_B = 1
    # - Selainnya -> P_U = 1
    P_F = (fire_t > 0).astype(np.float32)
    P_B = ((C_t > 0) & (fire_t == 0)).astype(np.float32)
    P_U = 1.0 - np.clip(P_F + P_B, 0.0, 1.0)
    state_t = np.stack([P_U, P_F, P_B], axis=0)  # [3, 256, 256]

    # Target t+24
    dt_next = dt + pd.Timedelta(days=1)
    A_next = get_mask(dt_next)
    C_next = np.maximum(C_t, A_next)              # Kumulatif C_{t+24}
    Y_next = ((C_next > 0) & (C_t == 0)).astype(np.uint8)  # Baru Y_{t+24} = C_{t+24} \ C_t

    valid_t = valid_masks.get(date_t, np.ones((H, W), dtype=np.uint8))
    valid_next = valid_masks.get(dt_next.strftime("%Y-%m-%d"), np.ones((H, W), dtype=np.uint8))

    return {
        "fire_t": fire_t,
        "fire_t_minus_1": fire_tm1,
        "fire_t_minus_2": fire_tm2,
        "count_14d": count_14d,
        "valid_t": valid_t.astype(np.float32),
        "state_t": state_t,
        "target_cum": C_next[np.newaxis, ...],   # [1, 256, 256]
        "target_new": Y_next[np.newaxis, ...],   # [1, 256, 256]
        "valid_next": valid_next[np.newaxis, ...],  # [1, 256, 256]
    }

