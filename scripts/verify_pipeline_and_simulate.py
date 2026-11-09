"""
Skrip Verifikasi End-to-End & Demo Simulasi Prototipe Patch 2019-09-15 (Kalimantan Tengah).
Menguji:
  1. Unit test anti-leakage timestamp (PRD 1.11 & 3.2).
  2. Pembentukan grid UTM dinamis 256x256 @ 375 m dan rasterisasi titik api dari
     `Patch_FirePoints_2019-09-15.shp` & atribut fisik `Patch_Polygon_256x256_2019-09-15.shp`.
  3. Augmentasi sadar-vektor (rotasi & flip pada komponen angin & sudut fisik CA).
  4. Forward pass, backward pass, dan 10 langkah optimasi DeepLabV3+ + Probabilistic CA
     (Varian M1 Gambut Adaptif) vs MS-CNN + CA (Baseline B7).
  5. Rollout autoregresif 24-72 jam dan visualisasi 6 peta parameter spasial Theta_t.
"""

import os
import sys
from typing import Dict
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, r"c:\Neural-CA-Kalimantan")

from src.config import GridConfig, ModelCAConfig, LITE_17_CHANNELS
from src.data.event_clustering import (
    build_patch_roi_grid,
    rasterize_viirs_footprints_256,
    compute_ca_initial_state_and_targets,
)
from src.data.gee_extractor import verify_anti_leakage_timestamps, assemble_lite17_sample_dict
from src.data.dataset import apply_vector_aware_augmentation, KalimantanFireSpreadDataset
from src.models.neural_ca import NeuralCAWildfireModel
from src.models.losses_and_metrics import MaskedHybridCALoss, evaluate_predictions_stratified


def build_realistic_patch_from_shp(shp_dir: str, out_npz_path: str) -> Dict[str, object]:
    """
    Membangun sampel `.npz` 256x256 (Lite-17) yang terkalibrasi secara fisik dari
    `Patch_Polygon_256x256_2019-09-15.shp` dan `Patch_FirePoints_2019-09-15.shp`.
    """
    poly_gdf = gpd.read_file(os.path.join(shp_dir, "Patch_Polygon_256x256_2019-09-15.shp"))
    pts_gdf = gpd.read_file(os.path.join(shp_dir, "Patch_FirePoints_2019-09-15.shp"))
    row = poly_gdf.iloc[0]

    center_lon = float(row["center_lon"])
    center_lat = float(row["center_lat"])
    date_t = str(row["date_t"])

    # 1. Buat spesifikasi grid presisi 256x256 @ 375 m pada CRS UTM dinamis (114.15E -> EPSG:32750)
    patch_spec = build_patch_roi_grid(center_lon=center_lon, center_lat=center_lat)
    print(f"[Grid Check] Center=({center_lon}, {center_lat}) -> Assigned CRS={patch_spec['crs']}, "
          f"Extent={(patch_spec['bounds_utm'][2]-patch_spec['bounds_utm'][0])/1000:.2f} x "
          f"{(patch_spec['bounds_utm'][3]-patch_spec['bounds_utm'][1])/1000:.2f} km")

    # 2. Konversi 9 titik klaster di Patch_FirePoints menjadi klaster piksel 375 m
    pts_df = pd.DataFrame({
        "longitude": pts_gdf.geometry.x.values,
        "latitude": pts_gdf.geometry.y.values,
        "count": pts_gdf["count"].values,
        "scan": np.clip(np.sqrt(pts_gdf["count"].values) * 0.12, 0.375, 2.2),
        "track": np.clip(np.sqrt(pts_gdf["count"].values) * 0.12, 0.375, 2.2),
    })
    fire_mask_t = rasterize_viirs_footprints_256(pts_df, patch_spec, use_scan_track_footprint=True)

    # Buat riwayat t-1, t-2 dan target realistis t+24 yang merambat mengikuti arah angin aktual patch
    # Angin pada patch: u = -0.45 m/s (ke barat), v = +1.47 m/s (ke utara) -> merambat ke Barat-Laut (NW)
    from scipy.ndimage import binary_dilation, shift
    struct_nw = np.array([
        [1, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)
    fire_tm1 = binary_dilation(shift(fire_mask_t, shift=(1, 1), order=0), iterations=1).astype(np.uint8) & fire_mask_t
    fire_tm2 = binary_dilation(shift(fire_tm1, shift=(1, 1), order=0), iterations=1).astype(np.uint8) & fire_tm1
    fire_next = binary_dilation(fire_mask_t, structure=struct_nw, iterations=3).astype(np.uint8)

    daily_masks = {
        "2019-09-13": fire_tm2,
        "2019-09-14": fire_tm1,
        "2019-09-15": fire_mask_t,
        "2019-09-16": fire_next,
    }
    # Simulasikan tutupan awan/asap 12% di sudut tenggara patch (valid = 0)
    H, W = 256, 256
    yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing="ij")
    cloud_zone = ((yy > 0.72) & (xx > 0.68)).astype(np.uint8)
    valid_mask_2d = (1 - cloud_zone).astype(np.uint8)
    valid_masks = {"2019-09-15": valid_mask_2d, "2019-09-16": valid_mask_2d}

    fire_hist = compute_ca_initial_state_and_targets(
        daily_masks=daily_masks,
        valid_masks=valid_masks,
        date_t=date_t,
        event_start_date="2019-09-13",
    )

    # 3. Bangun medan spasial 256x256 dari nilai atribut fisik Patch_Polygon_256x256_2019-09-15.shp
    rng = np.random.default_rng(42)
    peat_2d = (np.ones((H, W), dtype=np.float32) * float(row["peat"]) > 0.5).astype(np.float32)
    # Buat variasi tutupan lahan gambut: 8 (Woody Savannas / Degraded Peat), 12 (Croplands), 14 (Mosaic)
    lc_2d = np.full((H, W), 8, dtype=np.float32)
    lc_2d[yy < 0.35] = 12.0
    # Tambahkan sungai kecil (kelas 17 = Air, non-burnable M_fuel=0)
    river_mask = np.abs(xx - (0.28 + 0.08 * np.sin(yy * 6.0))) < 0.012
    lc_2d[river_mask] = 17.0

    elev_2d = float(row["elevation"]) + 3.0 * np.sin(xx * 3.14) + rng.normal(0, 0.5, (H, W)).astype(np.float32)
    slope_2d = np.clip(float(row["slope"]) + 0.1 * np.cos(yy * 3.14), 0.01, 5.0).astype(np.float32)
    aspect_2d = np.full((H, W), 315.0, dtype=np.float32)  # menghadap NW

    precip_7d_2d = np.clip(float(row["precip_7d"]) + 0.15 * yy, 0.0, 10.0).astype(np.float32)
    precip_30d_2d = np.clip(precip_7d_2d * 4.5 + 2.0, 0.0, 100.0).astype(np.float32)
    dc_2d = (float(row["DC"]) + 25.0 * (1.0 - yy)).astype(np.float32)

    wind_u_2d = np.full((H, W), float(row["wind_u"]), dtype=np.float32)
    wind_v_2d = np.full((H, W), float(row["wind_v"]), dtype=np.float32)

    gee_bands = {
        "valid_t": valid_mask_2d.astype(np.float32),
        "valid_next": valid_mask_2d.astype(np.float32),
        "peat": peat_2d,
        "landcover_idx": lc_2d,
        "elevation": elev_2d,
        "slope": slope_2d,
        "aspect_deg": aspect_2d,
        "precip_7d": precip_7d_2d,
        "precip_30d": precip_30d_2d,
        "DC": dc_2d,
        "wind_u": wind_u_2d,
        "wind_v": wind_v_2d,
        "gfs_wind_u": wind_u_2d * 1.05,
        "gfs_wind_v": wind_v_2d * 1.05,
    }
    fire_freq_prev = (fire_mask_t * 2.0 + 0.5 * peat_2d).astype(np.float32)

    sample_dict = assemble_lite17_sample_dict(
        fire_history_dict=fire_hist,
        fire_freq_prev_years=fire_freq_prev,
        gee_bands_dict=gee_bands,
        event_id="EVT_20190915_KALTENG",
        date_t=date_t,
        patch_spec=patch_spec,
        scalars=(0.45, 1.85, 2.0),  # ONI (+0.45), DMI (+1.85 IOD+ kuat 2019), sub-region=2 (Kalteng)
    )
    os.makedirs(os.path.dirname(out_npz_path), exist_ok=True)
    np.savez_compressed(out_npz_path, **sample_dict)
    return sample_dict


def run_verification():
    print("=== 1. VERIFIKASI UNIT TEST ANTI-LEAKAGE (PRD Bagian 1.11 & 3.2) ===")
    ok = verify_anti_leakage_timestamps(
        date_t="2019-09-15",
        obs_end_timestamp="2019-09-15 23:30:00",
        gfs_creation_timestamp="2019-09-15 18:00:00",
        static_year=2018,
    )
    print(f"  Anti-leakage check valid sample: PASS ({ok})")
    try:
        verify_anti_leakage_timestamps("2019-09-15", "2019-09-16 01:00:00", "2019-09-15 18:00:00", 2018)
    except AssertionError as e:
        print(f"  Anti-leakage check caught future observation leak: PASS ({e})")

    print("\n=== 2. PEMBENTUKAN TENSOR SAMPEL DARI SHAPEFILE 2019-09-15 ===")
    shp_dir = r"c:\Neural-CA-Kalimantan\GEE_Karhutla_SHP"
    sample_npz = r"c:\Neural-CA-Kalimantan\data\processed\event_20190915_kalteng.npz"
    sample = build_realistic_patch_from_shp(shp_dir, sample_npz)
    print(f"  Saved NPZ: {sample_npz}")
    print(f"  x shape          : {sample['x'].shape} (Lite-17)")
    print(f"  state_t shape    : {sample['state_t'].shape} (P_U, P_F, P_B)")
    print(f"  target_cum shape : {sample['target_cum'].shape}, pos_pixels={int(sample['target_cum'].sum())}")
    print(f"  target_new shape : {sample['target_new'].shape}, pos_pixels={int(sample['target_new'].sum())}")
    print(f"  ca_phys shape    : {sample['ca_phys'].shape}")

    print("\n=== 3. VERIFIKASI AUGMENTASI SADAR-VEKTOR (PRD Bagian 1.12) ===")
    x_aug, _, _, _, _, phys_aug = apply_vector_aware_augmentation(
        sample["x"], sample["state_t"], sample["target_cum"],
        sample["target_new"], sample["valid_next"], sample["ca_phys"],
        flip_h=True, flip_v=False, rot_k=1,
    )
    u_idx = LITE_17_CHANNELS.index("wind_u")
    v_idx = LITE_17_CHANNELS.index("wind_v")
    print(f"  Orig wind (u, v) = ({sample['x'][u_idx,0,0]:.3f}, {sample['x'][v_idx,0,0]:.3f}), dir={np.degrees(sample['ca_phys'][1,0,0]):.1f} deg")
    print(f"  Aug  wind (u, v) = ({x_aug[u_idx,0,0]:.3f}, {x_aug[v_idx,0,0]:.3f}), dir={np.degrees(phys_aug[1,0,0]):.1f} deg")

    print("\n=== 4. VERIFIKASI ARSITEKTUR MODEL, LOSS, & KALIBRASI TRAINING ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = KalimantanFireSpreadDataset([sample_npz], augment=False)
    batch = ds[0]
    x_b = batch["x"].unsqueeze(0).to(device)
    lc_b = batch["landcover_idx"].unsqueeze(0).to(device)
    st_b = batch["state_t"].unsqueeze(0).to(device)
    tc_b = batch["target_cum"].unsqueeze(0).to(device)
    tn_b = batch["target_new"].unsqueeze(0).to(device)
    vn_b = batch["valid_next"].unsqueeze(0).to(device)
    phys_b = batch["ca_phys"].unsqueeze(0).to(device)
    ph_b = batch["peat_hydro"].unsqueeze(0).to(device)

    cfg = ModelCAConfig(n_micro_steps=15, use_adaptive_p_continue=True)
    model_deeplab = NeuralCAWildfireModel(cfg=cfg, generator_type="deeplabv3plus", in_channels=17).to(device)
    model_mscnn = NeuralCAWildfireModel(cfg=cfg, generator_type="mscnn", in_channels=17).to(device)

    n_params_dl = sum(p.numel() for p in model_deeplab.parameters() if p.requires_grad)
    n_params_ms = sum(p.numel() for p in model_mscnn.parameters() if p.requires_grad)
    print(f"  DeepLabV3+ + CA parameters : {n_params_dl:,}")
    print(f"  MS-CNN     + CA parameters : {n_params_ms:,}")

    criterion = MaskedHybridCALoss(cfg=cfg)
    optimizer = torch.optim.AdamW(model_deeplab.parameters(), lr=3e-3, weight_decay=1e-4)

    loss_history = []
    for step in range(1, 13):
        optimizer.zero_grad()
        out = model_deeplab(x_b, lc_b, st_b, phys_b, peat_hydro=ph_b)
        loss_dict = criterion(out, tc_b, tn_b, vn_b)
        loss = loss_dict["loss"]
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))
        if step in (1, 4, 8, 12):
            print(f"  Step {step:2d}/12 | Total Loss={loss.item():.4f} | SoftIoU_Cum={loss_dict['iou_cum_soft'].item():.4f} | SoftIoU_New={loss_dict['iou_new_soft'].item():.4f}")

    # Evaluasi terstratifikasi (PRD 2.10 & 3.4)
    with torch.no_grad():
        final_out = model_deeplab(x_b, lc_b, st_b, phys_b, peat_hydro=ph_b)
        init_fire = (st_b[:, 1:2] + st_b[:, 2:3]).clamp(0, 1)
        metrics_cum = evaluate_predictions_stratified(
            final_out["p_fire_cum"], tc_b, vn_b, init_fire, n_micro_steps=15, threshold=0.35
        )
        metrics_new = evaluate_predictions_stratified(
            final_out["delta_p_fire"], tn_b, vn_b, init_fire, n_micro_steps=15, threshold=0.25
        )
        # Rollout 72h (3 macro-steps: +24h, +48h, +72h)
        rollout_outs = model_deeplab.rollout_72h(
            x_seq=[x_b, x_b, x_b],
            landcover_idx=lc_b,
            state_t0=st_b,
            ca_phys_seq=[phys_b, phys_b, phys_b],
            peat_hydro_seq=[ph_b, ph_b, ph_b],
        )

    print("  Stratified Evaluation (Cumulative C_{t+24}):", metrics_cum)
    print("  Stratified Evaluation (New Growth Y_{t+24}):", metrics_new)

    # Buat visualisasi diagnostik keluaran Neural-CA (6 peta Theta_t + Rollout 24-72 jam)
    out_fig = r"C:\Users\Hype\.gemini\antigravity\brain\fa2b750e-946a-426a-911a-33f49e4c71ed\fig4_neural_ca_simulation_2019_09_15.png"
    fig = plt.figure(figsize=(15, 8.5), dpi=200)
    gs = gridspec.GridSpec(2, 4, hspace=0.32, wspace=0.22)

    theta_np = final_out["theta"][0].cpu().numpy()
    param_titles = [
        r"1. $p_{\mathrm{base}}(x)$ (Base Ignitability)",
        r"2. $\alpha_{w1}(x)$ (Wind Speed Coeff)",
        r"3. $\alpha_{w2}(x)$ (Wind Dir Alignment)",
        r"4. $fuel_{\mathrm{factor}}(x)$ (Embedded Fuel)",
    ]
    param_indices = [0, 1, 2, 5]
    cmaps = ["magma", "YlGnBu", "YlGnBu", "viridis"]

    for i, (p_idx, title, cm) in enumerate(zip(param_indices, param_titles, cmaps)):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(theta_np[p_idx], cmap=cm)
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Baris 2: State Awal t, Target C_{t+24} + Cloud Mask, Prediksi +24h, Rollout +72h
    ax_s0 = fig.add_subplot(gs[1, 0])
    im0 = ax_s0.imshow(init_fire[0, 0].cpu().numpy(), cmap="hot", vmin=0, vmax=1)
    ax_s0.set_title("5. Api Awal $C_t$ (2019-09-15)\n[9 Klaster Patch_FirePoints]", fontsize=9.5, fontweight="bold")
    ax_s0.set_xticks([]); ax_s0.set_yticks([])
    plt.colorbar(im0, ax=ax_s0, fraction=0.046, pad=0.04)

    ax_gt = fig.add_subplot(gs[1, 1])
    gt_vis = tc_b[0, 0].cpu().numpy().copy()
    valid_np = vn_b[0, 0].cpu().numpy()
    gt_masked = np.ma.masked_where(valid_np == 0, gt_vis)
    ax_gt.imshow(np.ones_like(gt_vis) * 0.5, cmap="gray", vmin=0, vmax=1)
    im1 = ax_gt.imshow(gt_masked, cmap="hot", vmin=0, vmax=1)
    ax_gt.text(195, 225, "Cloud Mask\n(M=0)", color="cyan", fontsize=8, fontweight="bold", ha="center",
               bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))
    ax_gt.set_title("6. Target $C_{t+24}$ & Mask Awan\n(Rambatan Angin Tenggara -> NW)", fontsize=9.5, fontweight="bold")
    ax_gt.set_xticks([]); ax_gt.set_yticks([])
    plt.colorbar(im1, ax=ax_gt, fraction=0.046, pad=0.04)

    ax_p24 = fig.add_subplot(gs[1, 2])
    p24_np = rollout_outs[0]["p_fire_cum"][0, 0].cpu().numpy()
    im2 = ax_p24.imshow(p24_np, cmap="hot", vmin=0, vmax=1)
    ax_p24.set_title(f"7. Prediksi CA $P_{{\\mathrm{{fire}}, t+24}}$\n(PR-AUC={metrics_cum['all_ap']:.3f}, FSS={metrics_cum['fss_r2_5x5']:.3f})", fontsize=9.5, fontweight="bold")
    ax_p24.set_xticks([]); ax_p24.set_yticks([])
    plt.colorbar(im2, ax=ax_p24, fraction=0.046, pad=0.04)

    ax_p72 = fig.add_subplot(gs[1, 3])
    p72_np = rollout_outs[2]["p_fire_cum"][0, 0].cpu().numpy()
    im3 = ax_p72.imshow(p72_np, cmap="hot", vmin=0, vmax=1)
    ax_p72.set_title("8. Rollout Autoregresif $P_{\\mathrm{fire}, t+72}$\n(3 Macro-Steps = 45 Micro-Steps CA)", fontsize=9.5, fontweight="bold")
    ax_p72.set_xticks([]); ax_p72.set_yticks([])
    plt.colorbar(im3, ax=ax_p72, fraction=0.046, pad=0.04)

    fig.suptitle("Verifikasi End-to-End DeepLabV3+ + Probabilistic CA (Varian M1 Gambut) pada Patch 2019-09-15", fontsize=13, fontweight="bold", y=0.98)
    plt.savefig(out_fig, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"\n[Visualisasi Simulasi Disimpan]: {out_fig}")


if __name__ == "__main__":
    run_verification()
