"""
Modul Baseline Lengkap & Metrik Spasial/Arah/Kalibrasi Probabilitas sesuai PRD v2:
  - Bagian 3.3 (Perbandingan Model B0 - B7, P, P+):
      * B0: Persistence (batas bawah wajib)
      * B1: Dilasi Morfologis Berbias Angin (baseline fisik sederhana)
      * B4: U-Net (ResNet18) Segmentasi Langsung (tanpa CA)
      * B5: DeepLabV3+ Segmentasi Langsung (pemisahan efek CA vs P)
      * B6: CA dengan Parameter Konstan Global (efek parameterisasi spasial)
      * B7: Capacity-Matched MS-CNN + CA (kontrol kapasitas adil vs P)
  - Bagian 3.5 (Metrik Spasial & Arah):
      * Relative Area Error, Boundary-F1 dengan buffer toleransi, 95th-Percentile Hausdorff Distance (HD95),
        Centroid Shift Vector Error (km), Spread Direction Angle Error (deg), Front Spread Rate Error (m/hari).
  - Bagian 3.9 (Kalibrasi Probabilitas):
      * Reliability Diagram (Expected Calibration Error / ECE), Sharpness, Post-Hoc Temperature Scaling.
"""

from typing import Dict, List, Optional, Tuple
import math
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelCAConfig
from src.models.neural_ca import (
    CategoricalLandcoverEmbedding,
    BasicBlock2D,
    make_resnet_layer,
    ASPP,
    ProbabilisticCellularAutomata,
)


# =========================================================================
# 1. BASELINE MODELS (B0, B1, B4, B5, B6) — PRD Bagian 3.3 & 3.8
# =========================================================================

class PersistenceBaseline(nn.Module):
    """
    B0: Baseline Persistence (batas bawah wajib PRD 1.6.3 & 3.3).
    Memprediksi C_{t+24} = C_t (tidak ada pertumbuhan baru: delta_P_fire = 0).
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        landcover_idx: torch.Tensor,
        state_t: torch.Tensor,
        ca_phys: torch.Tensor,
        peat_hydro: Optional[torch.Tensor] = None,
        n_micro_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        p_fire_cum = (state_t[:, 1:2] + state_t[:, 2:3]).clamp(0.0, 1.0)
        delta_p_fire = torch.zeros_like(p_fire_cum)
        B, _, H, W = p_fire_cum.shape
        dummy_theta = torch.zeros((B, 6, H, W), device=state_t.device, dtype=state_t.dtype)
        return {
            "theta": dummy_theta,
            "state_next": state_t,
            "p_fire_cum": p_fire_cum,
            "delta_p_fire": delta_p_fire,
            "p_ignite": delta_p_fire,
        }


class WindBiasedMorphologicalDilationBaseline(nn.Module):
    """
    B1: Dilasi Morfologis dengan Bias Arah Angin & Mask Bahan Bakar (PRD Bagian 3.3).
    Menggunakan parameter skalar yang dapat dikalibrasi untuk laju dilasi sesuai arah angin.
    """

    def __init__(self, n_steps: int = 3):
        super().__init__()
        self.n_steps = n_steps
        self.base_rate_logit = nn.Parameter(torch.tensor(-0.5))
        self.wind_gain = nn.Parameter(torch.tensor(0.35))

    def forward(
        self,
        x: torch.Tensor,
        landcover_idx: torch.Tensor,
        state_t: torch.Tensor,
        ca_phys: torch.Tensor,
        peat_hydro: Optional[torch.Tensor] = None,
        n_micro_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        init_fire = (state_t[:, 1:2] + state_t[:, 2:3]).clamp(0.0, 1.0)
        curr = init_fire.clone()
        V = ca_phys[:, 0:1].clamp(0.0, 30.0)
        M_fuel = ca_phys[:, 4:5].clamp(0.0, 1.0)

        rate = torch.sigmoid(self.base_rate_logit + self.wind_gain * (V / 10.0)) * M_fuel
        for _ in range(self.n_steps):
            dil = F.max_pool2d(curr, kernel_size=3, stride=1, padding=1)
            growth = (dil - curr).clamp(0.0, 1.0) * rate
            curr = (curr + growth).clamp(0.0, 1.0)

        delta = (curr - init_fire).clamp(0.0, 1.0)
        B, _, H, W = curr.shape
        dummy_theta = rate.repeat(1, 6, 1, 1)
        return {
            "theta": dummy_theta,
            "state_next": state_t,
            "p_fire_cum": curr,
            "delta_p_fire": delta,
            "p_ignite": delta,
        }


class GlobalConstantCAParameterGenerator(nn.Module):
    """
    Generator untuk B6: CA dengan Parameter Konstan Global (PRD Bagian 3.3).
    Mempelajari hanya 6 skalar global [p_base, alpha_w1, alpha_w2, alpha_s, gamma, fuel_factor]
    untuk menguji kontribusi parameterisasi bervariasi spasial pada P (DeepLabV3+ + CA).
    """

    def __init__(self):
        super().__init__()
        self.raw_global = nn.Parameter(torch.zeros(1, 6, 1, 1))

    def forward(
        self,
        x: torch.Tensor,
        landcover_idx: torch.Tensor,
        lc_channel_pos: int = 7,
    ) -> Dict[str, torch.Tensor]:
        B, _, H, W = x.shape
        raw = self.raw_global.expand(B, 6, H, W)
        p_base = 0.001 + 0.949 * torch.sigmoid(raw[:, 0:1] - 1.5)
        alpha_w1 = 0.25 * torch.sigmoid(raw[:, 1:2] - 1.0)
        alpha_w2 = 0.40 * torch.sigmoid(raw[:, 2:3])
        alpha_s = 2.00 * torch.sigmoid(raw[:, 3:4] - 0.5)
        gamma = 0.05 + 3.95 * torch.sigmoid(raw[:, 4:5] - 0.5)
        fuel_factor = 0.05 + 2.95 * torch.sigmoid(raw[:, 5:6])
        theta = torch.cat([p_base, alpha_w1, alpha_w2, alpha_s, gamma, fuel_factor], dim=1)
        return {
            "theta": theta,
            "p_base": p_base,
            "alpha_w1": alpha_w1,
            "alpha_w2": alpha_w2,
            "alpha_s": alpha_s,
            "gamma": gamma,
            "fuel_factor": fuel_factor,
        }


class GlobalConstantCAModel(nn.Module):
    """B6: CA dengan Parameter Konstan Global + Probabilistic CA."""

    def __init__(self, cfg: ModelCAConfig = ModelCAConfig()):
        super().__init__()
        self.generator = GlobalConstantCAParameterGenerator()
        self.ca = ProbabilisticCellularAutomata(cfg=cfg)

    def forward(
        self,
        x: torch.Tensor,
        landcover_idx: torch.Tensor,
        state_t: torch.Tensor,
        ca_phys: torch.Tensor,
        peat_hydro: Optional[torch.Tensor] = None,
        n_micro_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        param_dict = self.generator(x, landcover_idx)
        ca_out = self.ca(state_t, param_dict, ca_phys, peat_hydro=peat_hydro, n_micro_steps=n_micro_steps)
        return {**param_dict, **ca_out}


class DirectDeepLabV3PlusSegmentation(nn.Module):
    """
    B5: DeepLabV3+ Segmentasi Langsung tanpa mekanisme Cellular Automata (PRD Bagian 3.3).
    Menggunakan arsitektur encoder-ASPP-decoder yang identik dengan generator P, tetapi
    langsung memprediksi probabilitas pertumbuhan baru delta_P_fire(x) = sigmoid(head(feat)).
    """

    def __init__(
        self,
        in_channels: int = 17,
        num_lc_classes: int = 18,
        lc_embed_dim: int = 4,
        aspp_out: int = 128,
    ):
        super().__init__()
        self.lc_embed = CategoricalLandcoverEmbedding(num_lc_classes, lc_embed_dim)
        # Menerima input yang sama persis + 3 kanal state_t (PRD 3.3: "Agar adil, B5 menerima input yang sama termasuk S_t")
        eff_in_channels = (in_channels - 1) + lc_embed_dim + 3

        self.stem = nn.Sequential(
            nn.Conv2d(eff_in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = make_resnet_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = make_resnet_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = make_resnet_layer(128, 256, num_blocks=2, stride=2)
        self.aspp = ASPP(in_channels=256, atrous_rates=(2, 4, 6), out_channels=aspp_out)
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(aspp_out + 32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.seg_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        landcover_idx: torch.Tensor,
        state_t: torch.Tensor,
        ca_phys: torch.Tensor,
        peat_hydro: Optional[torch.Tensor] = None,
        n_micro_steps: Optional[int] = None,
        lc_channel_pos: int = 7,
    ) -> Dict[str, torch.Tensor]:
        lc_emb = self.lc_embed(landcover_idx)
        x_without_lc = torch.cat([x[:, :lc_channel_pos], x[:, lc_channel_pos + 1:]], dim=1)
        x_in = torch.cat([x_without_lc, lc_emb, state_t], dim=1)

        input_size = x.shape[-2:]
        s = self.stem(x_in)
        low_feat = self.layer1(s)
        x2 = self.layer2(low_feat)
        high_feat = self.layer3(x2)

        aspp_feat = self.aspp(high_feat)
        aspp_up = F.interpolate(aspp_feat, size=low_feat.shape[-2:], mode="bilinear", align_corners=False)
        low_proj = self.low_level_proj(low_feat)
        dec = self.decoder(torch.cat([aspp_up, low_proj], dim=1))
        dec_full = F.interpolate(dec, size=input_size, mode="bilinear", align_corners=False)

        M_fuel = ca_phys[:, 4:5].clamp(0.0, 1.0)
        init_fire = (state_t[:, 1:2] + state_t[:, 2:3]).clamp(0.0, 1.0)
        p_U = state_t[:, 0:1]

        delta_p_fire = torch.sigmoid(self.seg_head(dec_full)) * p_U * M_fuel
        p_fire_cum = (init_fire + delta_p_fire).clamp(0.0, 1.0)
        dummy_theta = dec_full[:, :6]

        return {
            "theta": dummy_theta,
            "state_next": torch.cat([1.0 - p_fire_cum, delta_p_fire, init_fire], dim=1),
            "p_fire_cum": p_fire_cum,
            "delta_p_fire": delta_p_fire,
            "p_ignite": delta_p_fire,
        }


class DirectUNetResNet18Segmentation(nn.Module):
    """
    B4: U-Net (ResNet18) Segmentasi Langsung sesuai baseline WildfireSpreadTS [Gerard2023] (PRD Bagian 3.3).
    """

    def __init__(
        self,
        in_channels: int = 17,
        num_lc_classes: int = 18,
        lc_embed_dim: int = 4,
    ):
        super().__init__()
        self.lc_embed = CategoricalLandcoverEmbedding(num_lc_classes, lc_embed_dim)
        eff_in = (in_channels - 1) + lc_embed_dim + 3

        self.enc1 = nn.Sequential(
            nn.Conv2d(eff_in, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.enc2 = make_resnet_layer(64, 64, num_blocks=2, stride=2)    # 128x128
        self.enc3 = make_resnet_layer(64, 128, num_blocks=2, stride=2)   # 64x64
        self.enc4 = make_resnet_layer(128, 256, num_blocks=2, stride=2)  # 32x32

        self.up3 = nn.Sequential(nn.Conv2d(256 + 128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.Conv2d(128 + 64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.up1 = nn.Sequential(nn.Conv2d(64 + 64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.seg_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        landcover_idx: torch.Tensor,
        state_t: torch.Tensor,
        ca_phys: torch.Tensor,
        peat_hydro: Optional[torch.Tensor] = None,
        n_micro_steps: Optional[int] = None,
        lc_channel_pos: int = 7,
    ) -> Dict[str, torch.Tensor]:
        lc_emb = self.lc_embed(landcover_idx)
        x_without_lc = torch.cat([x[:, :lc_channel_pos], x[:, lc_channel_pos + 1:]], dim=1)
        x_in = torch.cat([x_without_lc, lc_emb, state_t], dim=1)

        e1 = self.enc1(x_in)  # 256
        e2 = self.enc2(e1)    # 128
        e3 = self.enc3(e2)    # 64
        e4 = self.enc4(e3)    # 32

        d3 = self.up3(torch.cat([F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], dim=1))
        d2 = self.up2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], dim=1))
        d1 = self.up1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], dim=1))

        M_fuel = ca_phys[:, 4:5].clamp(0.0, 1.0)
        init_fire = (state_t[:, 1:2] + state_t[:, 2:3]).clamp(0.0, 1.0)
        p_U = state_t[:, 0:1]

        delta_p_fire = torch.sigmoid(self.seg_head(d1)) * p_U * M_fuel
        p_fire_cum = (init_fire + delta_p_fire).clamp(0.0, 1.0)
        return {
            "theta": d1[:, :6],
            "state_next": torch.cat([1.0 - p_fire_cum, delta_p_fire, init_fire], dim=1),
            "p_fire_cum": p_fire_cum,
            "delta_p_fire": delta_p_fire,
            "p_ignite": delta_p_fire,
        }


# =========================================================================
# 2. METRIK SPASIAL, GEOMETRI, ARAH, & KALIBRASI (PRD Bagian 3.5 & 3.9)
# =========================================================================

def compute_spatial_and_directional_metrics(
    pred_bin_2d: np.ndarray,
    target_bin_2d: np.ndarray,
    init_fire_2d: np.ndarray,
    valid_2d: np.ndarray,
    pixel_size_m: float = 375.0,
    boundary_buffer_px: int = 1,
) -> Dict[str, float]:
    """
    Menghitung metrik geometri, batas, dan dinamika arah rambatan sesuai PRD Bagian 3.5:
      1. Relative Area Error: |A_pred - A_obs| / max(A_obs, 1)
      2. Boundary-F1 dengan buffer toleransi `boundary_buffer_px` (default 1 piksel = 375 m)
      3. HD95 (95th-Percentile Hausdorff Distance dalam meter)
      4. Centroid Shift Error (km) & Spread Direction Angle Error (derajat) dari C_t ke C_{t+24}
      5. Front95 Spread Rate Error (m/hari): selisih persentil ke-95 jarak piksel baru ke C_t
    """
    v = (valid_2d > 0).astype(bool)
    p = (pred_bin_2d > 0) & v
    t = (target_bin_2d > 0) & v
    c0 = (init_fire_2d > 0) & v

    # 1. Relative Area Error
    a_pred = float(np.sum(p))
    a_obs = float(np.sum(t))
    rel_area_err = abs(a_pred - a_obs) / max(a_obs, 1.0)

    # Ekstrak piksel batas (boundary = mask \ eroded(mask))
    p_bound = p ^ binary_erosion(p)
    t_bound = t ^ binary_erosion(t)

    if np.any(p_bound) and np.any(t_bound):
        dist_to_t = distance_transform_edt(~t_bound)
        dist_to_p = distance_transform_edt(~p_bound)
        d_pt = dist_to_t[p_bound]
        d_tp = dist_to_p[t_bound]

        # 2. Boundary-F1 dengan buffer
        b_prec = float(np.mean(d_pt <= boundary_buffer_px))
        b_rec = float(np.mean(d_tp <= boundary_buffer_px))
        boundary_f1 = (2.0 * b_prec * b_rec) / max(b_prec + b_rec, 1e-8)

        # 3. HD95 dalam meter
        hd95_px = max(float(np.percentile(d_pt, 95)), float(np.percentile(d_tp, 95)))
        hd95_m = hd95_px * pixel_size_m
    else:
        boundary_f1 = 0.0
        hd95_m = float("nan")

    # 4. Pergeseran Centroid & Sudut Arah Spread (dari C_t ke pertumbuhan baru)
    p_new = p & (~c0)
    t_new = t & (~c0)

    if np.any(c0) and np.any(p_new) and np.any(t_new):
        r0, col0 = np.mean(np.argwhere(c0), axis=0)
        rp, colp = np.mean(np.argwhere(p_new), axis=0)
        rt, colt = np.mean(np.argwhere(t_new), axis=0)

        # Vektor pergeseran (dx = col - col0, dy = -(row - row0) agar utara positif)
        vec_p = np.array([(colp - col0) * pixel_size_m, -(rp - r0) * pixel_size_m])
        vec_t = np.array([(colt - col0) * pixel_size_m, -(rt - r0) * pixel_size_m])

        centroid_err_km = float(np.linalg.norm(vec_p - vec_t) / 1000.0)
        ang_p = math.degrees(math.atan2(vec_p[1], vec_p[0]))
        ang_t = math.degrees(math.atan2(vec_t[1], vec_t[0]))
        angle_diff = abs((ang_p - ang_t + 180.0) % 360.0 - 180.0)

        # 5. Laju ekspansi front (persentil ke-90 jarak piksel baru ke C_t dalam m/hari)
        dist_from_c0_m = distance_transform_edt(~c0) * pixel_size_m
        rate_pred_m = float(np.percentile(dist_from_c0_m[p_new], 90))
        rate_obs_m = float(np.percentile(dist_from_c0_m[t_new], 90))
        rate_err_m = abs(rate_pred_m - rate_obs_m)
    else:
        centroid_err_km = 0.0
        angle_diff = 0.0
        rate_pred_m = 0.0
        rate_obs_m = 0.0
        rate_err_m = 0.0

    return {
        "rel_area_error": round(rel_area_err, 4),
        "boundary_f1_buf1px": round(boundary_f1, 4),
        "hd95_m": round(hd95_m, 1) if np.isfinite(hd95_m) else -1.0,
        "centroid_shift_error_km": round(centroid_err_km, 3),
        "spread_angle_error_deg": round(angle_diff, 2),
        "front_rate_p90_pred_m": round(rate_pred_m, 1),
        "front_rate_p90_obs_m": round(rate_obs_m, 1),
        "front_rate_error_m": round(rate_err_m, 1),
    }


def compute_calibration_and_temperature_scaling(
    probs: np.ndarray,
    targets: np.ndarray,
    valid_mask: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, object]:
    """
    Mengevaluasi kalibrasi probabilitas mean-field CA sesuai PRD Bagian 3.9:
      - Brier Score
      - Expected Calibration Error (ECE) & kurva Reliability Diagram (bin confidence vs empirical frequency)
      - Post-hoc Temperature Scaling (mencari T optimal pada logit probabilitas)
    """
    v = (valid_mask > 0).reshape(-1)
    p = np.clip(probs.reshape(-1)[v], 1e-6, 1.0 - 1e-6)
    y = targets.reshape(-1)[v].astype(np.float32)

    brier_raw = float(np.mean((p - y) ** 2))
    logits = np.log(p / (1.0 - p))

    # Optimasi Temperature Scaling T in [0.2, 5.0] untuk meminimalkan NLL / Brier
    best_T = 1.0
    best_brier = brier_raw
    for T_cand in np.linspace(0.25, 3.5, 66):
        p_cal = 1.0 / (1.0 + np.exp(-logits / T_cand))
        b_cand = float(np.mean((p_cal - y) ** 2))
        if b_cand < best_brier:
            best_brier = b_cand
            best_T = float(T_cand)

    p_calibrated = 1.0 / (1.0 + np.exp(-logits / best_T))

    def _reliability_bins(prob_arr: np.ndarray) -> Tuple[float, List[float], List[float], List[int]]:
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        bin_confs, bin_accs, bin_counts = [], [], []
        N = len(prob_arr)
        for i in range(n_bins):
            in_bin = (prob_arr >= bins[i]) & (prob_arr < bins[i + 1] if i < n_bins - 1 else prob_arr <= bins[i + 1])
            cnt = int(np.sum(in_bin))
            bin_counts.append(cnt)
            if cnt > 0:
                conf = float(np.mean(prob_arr[in_bin]))
                acc = float(np.mean(y[in_bin]))
                ece += (cnt / N) * abs(acc - conf)
                bin_confs.append(conf)
                bin_accs.append(acc)
            else:
                bin_confs.append(float(0.5 * (bins[i] + bins[i + 1])))
                bin_accs.append(0.0)
        return float(ece), bin_confs, bin_accs, bin_counts

    ece_raw, conf_raw, acc_raw, cnt_raw = _reliability_bins(p)
    ece_cal, conf_cal, acc_cal, _ = _reliability_bins(p_calibrated)

    return {
        "brier_raw": round(brier_raw, 5),
        "brier_calibrated": round(best_brier, 5),
        "ece_raw": round(ece_raw, 5),
        "ece_calibrated": round(ece_cal, 5),
        "optimal_temperature": round(best_T, 3),
        "bin_conf_raw": conf_raw,
        "bin_acc_raw": acc_raw,
        "bin_conf_cal": conf_cal,
        "bin_acc_cal": acc_cal,
        "bin_counts": cnt_raw,
    }

