"""
Modul Fungsi Loss Ter-masking (Masked Hybrid CA Loss) dan Evaluasi Terstratifikasi
('Dapat Dijangkau CA' vs 'Tak Terjangkau / Ignisi Baru') sesuai PRD Bagian 2.10, 2.12, 3.4, 3.5.
"""

from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from src.config import ModelCAConfig


class MaskedHybridCALoss(nn.Module):
    """
    Implementasi fungsi loss rujukan [Zhenirovskyy2026] yang dimodifikasi untuk label
    VIIRS berbintik dan tutupan awan/asap Kalimantan sesuai PRD Bagian 2.12:
      L = w_BCE * L_frontierBCE + w_MSE * L_MSE + w_area * L_area + w_IoU * L_IoU + w_pool * L_pool + w_TV * L_TV
    Seluruh suku dikalikan mask validitas M_{t+24} sehingga piksel tertutup awan/air
    tidak memberi penalti palsu.
    """

    def __init__(self, cfg: ModelCAConfig = ModelCAConfig(), use_morphological_closing: bool = True):
        super().__init__()
        self.cfg = cfg
        self.use_morphological_closing = use_morphological_closing

    @staticmethod
    def _morph_dilate_3x3(x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)

    @staticmethod
    def _morph_erode_3x3(x: torch.Tensor) -> torch.Tensor:
        return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)

    def _compute_frontier_mask(self, target: torch.Tensor) -> torch.Tensor:
        """
        Menghitung pita frontier pada target berbintik VIIRS (PRD Bagian 2.12 poin 2):
        Jika use_morphological_closing=True, lakukan closing 3x3 (dilasi diikuti erosi)
        terlebih dahulu untuk menyatukan piksel deteksi AF yang berdekatan, lalu ambil batas 3x3-nya.
        """
        base = target
        if self.use_morphological_closing:
            base = self._morph_erode_3x3(self._morph_dilate_3x3(target))
        dilated = self._morph_dilate_3x3(base)
        eroded = self._morph_erode_3x3(base)
        frontier = (dilated - eroded).clamp(0.0, 1.0)
        return frontier

    @staticmethod
    def total_variation_loss(theta: torch.Tensor) -> torch.Tensor:
        """Regularisasi kelancaran spasial pada ke-6 peta parameter CA (PRD Bagian 2.3)."""
        dh = torch.abs(theta[:, :, 1:, :] - theta[:, :, :-1, :]).mean()
        dw = torch.abs(theta[:, :, :, 1:] - theta[:, :, :, :-1]).mean()
        return dh + dw

    def _single_target_loss(
        self,
        pred_prob: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        eps = 1e-6
        valid = valid_mask.float().clamp(0.0, 1.0)
        n_valid = valid.sum().clamp(min=1.0)

        p = pred_prob.clamp(eps, 1.0 - eps)
        t = target.float().clamp(0.0, 1.0)

        # 1. Frontier-weighted BCE (bobot w_f = 10 pada pita frontier)
        frontier = self._compute_frontier_mask(t)
        weights = (1.0 + (self.cfg.frontier_weight - 1.0) * frontier) * valid
        bce_raw = F.binary_cross_entropy(p, t, reduction="none")
        l_frontier_bce = (bce_raw * weights).sum() / weights.sum().clamp(min=1.0)

        # 2. Masked MSE
        l_mse = (((p - t) ** 2) * valid).sum() / n_valid

        # 3. Masked Relative Area Error: |A_pred - A_obs| / (A_obs + 1) per sample di dalam batch
        B = p.shape[0]
        p_flat = (p * valid).view(B, -1)
        t_flat = (t * valid).view(B, -1)
        area_pred = p_flat.sum(dim=1)
        area_obs = t_flat.sum(dim=1)
        l_area = (torch.abs(area_pred - area_obs) / (area_obs + 10.0)).mean()

        # 4. Masked Soft IoU (Jaccard) Loss
        inter = (p_flat * t_flat).sum(dim=1)
        union = (p_flat + t_flat - p_flat * t_flat).sum(dim=1)
        soft_iou = (inter + 1.0) / (union + 1.0)
        l_iou = (1.0 - soft_iou).mean()

        # 5. Multi-scale Pooled MSE (4x4 average pooling untuk toleransi pergeseran geolokasi ±1-2 piksel)
        p_pool = F.avg_pool2d(p * valid, kernel_size=4, stride=4)
        t_pool = F.avg_pool2d(t * valid, kernel_size=4, stride=4)
        v_pool = F.avg_pool2d(valid, kernel_size=4, stride=4).clamp(min=eps)
        l_pool = (((p_pool / v_pool - t_pool / v_pool) ** 2) * (v_pool > 0.1).float()).mean()

        total = (
            self.cfg.w_frontier_bce * l_frontier_bce
            + self.cfg.w_mse * l_mse
            + self.cfg.w_area * l_area
            + self.cfg.w_iou * l_iou
            + self.cfg.w_pool * l_pool
        )
        return {
            "loss_total": total,
            "l_frontier_bce": l_frontier_bce,
            "l_mse": l_mse,
            "l_area": l_area,
            "l_iou": l_iou,
            "l_pool": l_pool,
        }

    def forward(
        self,
        model_out: Dict[str, torch.Tensor],
        target_cum: torch.Tensor,
        target_new: torch.Tensor,
        valid_next: torch.Tensor,
        new_growth_loss_weight: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Menghitung gabungan loss pada peta kumulatif C_{t+24} dan peta pertumbuhan baru Y_{t+24}
        serta regularisasi Total Variation pada parameter Theta_t (PRD Bagian 2.12 poin 4).
        """
        cum_losses = self._single_target_loss(model_out["p_fire_cum"], target_cum, valid_next)
        new_losses = self._single_target_loss(model_out["delta_p_fire"], target_new, valid_next)
        l_tv = self.total_variation_loss(model_out["theta"])

        total = (
            cum_losses["loss_total"]
            + new_growth_loss_weight * new_losses["loss_total"]
            + self.cfg.w_tv_smooth * l_tv
        )
        return {
            "loss": total,
            "loss_cum": cum_losses["loss_total"],
            "loss_new": new_losses["loss_total"],
            "loss_tv": l_tv,
            "iou_cum_soft": 1.0 - cum_losses["l_iou"],
            "iou_new_soft": 1.0 - new_losses["l_iou"],
        }


def compute_ca_reachability_mask(
    initial_fire_mask: torch.Tensor,
    n_micro_steps: int = 15,
) -> torch.Tensor:
    """
    Membuat mask jangkauan geometri maksimum CA (jarak Chebyshev <= n_micro_steps sel dari C_t)
    sesuai PRD Bagian 2.10 untuk memisahkan evaluasi antara:
      1. Strata 'Dapat Dijangkau CA' (perambatan dari api yang sudah menyala).
      2. Strata 'Tak Terjangkau CA' (ignisi antropogenik independen baru / spotting jauh).
    """
    k = 2 * n_micro_steps + 1
    reachable = F.max_pool2d(
        initial_fire_mask.float(),
        kernel_size=k,
        stride=1,
        padding=n_micro_steps,
    )
    return (reachable > 0.5).float()


def compute_fractions_skill_score(
    pred_bin: torch.Tensor,
    target_bin: torch.Tensor,
    valid_mask: torch.Tensor,
    neighborhood_radius_px: int = 1,
) -> float:
    """
    Menghitung Fractions Skill Score (FSS) pada jendela (2*r + 1) x (2*r + 1)
    untuk menoleransi ketidakpastian geolokasi ±1-2 piksel VIIRS 375 m (PRD Bagian 3.5).
    """
    k = 2 * neighborhood_radius_px + 1
    v = valid_mask.float()
    p_frac = F.avg_pool2d(pred_bin.float() * v, kernel_size=k, stride=1, padding=neighborhood_radius_px)
    t_frac = F.avg_pool2d(target_bin.float() * v, kernel_size=k, stride=1, padding=neighborhood_radius_px)

    mse_frac = torch.mean((p_frac - t_frac) ** 2).item()
    ref_mse = torch.mean(p_frac ** 2 + t_frac ** 2).item()
    if ref_mse < 1e-12:
        return 1.0
    return float(1.0 - (mse_frac / ref_mse))


def evaluate_predictions_stratified(
    pred_prob: torch.Tensor,
    target_bin: torch.Tensor,
    valid_mask: torch.Tensor,
    initial_fire_mask: torch.Tensor,
    n_micro_steps: int = 15,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Menghitung paket metrik lengkap PRD Bagian 3.4, 3.5, & 3.9 hanya pada sel valid (M=1):
      - PR-AUC (Average Precision), Precision, Recall, F1/Dice, IoU (Jaccard), Brier Score, FSS (r=1, r=2)
      - Dipisahkan untuk keseluruhan patch, strata 'Dapat Dijangkau CA', dan 'Tak Terjangkau CA'.
    """
    reach_mask = compute_ca_reachability_mask(initial_fire_mask, n_micro_steps=n_micro_steps)
    unreach_mask = 1.0 - reach_mask

    def _calc_subset(sub_mask: torch.Tensor, prefix: str) -> Dict[str, float]:
        mask_bool = (sub_mask > 0.5).cpu().numpy().reshape(-1)
        p_np = pred_prob.detach().cpu().numpy().reshape(-1)[mask_bool]
        t_np = target_bin.detach().cpu().numpy().reshape(-1)[mask_bool].astype(np.int32)

        if len(t_np) == 0:
            return {f"{prefix}_ap": 0.0, f"{prefix}_f1": 0.0, f"{prefix}_iou": 0.0, f"{prefix}_pos_count": 0.0}

        pos_count = int(t_np.sum())
        if pos_count > 0 and pos_count < len(t_np):
            ap = float(average_precision_score(t_np, p_np))
        elif pos_count == len(t_np) and len(t_np) > 0:
            ap = 1.0
        else:
            ap = 0.0

        p_bin = (p_np >= threshold).astype(np.int32)
        tp = float(np.sum((p_bin == 1) & (t_np == 1)))
        fp = float(np.sum((p_bin == 1) & (t_np == 0)))
        fn = float(np.sum((p_bin == 0) & (t_np == 1)))

        precision = tp / max(tp + fp, 1e-8)
        recall = tp / max(tp + fn, 1e-8)
        f1 = (2.0 * tp) / max(2.0 * tp + fp + fn, 1e-8)
        iou = tp / max(tp + fp + fn, 1e-8)
        brier = float(np.mean((p_np - t_np) ** 2))

        return {
            f"{prefix}_ap": round(ap, 4),
            f"{prefix}_precision": round(precision, 4),
            f"{prefix}_recall": round(recall, 4),
            f"{prefix}_f1": round(f1, 4),
            f"{prefix}_iou": round(iou, 4),
            f"{prefix}_brier": round(brier, 5),
            f"{prefix}_pos_pixels": float(pos_count),
        }

    v = valid_mask.float()
    metrics = {}
    metrics.update(_calc_subset(v, "all"))
    metrics.update(_calc_subset(v * reach_mask, "reachable_ca"))
    metrics.update(_calc_subset(v * unreach_mask, "unreachable_ignition"))

    pred_b_tensor = (pred_prob >= threshold).float()
    metrics["fss_r1_3x3"] = round(compute_fractions_skill_score(pred_b_tensor, target_bin, v, neighborhood_radius_px=1), 4)
    metrics["fss_r2_5x5"] = round(compute_fractions_skill_score(pred_b_tensor, target_bin, v, neighborhood_radius_px=2), 4)
    return metrics

