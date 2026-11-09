"""
Modul PyTorch Dataset, Leave-One-Year-Out (LOYO) & Spatial Block Splitter,
Normalisasi Anti-Leakage (PRD Bagian 1.13), dan Augmentasi Sadar-Vektor (PRD Bagian 1.12).
"""

import glob
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import LITE_17_CHANNELS


def create_loyo_and_spatial_splits(
    npz_files: List[str],
    test_year: int = 2023,
    val_year: int = 2019,
    spatial_block_deg: Optional[float] = None,
) -> Dict[str, List[str]]:
    """
    Pembagian dataset berdasarkan Tahun (Leave-One-Year-Out / LOYO) dan opsional Blok Spasial,
    dengan jaminan ketat: tidak ada tile dari event_id yang sama pada split berbeda (PRD 3.1 & 3.2).
    """
    train_files, val_files, test_files = [], [], []
    event_to_split: Dict[str, str] = {}

    for fpath in sorted(npz_files):
        with np.load(fpath, allow_pickle=True) as data:
            eid = str(data["event_id"])
            date_str = str(data["date"])
            year = int(date_str[:4])

        if eid in event_to_split:
            assigned = event_to_split[eid]
        else:
            if year == test_year:
                assigned = "test"
            elif year == val_year:
                assigned = "val"
            else:
                assigned = "train"
            event_to_split[eid] = assigned

        if assigned == "train":
            train_files.append(fpath)
        elif assigned == "val":
            val_files.append(fpath)
        else:
            test_files.append(fpath)

    return {"train": train_files, "val": val_files, "test": test_files}


def compute_training_normalization_stats(
    train_files: List[str],
    channel_names: List[str] = LITE_17_CHANNELS,
) -> Dict[str, np.ndarray]:
    """
    Menghitung statistik normalisasi (persentil 0.1, persentil 99.9, mean, std)
    HANYA dari split training sesuai PRD Bagian 1.13:
    - Hujan (precip_*): transformasi log(1 + x) sebelum pembakuan.
    - Indeks kategorik (landcover_idx, oil_palm_class) & mask biner (fire_*, valid_t, peat):
      tidak dibakukan (mean=0, std=1).
    """
    C = len(channel_names)
    non_standardized = {
        "fire_t", "fire_t_minus_1", "fire_t_minus_2", "valid_t", "peat", "landcover_idx", "oil_palm_class"
    }
    log_transform_channels = {"precip_1d", "precip_7d", "precip_30d", "precip_90d", "gfs_precip", "count_14d"}

    sampled_vals: List[np.ndarray] = [[] for _ in range(C)]
    for fpath in train_files:
        with np.load(fpath, allow_pickle=True) as data:
            x = data["x"].astype(np.float32)  # [C, H, W]
            # Subsample setiap 8 piksel untuk efisiensi memori
            x_sub = x[:, ::4, ::4].reshape(C, -1)
            for c_idx, ch_name in enumerate(channel_names):
                vals = x_sub[c_idx]
                vals = vals[np.isfinite(vals)]
                if ch_name in log_transform_channels:
                    vals = np.log1p(np.maximum(vals, 0.0))
                sampled_vals[c_idx].append(vals)

    p001 = np.zeros(C, dtype=np.float32)
    p999 = np.ones(C, dtype=np.float32) * 1e6
    mean = np.zeros(C, dtype=np.float32)
    std = np.ones(C, dtype=np.float32)

    for c_idx, ch_name in enumerate(channel_names):
        if ch_name in non_standardized or len(sampled_vals[c_idx]) == 0:
            p001[c_idx] = -1e9
            p999[c_idx] = 1e9
            mean[c_idx] = 0.0
            std[c_idx] = 1.0
            continue

        all_v = np.concatenate(sampled_vals[c_idx], axis=0)
        lo = float(np.percentile(all_v, 0.1))
        hi = float(np.percentile(all_v, 99.9))
        if hi <= lo:
            hi = lo + 1.0
        clipped = np.clip(all_v, lo, hi)
        m = float(np.mean(clipped))
        s = float(np.std(clipped))
        if s < 1e-6:
            s = 1.0

        p001[c_idx] = lo
        p999[c_idx] = hi
        mean[c_idx] = m
        std[c_idx] = s

    return {"p001": p001, "p999": p999, "mean": mean, "std": std}


def apply_vector_aware_augmentation(
    x: np.ndarray,
    state_t: np.ndarray,
    target_cum: np.ndarray,
    target_new: np.ndarray,
    valid_next: np.ndarray,
    ca_phys: np.ndarray,
    channel_names: List[str] = LITE_17_CHANNELS,
    flip_h: bool = False,
    flip_v: bool = False,
    rot_k: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Augmentasi spasial (horizontal flip, vertical flip, rotasi 90*k derajat Counter-Clockwise)
    yang ikut mentransformasi kanal vektor (wind_u, wind_v, gfs_wind_u, gfs_wind_v)
    serta sudut fisik CA (theta_wind, theta_upslope) sesuai PRD Bagian 1.12!
    """
    ch_to_idx = {name: i for i, name in enumerate(channel_names)}
    u_indices = [ch_to_idx[n] for n in ("wind_u", "gfs_wind_u") if n in ch_to_idx]
    v_indices = [ch_to_idx[n] for n in ("wind_v", "gfs_wind_v") if n in ch_to_idx]

    arrays = [x.copy(), state_t.copy(), target_cum.copy(), target_new.copy(), valid_next.copy(), ca_phys.copy()]

    # 1. Horizontal Flip (sumbu X dibalik: u -> -u, v -> v; sudut theta -> pi - theta)
    if flip_h:
        arrays = [np.flip(arr, axis=-1).copy() for arr in arrays]
        for ui in u_indices:
            arrays[0][ui] = -arrays[0][ui]
        # ca_phys[1] = wind_dir_rad, ca_phys[3] = upslope_dir_rad
        for ang_idx in (1, 3):
            arrays[5][ang_idx] = np.arctan2(np.sin(np.pi - arrays[5][ang_idx]), np.cos(np.pi - arrays[5][ang_idx]))

    # 2. Vertical Flip (sumbu Y utara-selatan dibalik: u -> u, v -> -v; sudut theta -> -theta)
    if flip_v:
        arrays = [np.flip(arr, axis=-2).copy() for arr in arrays]
        for vi in v_indices:
            arrays[0][vi] = -arrays[0][vi]
        for ang_idx in (1, 3):
            arrays[5][ang_idx] = -arrays[5][ang_idx]

    # 3. Rotasi k * 90 derajat Counter-Clockwise pada grid (axes=(-2, -1))
    rot_k = rot_k % 4
    if rot_k > 0:
        arrays = [np.rot90(arr, k=rot_k, axes=(-2, -1)).copy() for arr in arrays]
        for _ in range(rot_k):
            # Rotasi 90 deg CCW: (u', v') = (-v, u) dan theta' = theta + pi/2
            for ui, vi in zip(u_indices, v_indices):
                old_u = arrays[0][ui].copy()
                old_v = arrays[0][vi].copy()
                arrays[0][ui] = -old_v
                arrays[0][vi] = old_u
            for ang_idx in (1, 3):
                new_ang = arrays[5][ang_idx] + (np.pi / 2.0)
                arrays[5][ang_idx] = np.arctan2(np.sin(new_ang), np.cos(new_ang))

    return arrays[0], arrays[1], arrays[2], arrays[3], arrays[4], arrays[5]


class KalimantanFireSpreadDataset(Dataset):
    """
    Dataset PyTorch untuk pelatihan & evaluasi DeepLabV3+ + Probabilistic CA.
    Mendukung normalisasi anti-leakage dan augmentasi sadar-vektor.
    """

    def __init__(
        self,
        npz_files: List[str],
        norm_stats: Optional[Dict[str, np.ndarray]] = None,
        channel_names: List[str] = LITE_17_CHANNELS,
        augment: bool = False,
    ):
        self.npz_files = list(npz_files)
        self.norm_stats = norm_stats
        self.channel_names = list(channel_names)
        self.augment = augment
        self.log_channels = {"precip_1d", "precip_7d", "precip_30d", "precip_90d", "gfs_precip", "count_14d"}

    def __len__(self) -> int:
        return len(self.npz_files)

    def _standardize_x(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Menstandarisasi prediktor kontinu, mengisi NaN dengan 0 setelah standarisasi (PRD 1.5.2),
        dan mengekstrak `landcover_idx` sebagai tensor integer terpisah untuk layer Embedding.
        """
        x_proc = x.copy()
        lc_idx_pos = self.channel_names.index("landcover_idx") if "landcover_idx" in self.channel_names else -1
        if lc_idx_pos >= 0:
            lc_map = np.clip(np.nan_to_num(x_proc[lc_idx_pos], nan=0.0).astype(np.int64), 0, 17)
        else:
            lc_map = np.zeros(x_proc.shape[1:], dtype=np.int64)

        for c_i, ch_name in enumerate(self.channel_names):
            band = x_proc[c_i]
            nan_mask = ~np.isfinite(band)
            band = np.nan_to_num(band, nan=0.0)
            if ch_name in self.log_channels:
                band = np.log1p(np.maximum(band, 0.0))

            if self.norm_stats is not None:
                lo = self.norm_stats["p001"][c_i]
                hi = self.norm_stats["p999"][c_i]
                m = self.norm_stats["mean"][c_i]
                s = self.norm_stats["std"][c_i]
                band = (np.clip(band, lo, hi) - m) / s

            # Isi piksel hilang dengan 0 setelah standarisasi (~ rerata latih, PRD 1.5.2)
            band[nan_mask] = 0.0
            x_proc[c_i] = band

        return x_proc, lc_map

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fpath = self.npz_files[idx]
        with np.load(fpath, allow_pickle=True) as data:
            x = data["x"].astype(np.float32)
            state_t = data["state_t"].astype(np.float32)
            target_cum = data["target_cum"].astype(np.float32)
            target_new = data["target_new"].astype(np.float32)
            valid_next = data["valid_next"].astype(np.float32)
            ca_phys = data["ca_phys"].astype(np.float32)
            scalars = data["scalars"].astype(np.float32) if "scalars" in data else np.zeros(3, dtype=np.float32)

        # Ekstrak variabel fisik Mentah untuk p_continue adaptif gambut (M1): [peat, DC_norm, rain7d_norm]
        peat_idx = self.channel_names.index("peat") if "peat" in self.channel_names else 6
        dc_idx = self.channel_names.index("DC") if "DC" in self.channel_names else 12
        rain_idx = self.channel_names.index("precip_7d") if "precip_7d" in self.channel_names else 10

        if self.augment:
            fh = bool(np.random.rand() < 0.5)
            fv = bool(np.random.rand() < 0.5)
            rk = int(np.random.randint(0, 4))
            x, state_t, target_cum, target_new, valid_next, ca_phys = apply_vector_aware_augmentation(
                x, state_t, target_cum, target_new, valid_next, ca_phys,
                channel_names=self.channel_names, flip_h=fh, flip_v=fv, rot_k=rk,
            )

        peat_raw = np.clip(np.nan_to_num(x[peat_idx], nan=0.0), 0.0, 1.0)
        dc_raw_norm = np.clip(np.nan_to_num(x[dc_idx], nan=0.0) / 600.0, 0.0, 2.0)
        rain_raw_norm = np.clip(np.log1p(np.maximum(np.nan_to_num(x[rain_idx], nan=0.0), 0.0)) / 5.0, 0.0, 2.0)
        peat_hydro = np.stack([peat_raw, dc_raw_norm, rain_raw_norm], axis=0).astype(np.float32)

        x_norm, lc_map = self._standardize_x(x)

        return {
            "x": torch.from_numpy(x_norm),
            "landcover_idx": torch.from_numpy(lc_map),
            "state_t": torch.from_numpy(state_t),
            "target_cum": torch.from_numpy(target_cum),
            "target_new": torch.from_numpy(target_new),
            "valid_next": torch.from_numpy(valid_next),
            "ca_phys": torch.from_numpy(ca_phys),
            "peat_hydro": torch.from_numpy(peat_hydro),
            "scalars": torch.from_numpy(scalars),
        }

