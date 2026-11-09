"""
Skrip Pelatihan & Evaluasi Leave-One-Year-Out (LOYO) untuk:
  - Model Usulan (P / P+): DeepLabV3+ + Probabilistic CA (Varian M0 & M1 Gambut Adaptif)
  - Baseline Paper Rujukan (B7): MS-CNN + Probabilistic CA [Zhenirovskyy2026]
  - Baseline Persistence (B0): C_{t+24}^{pred} = C_t
Sesuai PRD v2 Bagian 3.1, 3.3, 3.4, 3.5, dan 3.6.
"""

import argparse
import glob
import os
import sys
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, r"c:\Neural-CA-Kalimantan")

from src.config import ModelCAConfig, LITE_17_CHANNELS
from src.data.dataset import (
    create_loyo_and_spatial_splits,
    compute_training_normalization_stats,
    KalimantanFireSpreadDataset,
)
from src.models.neural_ca import NeuralCAWildfireModel
from src.models.losses_and_metrics import MaskedHybridCALoss, evaluate_predictions_stratified


def evaluate_loader(
    model: NeuralCAWildfireModel,
    loader: DataLoader,
    criterion: MaskedHybridCALoss,
    device: torch.device,
    n_micro_steps: int = 15,
) -> Dict[str, float]:
    model.eval()
    agg_metrics: Dict[str, List[float]] = {}
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            lc = batch["landcover_idx"].to(device)
            st = batch["state_t"].to(device)
            tc = batch["target_cum"].to(device)
            tn = batch["target_new"].to(device)
            vn = batch["valid_next"].to(device)
            phys = batch["ca_phys"].to(device)
            ph = batch["peat_hydro"].to(device)

            out = model(x, lc, st, phys, peat_hydro=ph, n_micro_steps=n_micro_steps)
            loss_dict = criterion(out, tc, tn, vn)

            init_fire = (st[:, 1:2] + st[:, 2:3]).clamp(0.0, 1.0)
            m_cum = evaluate_predictions_stratified(
                out["p_fire_cum"], tc, vn, init_fire, n_micro_steps=n_micro_steps, threshold=0.35
            )
            m_new = evaluate_predictions_stratified(
                out["delta_p_fire"], tn, vn, init_fire, n_micro_steps=n_micro_steps, threshold=0.25
            )

            agg_metrics.setdefault("loss", []).append(float(loss_dict["loss"].item()))
            for k, v in m_cum.items():
                agg_metrics.setdefault(f"cum_{k}", []).append(float(v))
            for k, v in m_new.items():
                agg_metrics.setdefault(f"new_{k}", []).append(float(v))

    return {k: round(float(np.mean(vals)), 4) for k, vals in agg_metrics.items()}


def main():
    parser = argparse.ArgumentParser(description="Train & Evaluate Neural-CA Wildfire Spread (LOYO)")
    parser.add_argument("--data-dir", type=str, default=r"c:\Neural-CA-Kalimantan\data\processed")
    parser.add_argument("--generator", type=str, default="deeplabv3plus", choices=["deeplabv3plus", "mscnn"])
    parser.add_argument("--test-year", type=int, default=2023)
    parser.add_argument("--val-year", type=int, default=2019)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--micro-steps", type=int, default=15)
    parser.add_argument("--adaptive-peat", action="store_true", default=True)
    args = parser.parse_args()

    npz_files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if len(npz_files) == 0:
        raise FileNotFoundError(f"Tidak ada berkas .npz di {args.data_dir}")

    splits = create_loyo_and_spatial_splits(npz_files, test_year=args.test_year, val_year=args.val_year)
    train_files = splits["train"] if len(splits["train"]) > 0 else npz_files
    val_files = splits["val"] if len(splits["val"]) > 0 else npz_files

    print(f"[Split LOYO] Train={len(train_files)} sampel | Val={len(val_files)} sampel | Test={len(splits['test'])} sampel")
    norm_stats = compute_training_normalization_stats(train_files, channel_names=LITE_17_CHANNELS)

    train_ds = KalimantanFireSpreadDataset(train_files, norm_stats=norm_stats, augment=True)
    val_ds = KalimantanFireSpreadDataset(val_files, norm_stats=norm_stats, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ModelCAConfig(n_micro_steps=args.micro_steps, use_adaptive_p_continue=args.adaptive_peat)
    model = NeuralCAWildfireModel(cfg=cfg, generator_type=args.generator, in_channels=len(LITE_17_CHANNELS)).to(device)
    criterion = MaskedHybridCALoss(cfg=cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(
                batch["x"].to(device),
                batch["landcover_idx"].to(device),
                batch["state_t"].to(device),
                batch["ca_phys"].to(device),
                peat_hydro=batch["peat_hydro"].to(device),
            )
            loss_dict = criterion(
                out,
                batch["target_cum"].to(device),
                batch["target_new"].to(device),
                batch["valid_next"].to(device),
            )
            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            val_metrics = evaluate_loader(model, val_loader, criterion, device, n_micro_steps=args.micro_steps)
            print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss={np.mean(epoch_losses):.4f} | "
                  f"Val Loss={val_metrics['loss']:.4f} | Cum AP={val_metrics['cum_all_ap']:.4f} | "
                  f"New Reachable AP={val_metrics['new_reachable_ca_ap']:.4f} | FSS(5x5)={val_metrics['cum_fss_r2_5x5']:.4f}")


if __name__ == "__main__":
    main()

