"""MOSAIC training / evaluation entry point.

Self-contained: perturbation operator, loss aggregation, and ACC/SEN/SPE
metrics are inlined in this file so the project only depends on
`configs/`, `datasets/`, and `models/`.

Usage
-----
Train:
    python main.py --config configs/abide1.yaml --mode train

Evaluate:
    python main.py --config configs/abide1.yaml --mode eval \
        --ckpt runs/abide1/best.pt

Ablation example:
    python main.py --config configs/abide1.yaml --mode train --ablation no_pert
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from datasets.connectome_dataset import ConnectomeDataset, group_stratified_kfold
from models.mosaic import MOSAIC


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MOSAIC training / evaluation")
    p.add_argument("--config", type=str, required=True,
                   help="Path to YAML configuration file.")
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Checkpoint path required when --mode eval.")
    p.add_argument("--ablation", type=str, default=None,
                   choices=[None, "no_pert", "G1", "stgumbel",
                            "topk", "softgate", "l1"],
                   help="Mask-discovery ablation switch (Sec. 3.5 / Tab. 4).")
    p.add_argument("--fusion", type=str, default=None,
                   choices=[None, "concat", "gated", "bidir",
                            "reverse", "sc_anchor",
                            "sc2fc_only", "sc2ec_only", "shared"],
                   help="Fusion-design switch (Sec. 3.6).")
    p.add_argument("--modalities", type=str, default=None,
                   help="Comma-joined modalities, e.g. FC+SC+EC, FC+EC, SC.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default=None,
                   help="Override config.out_dir.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def merge_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.ablation is not None:
        cfg.setdefault("ablation", {})["mask_variant"] = args.ablation
    if args.fusion is not None:
        cfg.setdefault("fusion", {})["variant"] = args.fusion
    if args.modalities is not None:
        cfg["modalities"] = args.modalities.split("+")
    if args.out_dir is not None:
        cfg["out_dir"] = args.out_dir
    return cfg


# ---------------------------------------------------------------------------
# Modality-aware perturbation operator T(.) (Sec. 2.4 of the paper)
# ---------------------------------------------------------------------------
class ModalityPerturbation:
    """Symmetric edge-dropout + weight-jitter, modality-aware.

    For FC and SC the perturbation preserves symmetry; for EC, it preserves
    directionality (no symmetrization). Sign structure of EC is preserved
    when signed interactions are present.
    """

    def __init__(self, edge_dropout: float = 0.10,
                 weight_jitter: float = 0.05) -> None:
        self.p_drop = edge_dropout
        self.sigma = weight_jitter

    def _perturb_undirected(self, A: torch.Tensor) -> torch.Tensor:
        # A: (B, R, R) symmetric
        B, R, _ = A.shape
        triu_mask = torch.triu(torch.ones(R, R, device=A.device), diagonal=1)
        keep = (torch.rand(B, R, R, device=A.device) > self.p_drop).float()
        keep = keep * triu_mask + keep.transpose(-1, -2) * triu_mask.t()
        keep = keep + torch.eye(R, device=A.device).unsqueeze(0)
        jitter = 1.0 + self.sigma * torch.randn_like(A)
        jitter = 0.5 * (jitter + jitter.transpose(-1, -2))
        return A * keep * jitter

    def _perturb_directed(self, A: torch.Tensor) -> torch.Tensor:
        keep = (torch.rand_like(A) > self.p_drop).float()
        jitter = 1.0 + self.sigma * torch.randn_like(A)
        return A * keep * jitter

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for m, A in x.items():
            if m in ("FC", "SC"):
                out[m] = self._perturb_undirected(A)
            elif m == "EC":
                out[m] = self._perturb_directed(A)
            else:
                out[m] = A
        return out


# ---------------------------------------------------------------------------
# Total objective L_total (Sec. 2.6 of the paper)
# ---------------------------------------------------------------------------
def total_loss(outputs: dict, y: torch.Tensor, cfg: dict) -> Dict[str, torch.Tensor]:
    """Combine cls + RL + compactness + consistency - entropy.

    `outputs` is expected to contain at least:
        logits         (B, C)
        rl_loss        scalar  (group-relative REINFORCE term)
        compactness    scalar  (mean retention rate)
        consistency    scalar  (paired logit + mask agreement penalty)
        entropy        scalar  (mean Bernoulli entropy of the mask policy)
    """
    cls = F.cross_entropy(outputs["logits"], y)
    rl = outputs.get("rl_loss", cls.new_zeros(()))
    cmp_ = outputs.get("compactness", cls.new_zeros(()))
    cons = outputs.get("consistency", cls.new_zeros(()))
    ent = outputs.get("entropy", cls.new_zeros(()))
    w = cfg.get("loss_weights", {})
    total = (cls
             + w.get("rl", 1.0) * rl
             + w.get("cmp", 1.0) * cmp_
             + cons
             - w.get("ent", 0.01) * ent)
    return {"total": total, "cls": cls.detach(), "rl": rl.detach(),
            "cmp": cmp_.detach(), "cons": cons.detach(), "ent": ent.detach()}


# ---------------------------------------------------------------------------
# Metrics: ACC / SEN / SPE
# ---------------------------------------------------------------------------
def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    pred = logits.argmax(dim=-1)
    tp = ((pred == 1) & (labels == 1)).sum().item()
    tn = ((pred == 0) & (labels == 0)).sum().item()
    fp = ((pred == 1) & (labels == 0)).sum().item()
    fn = ((pred == 0) & (labels == 1)).sum().item()
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    sen = tp / max(tp + fn, 1)   # sensitivity / recall on positive class
    spe = tn / max(tn + fp, 1)   # specificity
    return {"acc": acc, "sen": sen, "spe": spe}


# ---------------------------------------------------------------------------
# Train / Eval loops
# ---------------------------------------------------------------------------
def run_one_fold(cfg: dict, fold_idx: int, train_idx, val_idx,
                 dataset: ConnectomeDataset, device: torch.device) -> dict:
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=cfg.get("num_workers", 2),
                              drop_last=True)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"],
                            shuffle=False, num_workers=cfg.get("num_workers", 2))

    model = MOSAIC(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(),
                              lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    perturb = ModalityPerturbation(
        edge_dropout=cfg["perturbation"]["edge_dropout"],
        weight_jitter=cfg["perturbation"]["weight_jitter"],
    )

    best_acc, best_state = 0.0, None
    for epoch in range(cfg["epochs"]):
        model.train()
        for batch in train_loader:
            x, y = batch["x"], batch["y"].to(device)
            x = {m: v.to(device) for m, v in x.items()}
            x_perturbed = perturb(x) if cfg["perturbation"]["enabled"] else None

            outputs = model(x, x_perturbed=x_perturbed,
                            group_size=cfg["rl"]["group_size"])
            losses = total_loss(outputs, y, cfg)
            optim.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

        # ---- Validation -----------------------------------------------------
        metrics = evaluate(model, val_loader, device)
        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

        print(f"[fold {fold_idx}] epoch {epoch:3d} | "
              f"val ACC={metrics['acc']:.4f} SEN={metrics['sen']:.4f} "
              f"SPE={metrics['spe']:.4f}")

    out_dir = Path(cfg["out_dir"]) / f"fold_{fold_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        torch.save(best_state, out_dir / "best.pt")
    return {"fold": fold_idx, "best_acc": best_acc}


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        x, y = batch["x"], batch["y"].to(device)
        x = {m: v.to(device) for m, v in x.items()}
        outputs = model(x, x_perturbed=None, group_size=1, deterministic=True)
        all_logits.append(outputs["logits"].cpu())
        all_labels.append(y.cpu())
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return classification_metrics(logits, labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    cfg = merge_cli_overrides(load_config(args.config), args)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ConnectomeDataset(
        data_root=cfg["data_root"],
        atlas=cfg["atlas"],
        modalities=cfg.get("modalities", ["FC", "SC", "EC"]),
    )

    if args.mode == "train":
        results = []
        for fold_idx, (train_idx, val_idx) in enumerate(
            group_stratified_kfold(dataset, n_splits=cfg["n_folds"], seed=args.seed)
        ):
            results.append(run_one_fold(cfg, fold_idx, train_idx, val_idx,
                                        dataset, device))
        accs = [r["best_acc"] for r in results]
        print(f"\nMean ACC over {len(accs)} folds: "
              f"{np.mean(accs):.4f} (+/- {np.std(accs):.4f})")

    else:  # eval
        assert args.ckpt is not None, "--ckpt is required when --mode eval"
        model = MOSAIC(cfg).to(device)
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        loader = DataLoader(dataset, batch_size=cfg["batch_size"],
                            shuffle=False, num_workers=cfg.get("num_workers", 2))
        metrics = evaluate(model, loader, device)
        print(f"ACC={metrics['acc']:.4f} | SEN={metrics['sen']:.4f} | "
              f"SPE={metrics['spe']:.4f}")


if __name__ == "__main__":
    main()
