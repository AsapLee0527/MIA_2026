"""Unified connectome dataset adapter.

The dataset is expected to have already been preprocessed into per-subject
``.npz`` files inside ``data_root``. Each archive must contain the keys
``FC``, ``SC``, ``EC`` (each an R x R numpy array on the cohort's atlas)
and ``label`` (binary). Optionally, a ``site`` key is used by the
group-stratified K-fold splitter to avoid acquisition-site leakage.

The same adapter handles ABIDE-I, ABIDE-II, ADHD-200, OASIS, and HCP; it
only differs in the per-cohort atlas (encoded in the YAML config).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

from .base import normalize_connectome, symmetrize


class ConnectomeDataset(Dataset):
    """Per-subject multimodal connectome dataset."""

    def __init__(
        self,
        data_root: str | os.PathLike,
        atlas: str,
        modalities: Sequence[str] = ("FC", "SC", "EC"),
        normalize: str = "zscore",
    ) -> None:
        self.root = Path(data_root)
        if not self.root.exists():
            raise FileNotFoundError(f"data_root not found: {self.root}")

        self.atlas = atlas
        self.modalities = list(modalities)
        self.normalize = normalize

        self.files = sorted(self.root.glob("*.npz"))
        if len(self.files) == 0:
            raise RuntimeError(
                f"No .npz files found under {self.root}. "
                "See README for the expected data layout."
            )

        # Cache labels and (optional) site ids for stratified splitting.
        self.labels: List[int] = []
        self.sites: List[str] = []
        for fp in self.files:
            with np.load(fp, allow_pickle=True) as f:
                self.labels.append(int(f["label"]))
                self.sites.append(str(f["site"]) if "site" in f.files else "0")

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        fp = self.files[idx]
        with np.load(fp, allow_pickle=True) as f:
            mats = {m: np.asarray(f[m], dtype=np.float32)
                    for m in self.modalities}
            label = int(f["label"])

        # Hygiene: enforce symmetry for FC/SC, leave EC asymmetric.
        if "FC" in mats:
            mats["FC"] = symmetrize(mats["FC"])
        if "SC" in mats:
            mats["SC"] = symmetrize(mats["SC"])

        if self.normalize != "none":
            for m in mats:
                mats[m] = normalize_connectome(mats[m], kind=self.normalize)

        x = {m: torch.from_numpy(v) for m, v in mats.items()}
        return {"x": x, "y": torch.tensor(label, dtype=torch.long),
                "site": self.sites[idx]}


# ----------------------------------------------------------------------
def group_stratified_kfold(
    dataset: ConnectomeDataset,
    n_splits: int = 5,
    seed: int = 42,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Stratified K-fold that also groups subjects by acquisition site.

    Note
    ----
    Sites are encoded by string id; folds are stratified jointly on
    (label, site) so that each fold contains the same label distribution
    while reducing site-specific leakage.
    """
    labels = np.asarray(dataset.labels)
    sites = np.asarray(dataset.sites)
    # Compose strata as "label-site" so StratifiedKFold respects both.
    strata = np.array([f"{l}-{s}" for l, s in zip(labels, sites)])

    # Some strata may be too small for K folds; merge them onto label only.
    unique, counts = np.unique(strata, return_counts=True)
    rare = set(unique[counts < n_splits])
    safe_strata = np.array(
        [str(l) if s_lbl in rare else s_lbl
         for s_lbl, l in zip(strata, labels)]
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indices = np.arange(len(dataset))
    for train_idx, val_idx in skf.split(indices, safe_strata):
        yield train_idx, val_idx
