"""CV protocol: Repeated Stratified 5-fold x 5 repeats = 25 outer.
Trong train moi fold, tach 20% validation (stratified) cho tuning/early-stop/nguong.
Luu fold index de tai lap. Smoke-test: repeat 0, fold 0.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit


def outer_splits(y: np.ndarray, n_splits=5, n_repeats=5, seed=42):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    idx = np.arange(len(y))
    for rep_fold, (tr, te) in enumerate(rskf.split(idx, y)):
        rep, fold = divmod(rep_fold, n_splits)
        yield {"repeat": int(rep), "fold": int(fold), "train_idx": tr, "test_idx": te}


def inner_val_split(y_train: np.ndarray, val_size=0.2, seed=42):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tr, va = next(sss.split(np.zeros(len(y_train)), y_train))
    return tr, va


def save_folds(y: np.ndarray, out_path: str | Path, n_splits=5, n_repeats=5, seed=42):
    out = []
    for s in outer_splits(y, n_splits, n_repeats, seed):
        out.append({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"seed": seed, "n_splits": n_splits, "n_repeats": n_repeats, "folds": out}, f)
    return out_path
