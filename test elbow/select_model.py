"""Fold-level summaries, one-standard-error selection and knee candidates."""
import numpy as np
import pandas as pd

PARAMS = ["k", "h1", "h2", "dropout", "lr", "weight_decay"]


def summarize(runs):
    if runs.empty:
        return pd.DataFrame()
    # Seeds are repeated measurements within a fold, not independent folds.
    fold_means = runs.groupby(["stage", "candidate", "fold"], as_index=False).agg(
        ap=("val_ap", "mean"), auc=("val_auc", "mean"))
    stats = fold_means.groupby(["stage", "candidate"], as_index=False).agg(
        mean_ap=("ap", "mean"), std_ap=("ap", "std"), n_folds=("fold", "nunique"),
        mean_auc=("auc", "mean"))
    stats["se_ap"] = stats.std_ap / np.sqrt(stats.n_folds)
    info = runs.groupby(["stage", "candidate"], as_index=False).agg(
        **{key: (key, "first") for key in PARAMS},
        parameters=("parameters", "first"), mean_edges=("edges", "mean"),
        mean_seconds=("seconds", "mean"), n_runs=("run_id", "size"),
        mean_best_epoch=("best_epoch", "mean"))
    return stats.merge(info, on=["stage", "candidate"])


def select_one_se(summary):
    if summary.empty or len(summary.n_folds.unique()) != 1 or summary.n_folds.min() < 2:
        raise ValueError("Selection requires complete candidates with at least two matching folds")
    best = summary.sort_values(["mean_ap", "candidate"], ascending=[False, True]).iloc[0]
    cutoff = best.mean_ap - best.se_ap
    eligible = summary[summary.mean_ap >= cutoff]
    # Explicit complexity order: graph size, parameter count, inference/training cost.
    chosen = eligible.sort_values(["mean_edges", "parameters", "mean_seconds", "candidate"]).iloc[0]
    return chosen, best, float(cutoff)


def knee_candidate(x, y):
    """Heuristic maximum distance above endpoint chord, not a forced optimum.

    Returns None for non-monotone, flat, linear, or boundary-dominated curves.
    One-SE validation is still required independently of this heuristic.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) < 1e-8:
        return None
    order = np.argsort(x)
    x, y = x[order], y[order]
    if np.any(np.diff(x) <= 0) or np.any(np.diff(y) < -1e-8) or y[-1] <= y[0]:
        return None
    xn = (x - x[0]) / np.ptp(x)
    yn = (y - y[0]) / np.ptp(y)
    distance = yn - xn
    pos = int(np.argmax(distance))
    if pos in (0, len(x) - 1) or distance[pos] < 0.05:
        return None
    return {"x": float(x[pos]), "y": float(y[pos]), "normalized_distance": float(distance[pos])}
