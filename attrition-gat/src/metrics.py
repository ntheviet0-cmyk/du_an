"""Metrics: PR-AUC chinh; phu ROC-AUC, MCC, Brier, F1/precision/recall @ nguong val, P@top-k%."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)


def precision_at_top_k(y_true: np.ndarray, proba: np.ndarray, k_frac: float) -> float:
    n = len(y_true)
    k = max(1, int(n * k_frac))
    top = np.argsort(proba)[::-1][:k]
    return float(y_true[top].mean())


def pick_threshold_recall_ge(val_proba: np.ndarray, y_val: np.ndarray, target_recall=0.70) -> float:
    """Chon nguong tren VAL: recall>=70% voi precision cao nhat; neu khong dat, lay recall max."""
    best_t, best_p, best_r = 0.5, -1.0, -1.0
    for t in np.arange(0.05, 0.95, 0.05):
        pred = (val_proba >= t).astype(int)
        r = recall_score(y_val, pred, zero_division=0)
        p = precision_score(y_val, pred, zero_division=0)
        if r >= target_recall and p > best_p:
            best_t, best_p, best_r = float(t), float(p), float(r)
    if best_p < 0:  # khong nguong nao dat recall
        for t in np.arange(0.05, 0.95, 0.05):
            pred = (val_proba >= t).astype(int)
            r = recall_score(y_val, pred, zero_division=0)
            if r > best_r:
                best_r, best_t = r, float(t)
    return best_t


def evaluate_test(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "brier": float(brier_score_loss(y_true, proba)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "threshold": float(threshold),
        "p_at_10": float(precision_at_top_k(y_true, proba, 0.10)),
        "p_at_20": float(precision_at_top_k(y_true, proba, 0.20)),
        "n_test": int(len(y_true)),
        "n_pos_test": int(y_true.sum()),
        "n_pred_pos": int(pred.sum()),
    }
