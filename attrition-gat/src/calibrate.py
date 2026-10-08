"""Calibration tren OOF (khong dung lai tap tuning):
- Platt (chinh): LogisticRegression tren oof_proba 1 chieu.
- Isotonic (phan tich do nhay): IsotonicRegression.
Platt don dieu -> khong doi PR/ROC, chi doi Brier va nguong. Fit tren OOF (out-of-sample).
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def fit_platt(oof: np.ndarray, y: np.ndarray) -> LogisticRegression:
    return LogisticRegression().fit(np.asarray(oof).reshape(-1, 1), np.asarray(y))


def fit_isotonic(oof: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    return IsotonicRegression(out_of_bounds="clip").fit(np.asarray(oof), np.asarray(y))


def apply_platt(cal: LogisticRegression, p: np.ndarray) -> np.ndarray:
    return cal.predict_proba(np.asarray(p).reshape(-1, 1))[:, 1]


def apply_isotonic(cal: IsotonicRegression, p: np.ndarray) -> np.ndarray:
    return np.clip(cal.predict(np.asarray(p)), 0.0, 1.0)
