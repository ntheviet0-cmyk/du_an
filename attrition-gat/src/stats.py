"""Kiem dinh gep cap hieu chinh (Nadeau-Bengio) cho repeated CV.
d_j = metric(model A) - metric(model B) theo fold j (J folds).
Phuong sai hieu chinh: var_nb = (1/J + n_test/n_train) * var(d)  (ddof=1).
t = mean(d)/sqrt(var_nb), df=J-1; CI95% = mean +/- t(.975,df)*sqrt(var_nb).
Ghi chu: cac fold trung lap (repeated CV) — day la hieu chinh tieu chuan, khong phai
suy luan doc lap day du; bao cao kem luu y nay.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats


def nb_paired_test(d: np.ndarray, n_test: int, n_train: int) -> dict:
    d = np.asarray(d, dtype=float)
    J = len(d)
    mean = float(d.mean())
    var = float(d.var(ddof=1)) if J > 1 else 0.0
    var_nb = (1.0 / J + n_test / n_train) * var
    se_nb = float(np.sqrt(var_nb))
    tcrit = float(sp_stats.t.ppf(0.975, J - 1)) if J > 1 else float("nan")
    if se_nb > 0 and J > 1:
        t = mean / se_nb
        p = float(2 * sp_stats.t.sf(abs(t), J - 1))
        lo, hi = mean - tcrit * se_nb, mean + tcrit * se_nb
    else:
        t, p, lo, hi = 0.0, 1.0, mean, mean
    return {"J": int(J), "mean": round(mean, 4), "se_nb": round(se_nb, 4),
            "t": round(float(t), 3), "p": round(float(p), 4),
            "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
            "n_pos": int((d > 0).sum()),
            "n_test": int(n_test), "n_train": int(n_train)}
