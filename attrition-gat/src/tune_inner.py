"""Tuning long nhau B1 (nghiem ngat): Optuna toi da hoa PR-AUC trung binh inner 3-fold CV.
Preprocessor fit RIENG trong moi inner-train split (khong thay inner-val) — dung df tho.
Space giu nguyen configs/rf.yaml, xgb.yaml.
"""
from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from tune_baseline import suggest_rf, suggest_xgb
from preprocess import AttritionPreprocessor

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _make(kind: str, params: dict):
    if kind == "rf":
        return RandomForestClassifier(**params)
    return XGBClassifier(**params)


def _finalize_params(kind: str, best: dict) -> dict:
    if kind == "rf":
        mf = best["max_features_choice"]
        md = None if best.get("max_depth_none_p", 1.0) < 0.2 else best["max_depth"]
        return {"n_estimators": best["n_estimators"], "max_depth": md,
                "min_samples_leaf": best["min_samples_leaf"],
                "max_features": best["max_features_frac"] if mf == "frac" else mf,
                "class_weight": best["class_weight"], "random_state": 42, "n_jobs": -1}
    return {"n_estimators": best["n_estimators"], "max_depth": best["max_depth"],
            "learning_rate": best["learning_rate"], "subsample": best["subsample"],
            "colsample_bytree": best["colsample_bytree"],
            "min_child_weight": best["min_child_weight"], "reg_lambda": best["reg_lambda"],
            "reg_alpha": best["reg_alpha"], "scale_pos_weight": best["scale_pos_weight"],
            "eval_metric": "logloss", "tree_method": "hist", "random_state": 42, "n_jobs": -1}


def _split_mats(df_out: pd.DataFrame, y_out: np.ndarray, tri, vai, use_engineered=False):
    pre = AttritionPreprocessor(use_engineered=use_engineered).fit(
        df_out.iloc[tri].reset_index(drop=True))
    return (pre.transform(df_out.iloc[tri]), y_out[tri],
            pre.transform(df_out.iloc[vai]), y_out[vai])


def tune_inner_cv(kind: str, df_out: pd.DataFrame, y_out: np.ndarray,
                  n_trials: int = 50, seed: int = 42, use_engineered: bool = False):
    """Tra (best_params, best_inner_pr, oof_proba). Preprocessor fit moi inner split."""
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    inner = list(skf.split(np.zeros(len(y_out)), y_out))
    mats = [_split_mats(df_out, y_out, tri, vai, use_engineered) for tri, vai in inner]

    def objective(trial):
        params = suggest_rf(trial) if kind == "rf" else suggest_xgb(trial)
        prs = []
        for Xtr, ytr, Xva, yva in mats:
            clf = _make(kind, params)
            clf.fit(Xtr, ytr)
            prs.append(average_precision_score(yva, clf.predict_proba(Xva)[:, 1]))
        return float(np.mean(prs))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    final = _finalize_params(kind, study.best_params)
    oof = np.zeros(len(y_out))
    for (tri, vai), (Xtr, ytr, Xva, yva) in zip(inner, mats):
        clf = _make(kind, final)
        clf.fit(Xtr, ytr)
        oof[vai] = clf.predict_proba(Xva)[:, 1]
    return final, float(study.best_value), oof
