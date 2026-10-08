"""Tuning RF/XGB voi ngan sach Optuna BANG NHAU. Chi dung train/val, khong cham test."""
from __future__ import annotations

import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest_rf(trial: optuna.Trial) -> dict:
    max_depth = trial.suggest_int("max_depth", 3, 20)
    if trial.suggest_float("max_depth_none_p", 0, 1) < 0.2:
        max_depth = None
    mf_choice = trial.suggest_categorical("max_features_choice", ["sqrt", "log2", "frac"])
    if mf_choice == "frac":
        mf = trial.suggest_float("max_features_frac", 0.3, 0.8)
    else:
        mf = mf_choice
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": max_depth,
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": mf,
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
        "random_state": 42,
        "n_jobs": -1,
    }


def suggest_xgb(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 8.0),
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
    }


def tune_model(kind: str, Xtr, ytr, Xva, yva, n_trials: int, seed=42):
    def objective(trial):
        params = suggest_rf(trial) if kind == "rf" else suggest_xgb(trial)
        clf = RandomForestClassifier(**params) if kind == "rf" else XGBClassifier(**params)
        clf.fit(Xtr, ytr)
        return float(average_precision_score(yva, clf.predict_proba(Xva)[:, 1]))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params
    if kind == "rf":
        mf = best["max_features_choice"]
        final = {
            "n_estimators": best["n_estimators"],
            "max_depth": best["max_depth"],
            "min_samples_leaf": best["min_samples_leaf"],
            "max_features": best["max_features_frac"] if mf == "frac" else mf,
            "class_weight": best["class_weight"],
            "random_state": 42,
            "n_jobs": -1,
        }
        clf = RandomForestClassifier(**final)
    else:
        final = {k: best[k] for k in
                 ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree",
                  "min_child_weight", "reg_lambda", "reg_alpha", "scale_pos_weight"]}
        final.update({"eval_metric": "logloss", "tree_method": "hist", "random_state": 42, "n_jobs": -1})
        clf = XGBClassifier(**final)
    clf.fit(Xtr, ytr)
    return clf, {"best_params": best, "best_val_pr": float(study.best_value)}
