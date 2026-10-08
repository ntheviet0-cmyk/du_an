"""Smoke-test Giai doan 1: 1 fold (repeat 0, fold 0), Optuna 3 trials/model, feature base+eng.
Do thoi gian de uoc luong full-run 25 folds x 50 trials.
Out: results/tables/smoke_baseline.csv, results/logs/smoke_time.json, results/folds/folds.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT / "src"))
from preprocess import AttritionPreprocessor, load_raw  # noqa: E402
from cv import outer_splits, inner_val_split, save_folds  # noqa: E402
from metrics import evaluate_test, pick_threshold_recall_ge  # noqa: E402
from tune_baseline import tune_model  # noqa: E402

SEED = 42
N_TRIALS = 3  # smoke; full-run 50

t0_all = time.time()
df = load_raw(str(ROOT / "data" / "raw.csv"))
y = df["Attrition"].values
save_folds(y, str(ROOT / "results" / "folds" / "folds.json"), n_splits=5, n_repeats=5, seed=SEED)
splits = list(outer_splits(y, n_splits=5, n_repeats=5, seed=SEED))
s = splits[0]  # repeat 0 fold 0
print(f"[smoke] repeat={s['repeat']} fold={s['fold']} train={len(s['train_idx'])} test={len(s['test_idx'])}", flush=True)

df_tr_full = df.iloc[s["train_idx"]].reset_index(drop=True)
df_te = df.iloc[s["test_idx"]].reset_index(drop=True)
y_tr_full = df_tr_full["Attrition"].values
y_te = df_te["Attrition"].values
tri, vai = inner_val_split(y_tr_full, val_size=0.2, seed=SEED)
df_tr, df_va = df_tr_full.iloc[tri].reset_index(drop=True), df_tr_full.iloc[vai].reset_index(drop=True)

rows = []
timings = {}
for use_eng in [False, True]:
    tag = "eng" if use_eng else "base"
    pre = AttritionPreprocessor(use_engineered=use_eng).fit(df_tr)
    Xtr, Xva, Xte = pre.transform(df_tr), pre.transform(df_va), pre.transform(df_te)
    print(f"[feat:{tag}] dim={pre.n_features_} (num={len(pre.num_cols_)} ohe={len(pre.ohe_feature_names_)})", flush=True)
    for kind in ["rf", "xgb"]:
        t0 = time.time()
        clf, info = tune_model(kind, Xtr, df_tr["Attrition"].values, Xva, df_va["Attrition"].values,
                               n_trials=N_TRIALS, seed=SEED)
        dt = time.time() - t0
        timings[f"{kind}_{tag}"] = round(dt, 1)
        pva = clf.predict_proba(Xva)[:, 1]
        pte = clf.predict_proba(Xte)[:, 1]
        thr = pick_threshold_recall_ge(pva, df_va["Attrition"].values, 0.70)
        m = evaluate_test(y_te, pte, thr)
        m.update({"model": kind, "features": tag, "val_pr": round(float(info["best_val_pr"]), 4),
                  "trial_n": N_TRIALS, "fit_time_s": round(dt, 1),
                  "best_params": json.dumps(info["best_params"], default=str)})
        rows.append(m)
        print(f"[{kind}:{tag}] valPR={info['best_val_pr']:.4f} thr={thr:.2f} "
              f"test PR={m['pr_auc']:.4f} ROC={m['roc_auc']:.4f} MCC={m['mcc']:.4f} "
              f"Brier={m['brier']:.4f} F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} "
              f"P@10={m['p_at_10']:.3f} time={dt:.1f}s", flush=True)

import csv
(ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
(ROOT / "results" / "logs").mkdir(parents=True, exist_ok=True)
with open(ROOT / "results" / "tables" / "smoke_baseline.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
total = time.time() - t0_all
# Uoc luong full-run: 25 folds x (50/3 trials) x 2 models x 2 feature sets,
# cong them overhead val/GAT sau nay (GAT rieng, chua tinh o day).
per_cfg = np.mean(list(timings.values()))
est_full_baseline_h = per_cfg * (50 / N_TRIALS) * 25 * 2 * 2 / 3600
with open(ROOT / "results" / "logs" / "smoke_time.json", "w") as f:
    json.dump({"timings_s": timings, "total_smoke_s": round(total, 1),
               "note": "1 fold, 3 trials/model/feat; full=25 folds x 50 trials",
               "est_full_baseline_CPU_h": round(float(est_full_baseline_h), 2),
               "warning": "day la uoc luong tuyen tinh; Optuna TPE + XGB early behavior co the lech; GAT (Giai doan 3-4) tinh rieng vi gap ~3 seeds x epochs"},
              f, indent=2)
print(f"[smoke done] total={total:.1f}s est_full_baseline~{est_full_baseline_h:.2f} CPU-h (tuyen tinh)", flush=True)
