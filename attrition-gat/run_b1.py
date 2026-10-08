"""Buoc B1: RF + XGBoost, feature GOC, 25 folds, Optuna long nhau 50 trials.
- Ham muc tieu: mean PR-AUC inner 3-fold CV tren outer-train (thay val split 20%).
- Calibration + nguong recall>=70%: tren OOF cua inner 3-fold (khong dung lai tap tuning);
  refit best params tren full outer-train roi du doan test.
- Platt chinh; isotonic do nhay. KHONG tuning GAT. GAT-G0 chi tham chieu tu buoc A.
- Log APPEND-ONLY theo run_id; --resume <run_id> noi tiep. --trial: chay thu 1 fold do gio.
Out: results/tables/b1_perfold_<runid>.csv, b1_params_<runid>.jsonl,
     b1_summary_<runid>.csv, results/logs/b1_report_<runid>.json
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT / "src"))
from preprocess import AttritionPreprocessor, load_raw  # noqa: E402
from cv import outer_splits  # noqa: E402
from metrics import evaluate_test, pick_threshold_recall_ge, precision_at_top_k  # noqa: E402
from tune_inner import tune_inner_cv, _make  # noqa: E402
from calibrate import fit_platt, fit_isotonic, apply_platt, apply_isotonic  # noqa: E402
from stats import nb_paired_test  # noqa: E402
from sklearn.metrics import brier_score_loss  # noqa: E402

SEED = 42
N_TRIALS = 50

FIELDS = (["repeat", "fold", "gfold", "model", "pr_auc", "roc_auc", "p_at_10", "p_at_20",
           "n_test", "n_pos_test", "inner_pr", "fit_time_s"]
          + [f"{m}_{v}" for v in ["raw", "platt", "iso"]
             for m in ["mcc", "f1", "precision", "recall", "threshold", "brier"]])


def run_fold_model(df_pool, df_test, kind, seed_inner, t0):
    y_out = df_pool["Attrition"].values
    y_test = df_test["Attrition"].values
    # Tuning + OOF: preprocessor fit rieng moi inner split (trong tune_inner_cv).
    best, inner_pr, oof = tune_inner_cv(kind, df_pool, y_out, n_trials=N_TRIALS, seed=seed_inner)
    platt = fit_platt(oof, y_out)
    iso = fit_isotonic(oof, y_out)
    # Refit final: preprocessor fit tren outer-train (test khong cham), model fit full outer-train.
    pre = AttritionPreprocessor(use_engineered=False).fit(df_pool)
    X_out = pre.transform(df_pool)
    X_test = pre.transform(df_test)
    clf = _make(kind, best)
    clf.fit(X_out, y_out)
    p_raw = clf.predict_proba(X_test)[:, 1]
    p_platt = apply_platt(platt, p_raw)
    p_iso = apply_isotonic(iso, p_raw)
    o_platt = apply_platt(platt, oof)
    o_iso = apply_isotonic(iso, oof)
    row = {"pr_auc": 0.0, "roc_auc": 0.0}  # placeholder, dien sau
    from sklearn.metrics import average_precision_score, roc_auc_score
    row["pr_auc"] = round(float(average_precision_score(y_test, p_raw)), 4)
    row["roc_auc"] = round(float(roc_auc_score(y_test, p_raw)), 4)
    row["p_at_10"] = round(float(precision_at_top_k(y_test, p_raw, 0.10)), 4)
    row["p_at_20"] = round(float(precision_at_top_k(y_test, p_raw, 0.20)), 4)
    for v, pv, ov in [("raw", p_raw, oof), ("platt", p_platt, o_platt), ("iso", p_iso, o_iso)]:
        thr = pick_threshold_recall_ge(ov, y_out, 0.70)
        pred = (pv >= thr).astype(int)
        from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score
        row[f"mcc_{v}"] = round(float(matthews_corrcoef(y_test, pred)), 4)
        row[f"f1_{v}"] = round(float(f1_score(y_test, pred, zero_division=0)), 4)
        row[f"precision_{v}"] = round(float(precision_score(y_test, pred, zero_division=0)), 4)
        row[f"recall_{v}"] = round(float(recall_score(y_test, pred, zero_division=0)), 4)
        row[f"threshold_{v}"] = round(float(thr), 4)
        row[f"brier_{v}"] = round(float(brier_score_loss(y_test, pv)), 4)
    return row, best, inner_pr, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", default=None)
    ap.add_argument("--trial", action="store_true", help="chay thu 1 fold do thoi gian")
    args = ap.parse_args()
    run_id = args.resume or time.strftime("%Y%m%d_%H%M%S")
    PF = ROOT / "results" / "tables" / f"b1_perfold_{run_id}.csv"
    PJ = ROOT / "results" / "tables" / f"b1_params_{run_id}.jsonl"
    done = set()
    if PF.exists():
        import pandas as pd
        for _, r in pd.read_csv(PF).iterrows():
            done.add((int(r["gfold"]), str(r["model"])))
    else:
        with open(PF, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    t0_all = time.time()

    df = load_raw(str(ROOT / "data" / "raw.csv"))
    y_all = df["Attrition"].values
    splits = list(outer_splits(y_all, n_splits=5, n_repeats=5, seed=SEED))
    folds = [(0, splits[0])] if args.trial else list(enumerate(splits))
    print(f"[B1:{run_id}] trial={args.trial} folds={len(folds)} trials={N_TRIALS}", flush=True)

    f = open(PF, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    for gi, s in folds:
        df_pool = df.iloc[s["train_idx"]].reset_index(drop=True)
        df_test = df.iloc[s["test_idx"]].reset_index(drop=True)
        for kind in ["rf", "xgb"]:
            if (gi, kind) in done:
                print(f"[skip] gfold{gi} {kind} (resume)", flush=True)
                continue
            t0 = time.time()
            row, best, inner_pr, dt = run_fold_model(df_pool, df_test, kind, SEED + gi, t0)
            row.update({"repeat": s["repeat"], "fold": s["fold"], "gfold": gi, "model": kind,
                        "n_test": len(df_test), "n_pos_test": int(df_test["Attrition"].sum()),
                        "inner_pr": round(inner_pr, 4), "fit_time_s": round(dt, 1)})
            w.writerow(row)
            f.flush()
            with open(PJ, "a") as jf:
                jf.write(json.dumps({"gfold": gi, "model": kind, "seed_inner": SEED + gi,
                                     "best_params": best}, default=str) + "\n")
            print(f"[gfold{gi} {kind}] innerPR={inner_pr:.4f} testPR={row['pr_auc']:.4f} "
                  f"Brier raw/platt/iso={row['brier_raw']:.4f}/{row['brier_platt']:.4f}/"
                  f"{row['brier_iso']:.4f} ({dt:.0f}s)", flush=True)
    f.close()
    total = time.time() - t0_all
    print(f"[B1:{run_id}] total={total:.0f}s", flush=True)
    if args.trial:
        est = total * 25
        print(f"[trial] 1 fold = {total:.0f}s -> uoc full 25 folds ~ {est/3600:.1f} CPU-h", flush=True)
        with open(ROOT / "results" / "logs" / f"b1_trial_{run_id}.json", "w") as jf:
            json.dump({"trial_s": round(total, 1), "est_full_s": round(est, 1),
                       "est_full_h": round(est / 3600, 2)}, jf, indent=2)
        return

    import pandas as pd
    pf = pd.read_csv(PF)
    assert len(pf) == 50, f"thieu dong: {len(pf)}/50"
    nb = {}
    for model in ["rf", "xgb"]:
        sub = pf[pf.model == model]
        nb[model] = {}
        for m in ["pr_auc", "roc_auc", "brier_raw", "brier_platt", "brier_iso"]:
            r = nb_paired_test(sub[m].values, 294, 1176)  # CI quanh mean (mo ta + hieu chinh NB)
            nb[model][m] = {"mean": r["mean"], "sd": round(float(sub[m].std(ddof=1)), 4),
                            "nb_ci_lo": r["ci_lo"], "nb_ci_hi": r["ci_hi"]}
    pf.groupby("model")[["pr_auc", "roc_auc"]].mean().round(4).to_csv(
        ROOT / "results" / "tables" / f"b1_summary_{run_id}.csv")
    with open(ROOT / "results" / "logs" / f"b1_report_{run_id}.json", "w") as jf:
        json.dump({"run_id": run_id, "n_trials": N_TRIALS, "features": "base",
                   "inner": "StratifiedKFold(3) mean PR-AUC, seed=42+gfold",
                   "calibration": "Platt chinh (LogReg tren OOF), isotonic do nhay",
                   "threshold": "recall>=70% tren OOF (raw/platt/iso rieng), refit full outer-train",
                   "gat_g0_note": ("Tham chieu tu buoc A (25 folds, 1 seed, HP co dinh): "
                                   "GAT-G0 la NN khong do thi, CHUA tuning, KHONG cung ngan sach "
                                   "— khong dung de ket luan ve do thi."),
                   "nb_mean_ci": nb, "elapsed_s": round(total, 1)}, jf, indent=2)
    print(f"[B1 done] run {run_id}", flush=True)


if __name__ == "__main__":
    main()
