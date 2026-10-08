"""Giai doan 3: GAT co dinh (khong tuning) tren G0/G1/G2/G3/G4/G4u, k={10,20}.
- 5 folds (repeat 0, fold 0-4) x 3 seeds (42,43,44); trung binh seed trong fold.
- Train graph chi gom node pool (outer-train); test chi them luc suy luan (pool->test).
- Preprocessor + Gower range fit tren inner-train. RF/XGB tham chiếu: base, 3 trials.
- Cong GĐ4: voi it nhat 1 do thi that, mean diff PR >= +0.01 va duong o >=4/5 fold
  so voi CA G0, G4, G4u (khop k). Khong tuning/ablation/hybrid/ket luan o GĐ3.
Out: results/tables/g3_perfold.csv, g3_summary.csv, g3_gate.json, results/logs/g3_time.json
Resume: python attrition-gat/run_g3.py --resume
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT / "src"))
from preprocess import AttritionPreprocessor, load_raw  # noqa: E402
from cv import outer_splits, inner_val_split  # noqa: E402
from metrics import evaluate_test, pick_threshold_recall_ge  # noqa: E402
from tune_baseline import tune_model  # noqa: E402
from graphs import (build_g0, build_g1, build_g3, build_g4_from_g1, build_g4u,  # noqa: E402
                    cosine_sim_matrix, topk_from_sim, nbr_to_edge_index,
                    attach_test_pool_to_test, cosine_query_topk, g3_query_topk, g4u_query)
from gower import GowerKNN  # noqa: E402
from models import AttritionGAT  # noqa: E402
from train_gat import train_gat, predict_proba  # noqa: E402

SEED = 42
K_LIST = [10, 20]
GAT_SEEDS = [42, 43, 44]
HP = {"hidden": 64, "heads": 4, "dropout": 0.3, "attn_dropout": 0.3,
      "lr": 3e-3, "wd": 1e-4, "max_epochs": 300, "patience": 30, "dropedge": 0.0}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PERFOLD = ROOT / "results" / "tables" / "g3_perfold.csv"


def done_keys():
    done = set()
    if PERFOLD.exists():
        import pandas as pd
        df = pd.read_csv(PERFOLD)
        for _, r in df.iterrows():
            done.add((int(r["repeat"]), int(r["fold"]), str(r["model"]), str(r["graph"]), str(r["k"])))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    t0_all = time.time()
    if not args.resume and PERFOLD.exists():
        PERFOLD.unlink()  # tranh header stale tu lan crash truoc
    done = done_keys() if args.resume else set()

    df = load_raw(str(ROOT / "data" / "raw.csv"))
    y_all = df["Attrition"].values
    splits = list(outer_splits(y_all, n_splits=5, n_repeats=5, seed=SEED))[:5]
    print(f"[g3] device={DEVICE} folds={len(splits)} seeds={GAT_SEEDS} hp={HP}", flush=True)

    fieldnames = ["repeat", "fold", "model", "graph", "k", "pr_auc", "roc_auc", "mcc",
                  "brier", "f1", "precision", "recall", "threshold", "p_at_10", "p_at_20",
                  "n_test", "n_pos_test", "n_pred_pos", "val_pr_mean", "epochs_mean", "fit_time_s"]
    new_file = not PERFOLD.exists()
    f = open(PERFOLD, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fieldnames)
    if new_file:
        w.writeheader()

    timings = {}
    for gi, s in enumerate(splits):
        rep, fo = s["repeat"], s["fold"]
        df_pool = df.iloc[s["train_idx"]].reset_index(drop=True)
        df_test = df.iloc[s["test_idx"]].reset_index(drop=True)
        y_pool = df_pool["Attrition"].values
        y_test = df_test["Attrition"].values
        tri, vai = inner_val_split(y_pool, val_size=0.2, seed=SEED)
        pool_pos = np.zeros(len(y_pool), bool)
        tr_mask = np.zeros(len(y_pool), bool)
        va_mask = np.zeros(len(y_pool), bool)
        tr_mask[tri] = True
        va_mask[vai] = True
        df_tr = df_pool.iloc[tri].reset_index(drop=True)

        pre = AttritionPreprocessor(use_engineered=False).fit(df_tr)
        X_pool = pre.transform(df_pool)
        X_test = pre.transform(df_test)
        X_all = np.vstack([X_pool, X_test]).astype(np.float32)
        n_pool = len(X_pool)
        if gi == 0:
            print(f"[feat] base dim={pre.n_features_}", flush=True)
        role_p = df_pool["JobRole"].values
        level_p = df_pool["JobLevel"].values
        role_t = df_test["JobRole"].values
        level_t = df_test["JobLevel"].values
        S = cosine_sim_matrix(X_pool)
        gow = GowerKNN().fit(df_tr)

        # ---- RF/XGB tham chieu (base, 3 trials) ----
        for kind in ["rf", "xgb"]:
            key = (rep, fo, kind, "-", "-")
            if key in done:
                print(f"[skip] fold{fo} {kind} (resume)", flush=True)
                continue
            t0 = time.time()
            clf, info = tune_model(kind, X_pool[tr_mask], y_pool[tr_mask],
                                   X_pool[va_mask], y_pool[va_mask], n_trials=3, seed=SEED)
            pva = clf.predict_proba(X_pool[va_mask])[:, 1]
            pte = clf.predict_proba(X_test)[:, 1]
            thr = pick_threshold_recall_ge(pva, y_pool[va_mask], 0.70)
            m = evaluate_test(y_test, pte, thr)
            dt = time.time() - t0
            timings[f"{kind}_f{fo}"] = round(dt, 1)
            w.writerow({"repeat": rep, "fold": fo, "model": kind, "graph": "-", "k": "-",
                        **{kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in m.items()},
                        "val_pr_mean": round(float(info["best_val_pr"]), 4),
                        "epochs_mean": "-", "fit_time_s": round(dt, 1)})
            f.flush()
            print(f"[fold{fo} {kind}] valPR={info['best_val_pr']:.4f} testPR={m['pr_auc']:.4f} "
                  f"ROC={m['roc_auc']:.4f} MCC={m['mcc']:.4f} Brier={m['brier']:.4f} ({dt:.0f}s)", flush=True)

        # ---- GAT configs ----
        configs = [("G0", None)]
        for k in K_LIST:
            for g in ["G1", "G2", "G3", "G4", "G4u"]:
                configs.append((g, k))
        for gname, k in configs:
            kstr = "-" if k is None else str(k)
            key = (rep, fo, "gat", gname, kstr)
            if key in done:
                print(f"[skip] fold{fo} gat-{gname}{kstr} (resume)", flush=True)
                continue
            t0 = time.time()
            if gname == "G0":
                pi, pw = build_g0(n_pool)
                ti, tw = build_g0(len(X_test))  # rong: test chi co self-loop tu conv
            elif gname == "G1":
                pi, pw = topk_from_sim(S, k)
                ti, tw = cosine_query_topk(X_test, X_pool, k)
            elif gname == "G2":
                pi, pw = gow.pool_neighbors(df_pool, k)
                ti, tw = gow.query_neighbors(df_test, df_pool, k)
            elif gname == "G3":
                pi, pw = build_g3(X_pool, role_p, level_p, k)
                ti, tw = g3_query_topk(X_test, X_pool, role_t, role_p, level_t, level_p, k)
            elif gname == "G4":
                g1_idx, _ = topk_from_sim(S, k)
                pi = build_g4_from_g1(g1_idx, seed=42000 + gi * 100 + k)
                pw = np.ones_like(pi, dtype=float)
                ti, tw = g4u_query(len(X_test), n_pool, k, seed=45000 + gi * 100 + k)
            elif gname == "G4u":
                pi, pw = build_g4u(n_pool, k, seed=43000 + gi * 100 + k)
                ti, tw = g4u_query(len(X_test), n_pool, k, seed=44000 + gi * 100 + k)
            ei_pool, ea_pool = nbr_to_edge_index(pi, pw)
            ei_t, ea_t = attach_test_pool_to_test(ti, tw, n_pool)
            import torch as _t
            ei_all = _t.cat([ei_pool, ei_t], dim=1)
            ea_all = _t.cat([ea_pool, ea_t], dim=0)

            pva_seeds, pte_seeds, eps = [], [], []
            for sd in GAT_SEEDS:
                model = AttritionGAT(X_pool.shape[1], hidden=HP["hidden"], heads=HP["heads"],
                                     dropout=HP["dropout"], attn_dropout=HP["attn_dropout"])
                model, info_tr = train_gat(model, X_pool, y_pool, ei_pool, ea_pool,
                                           tr_mask, va_mask, lr=HP["lr"], wd=HP["wd"],
                                           max_epochs=HP["max_epochs"], patience=HP["patience"],
                                           seed=sd, device=DEVICE)
                eps.append(info_tr["epochs"])
                pva_seeds.append(predict_proba(model, X_pool, ei_pool, ea_pool, DEVICE)[va_mask])
                pte_seeds.append(predict_proba(model, X_all, ei_all, ea_all, DEVICE)[n_pool:])
                del model
                if DEVICE == "cuda":
                    _t.cuda.empty_cache()
            pva_m = np.mean(pva_seeds, axis=0)
            pte_m = np.mean(pte_seeds, axis=0)
            thr = pick_threshold_recall_ge(pva_m, y_pool[va_mask], 0.70)
            m = evaluate_test(y_test, pte_m, thr)
            dt = time.time() - t0
            timings[f"gat{gname}{kstr}_f{fo}"] = round(dt, 1)
            from sklearn.metrics import average_precision_score as _ap
            w.writerow({"repeat": rep, "fold": fo, "model": "gat", "graph": gname, "k": kstr,
                        **{kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in m.items()},
                        "val_pr_mean": round(float(_ap(y_pool[va_mask], pva_m)), 4),
                        "epochs_mean": round(float(np.mean(eps)), 1), "fit_time_s": round(dt, 1)})
            f.flush()
            print(f"[fold{fo} gat-{gname}{kstr}] valPR={_ap(y_pool[va_mask], pva_m):.4f} "
                  f"testPR={m['pr_auc']:.4f} ROC={m['roc_auc']:.4f} MCC={m['mcc']:.4f} "
                  f"Brier={m['brier']:.4f} ep={np.mean(eps):.0f} ({dt:.0f}s)", flush=True)
    f.close()

    # ---- Tong hop + cong GĐ4 ----
    import pandas as pd
    pf = pd.read_csv(PERFOLD)
    pf = pf[(pf.repeat == 0)].copy()
    g = pf[pf.model == "gat"].copy()
    ctrl = {c: g[g.graph == c].set_index(["fold", "k"])["pr_auc"] for c in ["G0"]}
    gate_rows = []
    for gname in ["G1", "G2", "G3"]:
        for k in [str(kk) for kk in K_LIST]:
            sub = g[(g.graph == gname) & (g.k == k)].set_index("fold").sort_index()
            if len(sub) < 5:
                continue
            res = {"graph": gname, "k": k, "mean_pr": round(float(sub.pr_auc.mean()), 4),
                   "folds": ",".join(f"{v:.4f}" for v in sub.pr_auc.values)}
            for c in ["G0", "G4", "G4u"]:
                if c == "G0":
                    cc = g[g.graph == "G0"].set_index("fold").sort_index()["pr_auc"]
                else:
                    cc = g[(g.graph == c) & (g.k == k)].set_index("fold").sort_index()["pr_auc"]
                d = (sub["pr_auc"].values - cc.values)
                res[f"diff_{c}_mean"] = round(float(d.mean()), 4)
                res[f"npos_{c}"] = int((d > 0).sum())
            gate_rows.append(res)
    gate = pd.DataFrame(gate_rows)
    gate["pass"] = ((gate.diff_G0_mean >= 0.01) & (gate.npos_G0 >= 4)
                    & (gate.diff_G4_mean >= 0.01) & (gate.npos_G4 >= 4)
                    & (gate.diff_G4u_mean >= 0.01) & (gate.npos_G4u >= 4))
    gate.to_csv(ROOT / "results" / "tables" / "g3_gate.csv", index=False)
    summ = pf.groupby(["model", "graph", "k"])[["pr_auc", "roc_auc", "mcc", "brier"]].mean().round(4)
    summ.to_csv(ROOT / "results" / "tables" / "g3_summary.csv")
    GO = bool(gate["pass"].any()) if len(gate) else False
    with open(ROOT / "results" / "logs" / "g3_gate.json", "w") as jf:
        json.dump({"hp": HP, "gat_seeds": GAT_SEEDS, "device": DEVICE,
                   "gate_rule": "mean diff PR>=+0.01 & duong >=4/5 folds vs CA G0,G4,G4u (khop k)",
                   "GO": GO, "elapsed_s": round(time.time() - t0_all, 1)}, jf, indent=2)
    with open(ROOT / "results" / "logs" / "g3_time.json", "w") as jf:
        json.dump({"timings_s": timings, "total_s": round(time.time() - t0_all, 1),
                   "device": DEVICE}, jf, indent=2)
    print(f"[g3 done] {time.time() - t0_all:.0f}s GO={GO}", flush=True)
    print(gate.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
