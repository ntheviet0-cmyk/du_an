"""Buoc A: xac nhan ket qua am, KHONG tuning.
- Configs: G0, G1-20, G4-20 (G1-20 theo luat hoa: G2-10 cao nhat danh nghia +0.0005 = nhieu).
- 25 folds (5x5) x 1 seed (42), sieu tham so y GĐ3. Loss/threshold/val nhu GĐ3.
- Hieu so PR-AUC gep cap + Nadeau-Bengio t-test + CI95%.
- Quy tac co dinh truoc: neu can tren CI cua (G1-G0) VA (G1-G4) < +0.02
  -> 'khong co cai thien >= 0.02 trong pham vi thi nghiem'; nguoc lai 'chua ket luan duoc'.
- Log APPEND-ONLY: moi lan chay 1 run_id (timestamp); --resume <run_id> chi them dong thieu.
Out: results/tables/a_perfold_<runid>.csv, a_summary_<runid>.csv, results/logs/a_test_<runid>.json
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
from graphs import build_g0, nbr_to_edge_index, attach_test_pool_to_test  # noqa: E402
from graphs import cosine_sim_matrix, topk_from_sim, build_g4_from_g1, cosine_query_topk, g4u_query  # noqa: E402
from models import AttritionGAT  # noqa: E402
from train_gat import train_gat, predict_proba  # noqa: E402
from stats import nb_paired_test  # noqa: E402

SEED = 42
HP = {"hidden": 64, "heads": 4, "dropout": 0.3, "attn_dropout": 0.3,
      "lr": 3e-3, "wd": 1e-4, "max_epochs": 300, "patience": 30, "dropedge": 0.0}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRAPH_CHOICE_NOTE = ("G1-20 theo luat hoa: G2-10 cao nhat danh nghia o GĐ3 "
                     "(0.6123 vs 0.6118, chenh 0.0005 = nhieu) -> chon G1-20")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", default=None, help="run_id de noi tiep (append-only)")
    args = ap.parse_args()
    run_id = args.resume or time.strftime("%Y%m%d_%H%M%S")
    PF = ROOT / "results" / "tables" / f"a_perfold_{run_id}.csv"
    fieldnames = ["repeat", "fold", "gfold", "model", "graph", "k", "pr_auc", "roc_auc",
                  "mcc", "brier", "f1", "precision", "recall", "threshold",
                  "n_test", "n_pos_test", "val_pr", "epochs", "fit_time_s"]
    done = set()
    if PF.exists():
        import pandas as pd
        old = pd.read_csv(PF)
        for _, r in old.iterrows():
            done.add((int(r["gfold"]), str(r["graph"])))
    else:
        with open(PF, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()
    t0_all = time.time()

    df = load_raw(str(ROOT / "data" / "raw.csv"))
    y_all = df["Attrition"].values
    splits = list(outer_splits(y_all, n_splits=5, n_repeats=5, seed=SEED))
    assert len(splits) == 25
    print(f"[A:{run_id}] device={DEVICE} {GRAPH_CHOICE_NOTE}", flush=True)

    f = open(PF, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fieldnames)
    for gi, s in enumerate(splits):
        rep, fo = s["repeat"], s["fold"]
        df_pool = df.iloc[s["train_idx"]].reset_index(drop=True)
        df_test = df.iloc[s["test_idx"]].reset_index(drop=True)
        y_pool = df_pool["Attrition"].values
        y_test = df_test["Attrition"].values
        tri, vai = inner_val_split(y_pool, val_size=0.2, seed=SEED)
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
        S = cosine_sim_matrix(X_pool)
        g1_idx, _ = topk_from_sim(S, 20)

        cfgs = {"G0": None, "G1-20": ("g1", None), "G4-20": ("g4", None)}
        for gname in ["G0", "G1-20", "G4-20"]:
            if (gi, gname) in done:
                print(f"[skip] gfold{gi} {gname} (resume {run_id})", flush=True)
                continue
            t0 = time.time()
            if gname == "G0":
                pi, pw = build_g0(n_pool)
                ti, tw = build_g0(len(X_test))
            elif gname == "G1-20":
                pi, pw = topk_from_sim(S, 20)  # edge_attr = cosine sim (nhu GĐ3)
                ti, tw = cosine_query_topk(X_test, X_pool, 20)
            else:
                pi = build_g4_from_g1(g1_idx, seed=42000 + gi * 100 + 20)
                pw = np.ones_like(pi, dtype=float)
                ti, tw = g4u_query(len(X_test), n_pool, 20, seed=45000 + gi * 100 + 20)
            ei_pool, ea_pool = nbr_to_edge_index(pi, pw)
            ei_t, ea_t = attach_test_pool_to_test(ti, tw, n_pool)
            ei_all = torch.cat([ei_pool, ei_t], dim=1)
            ea_all = torch.cat([ea_pool, ea_t], dim=0)
            model = AttritionGAT(X_pool.shape[1], hidden=HP["hidden"], heads=HP["heads"],
                                 dropout=HP["dropout"], attn_dropout=HP["attn_dropout"])
            model, info_tr = train_gat(model, X_pool, y_pool, ei_pool, ea_pool,
                                      tr_mask, va_mask, lr=HP["lr"], wd=HP["wd"],
                                      max_epochs=HP["max_epochs"], patience=HP["patience"],
                                      seed=SEED, device=DEVICE)
            pva = predict_proba(model, X_pool, ei_pool, ea_pool, DEVICE)[va_mask]
            pte = predict_proba(model, X_all, ei_all, ea_all, DEVICE)[n_pool:]
            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            from sklearn.metrics import average_precision_score as _ap
            thr = pick_threshold_recall_ge(pva, y_pool[va_mask], 0.70)
            m = evaluate_test(y_test, pte, thr)
            dt = time.time() - t0
            row = {"repeat": rep, "fold": fo, "gfold": gi, "model": "gat", "graph": gname,
                   "k": ("-" if gname == "G0" else "20"),
                   **{kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in m.items()
                       if kk in fieldnames},
                   "val_pr": round(float(_ap(y_pool[va_mask], pva)), 4),
                   "epochs": info_tr["epochs"], "fit_time_s": round(dt, 1)}
            w.writerow(row)
            f.flush()
            print(f"[gfold{gi} {gname}] testPR={m['pr_auc']:.4f} ROC={m['roc_auc']:.4f} "
                  f"ep={info_tr['epochs']} ({dt:.0f}s)", flush=True)
    f.close()

    import pandas as pd
    pf = pd.read_csv(PF)
    assert len(pf) == 75, f"thieu dong: {len(pf)}/75"
    piv = pf.pivot(index="gfold", columns="graph", values="pr_auc")
    tests = {}
    for a, b in [("G1-20", "G0"), ("G1-20", "G4-20")]:
        d = (piv[a] - piv[b]).values
        tests[f"{a}_vs_{b}"] = nb_paired_test(d, n_test=294, n_train=1176)
        tests[f"{a}_vs_{b}"]["diffs"] = [round(float(v), 4) for v in d]
    summ = pf.groupby("graph")[["pr_auc", "roc_auc", "mcc", "brier"]].mean().round(4)
    summ.to_csv(ROOT / "results" / "tables" / f"a_summary_{run_id}.csv")
    up_g0 = tests["G1-20_vs_G0"]["ci_hi"]
    up_g4 = tests["G1-20_vs_G4-20"]["ci_hi"]
    conclusion = ("khong co cai thien >= 0.02 trong pham vi thi nghiem"
                  if (up_g0 < 0.02 and up_g4 < 0.02) else "chua ket luan duoc")
    with open(ROOT / "results" / "logs" / f"a_test_{run_id}.json", "w") as jf:
        json.dump({"run_id": run_id, "hp": HP, "seed": SEED, "n_folds": 25,
                   "graph_choice": GRAPH_CHOICE_NOTE, "device": DEVICE,
                   "nb_tests": tests,
                   "rule": "can tren CI (G1-G0) va (G1-G4) < +0.02 -> khong cai thien >=0.02",
                   "conclusion": conclusion,
                   "elapsed_s": round(time.time() - t0_all, 1)}, jf, indent=2)
    print(f"[A done] {time.time() - t0_all:.0f}s", flush=True)
    print(summ.to_string(), flush=True)
    print(json.dumps({k: {kk: v for kk, v in vv.items() if kk != "diffs"}
                       for k, vv in tests.items()}, indent=2), flush=True)
    print("CONCLUSION:", conclusion, flush=True)


if __name__ == "__main__":
    main()
