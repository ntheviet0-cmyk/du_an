"""Giai doan 2: chan doan homophily GO/NO-GO.
- Do thi: G1 (k=5,10,15,20,30), G3 (cung cap k), G4 (null giu bac theo tung k G1).
  Bo G0. G2 (Gower) HOAN sang Giai doan 3 — chua do, ghi ro trong log.
- Pool = outer-train (1176 nodes/fold), LOO trong pool, khong cham test.
- Preprocessor fit tren pool (khong dung test). Feature: base (goc); eng ablation o Giai doan 4.
- Seed G4: 42000 + fold_global*100 + k (co dinh, khac nhau moi fold/k).
Out: results/tables/homophily_perfold.csv, homophily_summary.csv,
     results/figs/homophily_{lift,roc}.png, results/logs/homophily.json
"""
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT / "src"))
from preprocess import AttritionPreprocessor, load_raw  # noqa: E402
from cv import outer_splits  # noqa: E402
from graphs import K_LIST, build_g1, build_g3, build_g4_from_g1, homophily_metrics  # noqa: E402

SEED = 42
T_CRIT = float(sp_stats.t.ppf(0.975, 24))  # 25 folds -> ~2.064

t0 = time.time()
df = load_raw(str(ROOT / "data" / "raw.csv"))
y_all = df["Attrition"].values
role_all = df["JobRole"].values
level_all = df["JobLevel"].values
splits = list(outer_splits(y_all, n_splits=5, n_repeats=5, seed=SEED))
assert len(splits) == 25

rows = []
for gi, s in enumerate(splits):
    df_pool = df.iloc[s["train_idx"]].reset_index(drop=True)
    y_pool = df_pool["Attrition"].values
    role_p = df_pool["JobRole"].values
    level_p = df_pool["JobLevel"].values
    pre = AttritionPreprocessor(use_engineered=False).fit(df_pool)
    X_pool = pre.transform(df_pool)
    if gi == 0:
        print(f"[feat] base dim={pre.n_features_}", flush=True)
    from graphs import cosine_sim_matrix, topk_from_sim
    S = cosine_sim_matrix(X_pool)
    for k in K_LIST:
        g1_idx, _ = topk_from_sim(S, k)
        g4_idx = build_g4_from_g1(g1_idx, seed=42000 + gi * 100 + k)
        g3_idx, _ = build_g3(X_pool, role_p, level_p, k)
        for gname, gidx in [("G1", g1_idx), ("G4", g4_idx), ("G3", g3_idx)]:
            m = homophily_metrics(gidx, y_pool)
            m.update({"repeat": s["repeat"], "fold": s["fold"], "graph": gname, "k": k})
            rows.append(m)
    if (gi + 1) % 5 == 0:
        print(f"[progress] {gi + 1}/25 folds ({time.time() - t0:.0f}s)", flush=True)

(ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
(ROOT / "results" / "figs").mkdir(parents=True, exist_ok=True)
(ROOT / "results" / "logs").mkdir(parents=True, exist_ok=True)
with open(ROOT / "results" / "tables" / "homophily_perfold.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# Tong hop: mean + CI95% (t, df=24) theo (graph, k)
import pandas as pd
pf = pd.DataFrame(rows)
METRICS = ["P1", "P0", "background", "lift", "frac_pos_no_pos_nbr",
           "edge_homophily", "knn_pr_auc", "knn_roc_auc", "mean_degree"]
summary = []
for (g, k), sub in pf.groupby(["graph", "k"]):
    r = {"graph": g, "k": int(k), "n_folds": int(len(sub))}
    for m in METRICS:
        v = sub[m].values.astype(float)
        mean, se = float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))
        r[f"{m}_mean"] = round(mean, 4)
        r[f"{m}_lo"] = round(mean - T_CRIT * se, 4)
        r[f"{m}_hi"] = round(mean + T_CRIT * se, 4)
    summary.append(r)
summ = pd.DataFrame(summary).sort_values(["graph", "k"])
summ.to_csv(ROOT / "results" / "tables" / "homophily_summary.csv", index=False)

# Quyet dinh GO/NO-GO (nguong co dinh truoc)
decisions = []
for k in K_LIST:
    g1 = summ[(summ.graph == "G1") & (summ.k == k)].iloc[0]
    g4 = summ[(summ.graph == "G4") & (summ.k == k)].iloc[0]
    c_lift = bool(g1["lift_mean"] >= 1.5 and (g1["lift_lo"] > g4["lift_mean"] or g1["lift_hi"] < g4["lift_mean"]))
    c_roc = bool(g1["knn_roc_auc_lo"] > g4["knn_roc_auc_hi"])
    decisions.append({"k": k, "lift>=1.5_va_CI_khong_chua_G4": c_lift,
                      "ROC_CI_khong_chong_lan_va_G1_cao_hon": c_roc,
                      "GO_tai_k": bool(c_lift and c_roc)})
GO = any(d["GO_tai_k"] for d in decisions)

# Kiem tra sanity G4 ~= nen
g4check = []
for k in K_LIST:
    g4 = summ[(summ.graph == "G4") & (summ.k == k)].iloc[0]
    g4check.append({"k": k, "G4_P1": g4["P1_mean"], "background": g4["background_mean"],
                    "G4_lift": g4["lift_mean"], "G4_ROC": g4["knn_roc_auc_mean"],
                    "gan_nen": bool(abs(g4["P1_mean"] - g4["background_mean"]) < 0.02
                                    and abs(g4["lift_mean"] - 1.0) < 0.15
                                    and abs(g4["knn_roc_auc_mean"] - 0.5) < 0.05)})

for d in decisions:
    g1 = summ[(summ.graph == "G1") & (summ.k == d["k"])].iloc[0]
    g4 = summ[(summ.graph == "G4") & (summ.k == d["k"])].iloc[0]
    print(f"[k={d['k']}] G1 lift={g1['lift_mean']:.3f} [{g1['lift_lo']:.3f},{g1['lift_hi']:.3f}] "
          f"vs G4 lift={g4['lift_mean']:.3f} | G1 ROC={g1['knn_roc_auc_mean']:.3f} "
          f"[{g1['knn_roc_auc_lo']:.3f},{g1['knn_roc_auc_hi']:.3f}] vs G4 ROC={g4['knn_roc_auc_mean']:.3f} "
          f"[{g4['knn_roc_auc_lo']:.3f},{g4['knn_roc_auc_hi']:.3f}] -> GO={d['GO_tai_k']}", flush=True)

# Ve hinh
for metric, title, fname in [("lift", "Lift P(nb=1|node=1)/nen (CI95%, 25 folds)", "homophily_lift"),
                             ("knn_roc_auc", "ROC-AUC diem kNN-label LOO (CI95%, 25 folds)", "homophily_roc")]:
    plt.figure(figsize=(7, 4.5))
    for g, c in [("G1", "#1f77b4"), ("G3", "#2ca02c"), ("G4", "#7f7f7f")]:
        sub = summ[summ.graph == g].sort_values("k")
        plt.errorbar(sub["k"], sub[f"{metric}_mean"],
                     yerr=[sub[f"{metric}_mean"] - sub[f"{metric}_lo"],
                           sub[f"{metric}_hi"] - sub[f"{metric}_mean"]],
                     marker="o", label=g, color=c, capsize=4)
    if metric == "lift":
        plt.axhline(1.5, ls="--", color="red", label="nguong GO lift=1.5")
        plt.axhline(1.0, ls=":", color="black", label="nen=1.0")
    else:
        plt.axhline(0.5, ls=":", color="black", label="ngau nhien=0.5")
    plt.xlabel("k")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "results" / "figs" / f"{fname}.png", dpi=130)
    plt.close()

with open(ROOT / "results" / "logs" / "homophily.json", "w") as f:
    json.dump({"seed": SEED, "n_folds": 25, "pool": "outer-train LOO",
               "features": "base (eng ablation o Giai doan 4)",
               "G2": "HOAN sang Giai doan 3 — chua do",
               "G0": "bo theo dieu chinh",
               "G4_seed_rule": "42000 + fold_global*100 + k",
               "t_crit_95_df24": round(T_CRIT, 4),
               "go_rule": "lift>=1.5 & CI95% khong chua muc G4 & ROC kNN-label CI khong chong lan & G1 cao hon, it nhat 1 k",
               "decisions": decisions, "G4_sanity": g4check,
               "GO": GO, "elapsed_s": round(time.time() - t0, 1)}, f, indent=2)
print(f"[homophily done] {time.time() - t0:.0f}s GO={GO}", flush=True)
