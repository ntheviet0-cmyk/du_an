"""redraw_roc_finetune.py - Ve lai roc_curves.png voi duong Graph fine-tune best.
RF/XGB: proba official tu models/ibm/*.joblib (cung split train_ibm seed 42).
Graph-ft: SAGE h64-drop0.3-lr0.01-wd5e-5, seed init 42009 (=42*1000+idx9 nhu tune_sage_grid),
          split/graph giong het train_ibm.py (kNN-cosine k=15, val SSS seed 42).
Luu proba de tai dung: reports/ibm/roc_proba.npz
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.neighbors import kneighbors_graph as kng_graph

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gnn_models import SAGE as SAGE_G

RANDOM_STATE = 42
FT_SEED = 42 * 1000 + 9  # idx 9 trong GRID 24 cau hinh (h64-drop0.3-lr0.01-wd5e-5)
K_SAGE = 15
OUT_DIR = Path("reports/ibm")
ART_DIR = Path("models/ibm")

# ---------- data & split giong het train_ibm.py ----------
df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
df["Attrition"] = (df["Attrition"] == "Yes").astype(int)
df = df.drop(columns=["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"])
CAT = ["BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime"]
NUM = [c for c in df.columns if c not in CAT + ["Attrition"]]
X = df[CAT + NUM]
y = df["Attrition"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

pre = joblib.load(ART_DIR / "preprocessor.joblib")
rf = joblib.load(ART_DIR / "random_forest.joblib")
xgbm = joblib.load(ART_DIR / "xgboost.joblib")
X_train_t = pre.transform(X_train)
X_test_t = pre.transform(X_test)

proba_rf = rf.predict_proba(X_test_t)[:, 1]
proba_xgb = xgbm.predict_proba(X_test_t)[:, 1]

# ---------- graph + SAGE fine-tune (khung train_ibm, seed init FT_SEED) ----------
X_all_t = np.vstack([X_train_t, X_test_t])
if hasattr(X_all_t, "toarray"):
    X_all_t = X_all_t.toarray()
X_all_t = X_all_t.astype(np.float32)
N_all = len(X_all_t)
n_tr = len(X_train_t)
y_all = np.concatenate([y_train, y_test])
idx_tr_all = np.arange(n_tr)
idx_te_all = np.arange(n_tr, N_all)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
idx_tr2a, idx_vaa = next(sss.split(idx_tr_all, y_train))

A_g = kng_graph(X_all_t, n_neighbors=K_SAGE, mode="connectivity",
                metric="cosine", include_self=False)
A_g = A_g.maximum(A_g.T)
ei_g = torch.tensor(np.vstack(A_g.nonzero()), dtype=torch.long)

torch.manual_seed(FT_SEED)
np.random.seed(FT_SEED)
Xt_g = torch.tensor(X_all_t)
yt_g = torch.tensor(y_all, dtype=torch.float32)
sage_m = SAGE_G(X_all_t.shape[1], h=64, drop=0.3)
opt_g = torch.optim.Adam(sage_m.parameters(), lr=0.01, weight_decay=5e-5)
pos_g = float((len(idx_tr2a) - y_all[idx_tr2a].sum()) / max(y_all[idx_tr2a].sum(), 1))
crit_g = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_g]))
best_pr, bs_g, bad_g = -1, None, 0
for ep_g in range(200):
    sage_m.train()
    opt_g.zero_grad()
    out_g = sage_m(Xt_g, ei_g)
    loss_g = crit_g(out_g[idx_tr2a], yt_g[idx_tr2a])
    loss_g.backward()
    opt_g.step()
    sage_m.eval()
    with torch.no_grad():
        pv_g = torch.sigmoid(sage_m(Xt_g, ei_g)[idx_vaa]).numpy()
    try:
        pr_v = float(average_precision_score(y_all[idx_vaa], pv_g))
    except Exception:
        pr_v = 0.0
    if pr_v > best_pr:
        best_pr, bs_g, bad_g = pr_v, {k: v.cpu().clone() for k, v in sage_m.state_dict().items()}, 0
    else:
        bad_g += 1
    if bad_g >= 30:
        break
sage_m.load_state_dict(bs_g)
sage_m.eval()
with torch.no_grad():
    proba_ft = torch.sigmoid(sage_m(Xt_g, ei_g)[idx_te_all]).numpy()

auc_rf = float(roc_auc_score(y_test, proba_rf))
auc_xgb = float(roc_auc_score(y_test, proba_xgb))
auc_ft = float(roc_auc_score(y_test, proba_ft))
print(f"[proba] RF AUC={auc_rf:.4f} XGB AUC={auc_xgb:.4f} Graph-ft AUC={auc_ft:.4f} (ep={ep_g + 1}, valPR={best_pr:.4f})")

np.savez(OUT_DIR / "roc_proba.npz", y_test=y_test, proba_rf=proba_rf,
         proba_xgb=proba_xgb, proba_graph_ft=proba_ft,
         auc_ft=auc_ft, ft_seed=FT_SEED)

# ---------- ve ROC ----------
plt.figure(figsize=(6, 5))
for proba, name, c in [(proba_rf, "RandomForest", "#1f77b4"),
                       (proba_xgb, "XGBoost", "#d62728"),
                       (proba_ft, "Graph fine-tune", "#2ca02c")]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} AUC={roc_auc_score(y_test, proba):.3f}", color=c, lw=2)
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: Random Forest vs XGBoost vs Graph (fine-tune)")
plt.legend(loc="lower right")
plt.figtext(0.5, 0.02, "Graph = best fine-tune SAGE (h=64 drop=0.3 lr=0.01 wd=5e-5, thr VAL); RF/XGBoost = official.",
            ha="center", fontsize=7, style="italic", color="#555555")
plt.subplots_adjust(bottom=0.2)
plt.savefig(OUT_DIR / "roc_curves.png", dpi=120)
plt.close()
print("saved reports/ibm/roc_curves.png + roc_proba.npz")
