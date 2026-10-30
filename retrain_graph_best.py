"""retrain_graph_best.py - Retrain GraphSAGE best-tune (h64-drop0.3-lr0.01-wd5e-5, seed 42009).

Khung train/val/test + graph giong het train_ibm.py (kNN-cosine k=15, split seed 42,
val SSS seed 42, early-stop val PR-AUC, nguong accuracy tren VAL nhu tune_sage_grid.py).
Doi chieu voi reports/ibm_gnn/sage_grid.json -> test_once (AUC~0.7901, CM [[236,11],[25,22]]).

Output:
  - backup models/ibm/graph.joblib -> models/ibm/graph_official.joblib (ban official seed 42)
  - models/ibm/graph.joblib moi (best-tune) + cap nhat models/ibm/meta.joblib
  - cap nhat reports/ibm/ibm_results.json -> models.Graph (giu official trong official_run)
"""
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.neighbors import kneighbors_graph as kng_graph

from gnn_models import SAGE as SAGE_G

RANDOM_STATE = 42
FT_SEED = 42 * 1000 + 9  # idx 9 trong GRID (h64-drop0.3-lr0.01-wd5e-5), nhu tune_sage_grid
K_SAGE = 15
REPO = Path(__file__).resolve().parent
ART_DIR = REPO / "models" / "ibm"
OUT_DIR = REPO / "reports" / "ibm"

# ---------------- data & split (giong het train_ibm.py) ----------------
df = pd.read_csv(REPO / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv")
df["Attrition"] = (df["Attrition"] == "Yes").astype(int)
df = df.drop(columns=["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"])
CAT = ["BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime"]
NUM = [c for c in df.columns if c not in CAT + ["Attrition"]]
X = df[CAT + NUM]
y = df["Attrition"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

pre = joblib.load(ART_DIR / "preprocessor.joblib")
X_train_t = pre.transform(X_train)
X_test_t = pre.transform(X_test)

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

# ---------------- train SAGE best-tune ----------------
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
print(f"[train] val_PR-AUC={best_pr:.4f} ep={ep_g + 1}")

with torch.no_grad():
    lg_all = sage_m(Xt_g, ei_g).numpy()
pv_va = torch.sigmoid(torch.tensor(lg_all[idx_vaa])).numpy()
pt_te = torch.sigmoid(torch.tensor(lg_all[idx_te_all])).numpy()

# nguong accuracy tren VAL (giong tune_sage_grid.best_thr_acc)
bt, ba = 0.5, -1.0
for t in np.arange(0.1, 0.91, 0.05):
    a = float(((pv_va >= t).astype(int) == y_all[idx_vaa]).mean())
    if a > ba:
        ba, bt = a, t
pred = (pt_te >= bt).astype(int)
res_graph = {
    "model": "Graph",
    "roc_auc": float(roc_auc_score(y_test, pt_te)),
    "pr_auc": float(average_precision_score(y_test, pt_te)),
    "precision": float(precision_score(y_test, pred, zero_division=0)),
    "recall": float(recall_score(y_test, pred, zero_division=0)),
    "f1": float(f1_score(y_test, pred, zero_division=0)),
    "val_f1": float(f1_score(y_all[idx_vaa], (pv_va >= bt).astype(int), zero_division=0)),
    "best_threshold": float(bt),
    "accuracy_default": float((((pt_te >= 0.5).astype(int)) == y_test).mean()),
    "test_acc_valthr": float(accuracy_score(y_test, pred)),
    "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    "n_test": int(len(y_test)),
    "n_pred_positive": int(pred.sum()),
    "seed_run": FT_SEED,
    "cfg": {"h": 64, "drop": 0.3, "lr": 0.01, "wd": 5e-5},
}
print(f"[Graph-best] thr_val={bt:.2f} ACC={res_graph['test_acc_valthr']:.4f} "
      f"AUC={res_graph['roc_auc']:.4f} PR={res_graph['pr_auc']:.4f} F1={res_graph['f1']:.4f} "
      f"P={res_graph['precision']:.3f} R={res_graph['recall']:.3f} CM={res_graph['confusion_matrix']}")

# ---------------- doi chieu sage_grid.json (log, cho phep noise GNN) ----------------
grid = json.loads((REPO / "reports" / "ibm_gnn" / "sage_grid.json").read_text(encoding="utf-8"))
ref = None
for t in grid.get("test_once", []):
    c = t.get("cfg", {})
    if c.get("h") == 64 and abs(c.get("drop", -1) - 0.3) < 1e-12 and abs(c.get("lr", -1) - 0.01) < 1e-12:
        ref = t
        break
assert ref is not None
print(f"[ref sage_grid] AUC={ref['test_auc']:.4f} PR={ref['test_pr']:.4f} F1={ref['test_f1']:.4f} CM={ref['cm']}")
assert res_graph["confusion_matrix"][1] == ref["cm"][1], "hang Roi di phai giong (TP/FN)"
assert abs(res_graph["roc_auc"] - ref["test_auc"]) < 0.02, "AUC lech qua 0.02 -> kiem tra lai"
print("[check] nam trong noise GNN cho phep (TP/FN giong, AUC ~= ref) -> lay so retrain thuc te")

# ---------------- bundle train-only + backup + luu ----------------
A_tr = kng_graph(np.array(X_train_t.toarray() if hasattr(X_train_t, "toarray") else X_train_t,
                          dtype=np.float32),
                 n_neighbors=K_SAGE, mode="connectivity", metric="cosine", include_self=False)
A_tr = A_tr.maximum(A_tr.T)
ei_tr = torch.tensor(np.vstack(A_tr.nonzero()), dtype=torch.long)
graph_bundle = {
    "kind": "sage",
    "sage_state": {k: v.cpu() for k, v in sage_m.state_dict().items()},
    "in_dim": int(X_all_t.shape[1]),
    "X_train": X_train_t.toarray().astype(np.float32) if hasattr(X_train_t, "toarray") else np.array(X_train_t, dtype=np.float32),
    "ei_train": ei_tr,
    "h": 64, "drop": 0.3,
    "seed_run": FT_SEED,
}
shutil.copy2(ART_DIR / "graph.joblib", ART_DIR / "graph_official.joblib")
print("[backup] graph.joblib -> graph_official.joblib")
joblib.dump(graph_bundle, ART_DIR / "graph.joblib")

meta = joblib.load(ART_DIR / "meta.joblib")
meta["official_graph"] = {"thr_graph": meta.get("thr_graph"), "metrics": meta["metrics"]["Graph"]}
meta["thr_graph"] = res_graph["best_threshold"]
meta["metrics"]["Graph"] = res_graph
meta["graph_params"] = {"model": "GraphSAGE-end2end", "graph": "knn-cosine",
                        "k": K_SAGE, "sage": "64-32", "lr": 0.01, "wd": 5e-5,
                        "drop": 0.3, "early_stop": "val-PR-AUC",
                        "seed_run": FT_SEED, "tuned": "best-tune h64-drop0.3 (test 1 lan)"}
joblib.dump(meta, ART_DIR / "meta.joblib")

rep_path = OUT_DIR / "ibm_results.json"
rep = json.loads(rep_path.read_text(encoding="utf-8"))
rep["models"]["Graph_official"] = rep["models"]["Graph"]
rep["models"]["Graph"] = res_graph
rep["best_model_policy"]["by_policy_main"] = (
    "Graph best-tune (SAGE h64-drop0.3-lr0.01-wd5e-5, seed 42009, k=15): "
    f"AUC {res_graph['roc_auc']:.4f}, PR {res_graph['pr_auc']:.4f}, F1 {res_graph['f1']:.4f}, "
    f"ACC {res_graph['test_acc_valthr']:.4f}; official (seed 42) giu trong Graph_official")
rep_path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
print("[done] saved graph.joblib (best-tune) + meta.joblib + ibm_results.json")
