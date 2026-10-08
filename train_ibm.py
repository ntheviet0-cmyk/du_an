"""
train_ibm.py - Employee Attrition Prediction & Causal Graph on the IBM HR Analytics
dataset (WA_Fn-UseC_-HR-Employee-Attrition.csv).

Report deliverables:
  PRIORITY 1 : Random Forest  (SMOTE + 5-fold CV tuning, threshold optimization)
  PRIORITY 2 : XGBoost        (SMOTE + scale_pos_weight + 5-fold CV, threshold optimization)
  PRIORITY 3 : Graph (predictive, OFFICIAL) - GraphSAGE end-to-end (kNN + SAGE)
  PRIORITY 4 : Causal DAG via DoWhy: OverTime -> Attrition + 3-model comparison figure

Outputs: reports/ibm/{roc_curves.png, metrics_bar.png, graph_similarity.png, attrition_dag.png, ibm_results.json}
"""
import json
import joblib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, precision_recall_curve,
)
from imblearn.over_sampling import SMOTE

import xgboost as xgb
import shap
import dowhy
from dowhy import CausalModel
import networkx as nx
from sklearn.neighbors import kneighbors_graph

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_PATH = Path("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
OUT_DIR = Path("reports/ibm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 0. Load & prepare
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
df["Attrition"] = (df["Attrition"] == "Yes").astype(int)

DROP = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
df = df.drop(columns=DROP)

CAT = ["BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime"]
NUM = [c for c in df.columns if c not in CAT + ["Attrition"]]

X = df[CAT + NUM]
y = df["Attrition"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

pre = ColumnTransformer(
    [("num", StandardScaler(), NUM),
     ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)],
    remainder="drop",
)
pre.fit(X_train)
feat_names = pre.get_feature_names_out()
X_train_t = pre.transform(X_train)
X_test_t = pre.transform(X_test)

print(f"[data] rows={len(df)}  positive(Attrition=1)={int(y.sum())} "
      f"({y.mean()*100:.1f}%)  train={len(y_train)} test={len(y_test)}")

# ---------------------------------------------------------------------------
# Helper: threshold optimization & evaluation
# ---------------------------------------------------------------------------
def evaluate(model, Xt, yt, name):
    proba = model.predict_proba(Xt)[:, 1]
    base = 0.5
    # optimal F1 threshold
    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.1, 0.9, 0.05):
        pred = (proba >= t).astype(int)
        f1 = f1_score(yt, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    pred = (proba >= best_t).astype(int)
    res = {
        "model": name,
        "roc_auc": float(roc_auc_score(yt, proba)),
        "pr_auc": float(average_precision_score(yt, proba)),
        "precision": float(precision_score(yt, pred, zero_division=0)),
        "recall": float(recall_score(yt, pred, zero_division=0)),
        "f1": float(best_f1),
        "best_threshold": float(best_t),
        "confusion_matrix": confusion_matrix(yt, pred).tolist(),
        "n_test": int(len(yt)),
        "n_pred_positive": int(pred.sum()),
    }
    print(f"[{name}] ROC-AUC={res['roc_auc']:.4f} PR-AUC={res['pr_auc']:.4f} "
          f"F1={res['f1']:.4f} (thr={res['best_threshold']:.2f}) "
          f"P={res['precision']:.3f} R={res['recall']:.3f}")
    return res, proba, pred


# ---------------------------------------------------------------------------
# PRIORITY 1: Random Forest
# ---------------------------------------------------------------------------
print("\n=== PRIORITY 1: RANDOM FOREST ===")
pipe_rf = Pipeline([
    ("smote", SMOTE(random_state=RANDOM_STATE)),
    ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
])
grid_rf = {"clf__n_estimators": [200, 400],
           "clf__max_depth": [None, 15],
           "clf__min_samples_leaf": [1, 2]}
gs_rf = GridSearchCV(pipe_rf, grid_rf, cv=5, scoring="f1", n_jobs=-1)
gs_rf.fit(X_train_t, y_train)
rf = gs_rf.best_estimator_
print("best RF params:", gs_rf.best_params_)
res_rf, proba_rf, pred_rf = evaluate(rf, X_test_t, y_test, "RandomForest")

# ---------------------------------------------------------------------------
# PRIORITY 2: XGBoost 
# ---------------------------------------------------------------------------
print("\n=== PRIORITY 2: XGBOOST ===")
spw = (len(y_train) - y_train.sum()) / y_train.sum()
pipe_xgb = Pipeline([
    ("smote", SMOTE(random_state=RANDOM_STATE)),
    ("clf", xgb.XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                              eval_metric="logloss", scale_pos_weight=spw)),
])
grid_xgb = {"clf__n_estimators": [200, 400],
            "clf__max_depth": [4, 6],
            "clf__learning_rate": [0.1, 0.2]}
gs_xgb = GridSearchCV(pipe_xgb, grid_xgb, cv=5, scoring="f1", n_jobs=-1)
gs_xgb.fit(X_train_t, y_train)
xgbm = gs_xgb.best_estimator_
print("best XGB params:", gs_xgb.best_params_)
res_xgb, proba_xgb, pred_xgb = evaluate(xgbm, X_test_t, y_test, "XGBoost")

# ---------------------------------------------------------------------------
# PRIORITY 3: GRAPH (predictive, OFFICIAL) - GraphSAGE end-to-end
# kNN cosine k=15 tren full 51-dim + SAGE 64-32 (h=64, drop=0.3, lr=0.01,
# wd=5e-5) + BCE pos_weight + early-stop val PR-AUC + threshold F1 tren VAL.
# Chi tiet: gnn_models.py (SAGE)
# ---------------------------------------------------------------------------
print("\n=== PRIORITY 3: GRAPH (GraphSAGE end-to-end, official) ===")
import torch
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph as kng_graph
from gnn_models import SAGE as SAGE_G, attach_node

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
X_all_t = np.vstack([X_train_t, X_test_t])
if hasattr(X_all_t, "toarray"):
    X_all_t = X_all_t.toarray()
X_all_t = X_all_t.astype(np.float32)
N_all = len(X_all_t)
n_tr = len(X_train_t)
y_all = np.concatenate([y_train, y_test])
idx_tr_all = np.arange(n_tr)
idx_te_all = np.arange(n_tr, N_all)
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
idx_tr2a, idx_vaa = next(sss.split(idx_tr_all, y_train))

K_SAGE = 15
A_g = kng_graph(X_all_t, n_neighbors=K_SAGE, mode="connectivity",
                metric="cosine", include_self=False)
A_g = A_g.maximum(A_g.T)
ei_g = torch.tensor(np.vstack(A_g.nonzero()), dtype=torch.long)
r_g, c_g = ei_g.numpy()
print(f"[graph] kNN-cosine k={K_SAGE} edges={A_g.nnz//2} "
      f"homophily={float((y_all[r_g]==y_all[c_g]).mean()):.3f}")

Xt_g = torch.tensor(X_all_t)
yt_g = torch.tensor(y_all, dtype=torch.float32)
sage_m = SAGE_G(X_all_t.shape[1], h=64, drop=0.3)
opt_g = torch.optim.Adam(sage_m.parameters(), lr=0.01, weight_decay=5e-5)
pos_g = float((len(idx_tr2a) - y_all[idx_tr2a].sum()) / max(y_all[idx_tr2a].sum(), 1))
crit_g = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_g]))
best_pr, bs_g, bad_g = -1, None, 0
for ep_g in range(200):
    sage_m.train(); opt_g.zero_grad()
    out_g = sage_m(Xt_g, ei_g)
    loss_g = crit_g(out_g[idx_tr2a], yt_g[idx_tr2a]); loss_g.backward(); opt_g.step()
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
sage_m.load_state_dict(bs_g); sage_m.eval()
print(f"[graph] SAGE val_PR-AUC={best_pr:.4f} ep={ep_g+1}")
with torch.no_grad():
    lg_all = sage_m(Xt_g, ei_g).numpy()
pv_va = torch.sigmoid(torch.tensor(lg_all[idx_vaa])).numpy()
pt_te = torch.sigmoid(torch.tensor(lg_all[idx_te_all])).numpy()
# threshold F1 toi uu tren VAL
best_t_g, best_f1v = 0.5, -1
for t_g in np.arange(0.1, 0.91, 0.05):
    f1v = f1_score(y_all[idx_vaa], (pv_va >= t_g).astype(int), zero_division=0)
    if f1v > best_f1v:
        best_f1v, best_t_g = f1v, t_g
pred_g = (pt_te >= best_t_g).astype(int)
proba_graph = pt_te
res_graph = {
    "model": "Graph",
    "roc_auc": float(roc_auc_score(y_test, pt_te)),
    "pr_auc": float(average_precision_score(y_test, pt_te)),
    "precision": float(precision_score(y_test, pred_g, zero_division=0)),
    "recall": float(recall_score(y_test, pred_g, zero_division=0)),
    "f1": float(f1_score(y_test, pred_g, zero_division=0)),
    "val_f1": float(best_f1v),
    "best_threshold": float(best_t_g),
    "accuracy_default": float((((pt_te >= 0.5).astype(int)) == y_test).mean()),
    "confusion_matrix": confusion_matrix(y_test, pred_g).tolist(),
    "n_test": int(len(y_test)),
    "n_pred_positive": int(pred_g.sum()),
}
print(f"[Graph] ROC-AUC={res_graph['roc_auc']:.4f} PR-AUC={res_graph['pr_auc']:.4f} "
      f"F1={res_graph['f1']:.4f} (thr_val={best_t_g:.2f}) ACC@0.5={res_graph['accuracy_default']:.4f} "
      f"P={res_graph['precision']:.3f} R={res_graph['recall']:.3f}")
# train-only graph cho app inference (SAGE inductive: gan node moi vao graph train)
A_tr = kng_graph(np.array(X_train_t.toarray() if hasattr(X_train_t, "toarray") else X_train_t, dtype=np.float32),
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
}
print(f"[graph] bundle: train_nodes={len(X_train_t)} edges={ei_tr.shape[1]//2}")

# best by metric (reference) + main model by policy (Graph SAGE-thuan k=15)
all_res = {"RandomForest": res_rf, "XGBoost": res_xgb, "Graph": res_graph}
best_metric = max(all_res, key=lambda k: all_res[k]["roc_auc"])
best_name = "Graph"  # policy: SAGE thuan lam mo hinh chinh, RF/XGB la baseline
print(f"\n[summary] best by ROC-AUC (3-way) = {best_metric}; main model by policy = {best_name}")

# ---------------------------------------------------------------------------
# Persist artifacts for the Streamlit app
# ---------------------------------------------------------------------------
ART_DIR = Path("models/ibm")
ART_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(pre, ART_DIR / "preprocessor.joblib")
joblib.dump(rf, ART_DIR / "random_forest.joblib")
joblib.dump(xgbm, ART_DIR / "xgboost.joblib")
joblib.dump(graph_bundle, ART_DIR / "graph.joblib")
meta = {
    "features": CAT + NUM, "cat": CAT, "num": NUM,
    "feat_names": feat_names.tolist(), "best_model": best_name,
    "thr_rf": res_rf["best_threshold"], "thr_xgb": res_xgb["best_threshold"],
    "thr_graph": res_graph["best_threshold"],
    "metrics": {"RandomForest": res_rf, "XGBoost": res_xgb, "Graph": res_graph},
    "graph_params": {"model": "GraphSAGE-end2end", "graph": "knn-cosine",
                     "k": K_SAGE, "sage": "64-32", "lr": 0.01, "wd": 5e-5,
                     "drop": 0.3, "early_stop": "val-PR-AUC"},
}
joblib.dump(meta, ART_DIR / "meta.joblib")
print(f"[artifacts] saved models + preprocessor -> {ART_DIR}")
# Graph predictive node/edge stats
print(f"[graph] nodes: all={N_all} (train={n_tr} test={len(idx_te_all)}) "
      f"edges={A_g.nnz//2} train_edges={A_tr.nnz//2} "
      f"feat_dim={X_all_t.shape[1]} emb_dim=32")

# ---------------------------------------------------------------------------
# SHAP for both models
# ---------------------------------------------------------------------------
print("\n=== SHAP (global feature importance) ===")
explainer_rf = shap.TreeExplainer(rf.named_steps["clf"])
sv_rf = explainer_rf.shap_values(X_test_t)
if isinstance(sv_rf, list):
    sv_rf = sv_rf[1]
shap.summary_plot(sv_rf, X_test_t, feature_names=feat_names, show=False, max_display=15)
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_rf.png", dpi=120)
plt.close()

explainer_xgb = shap.TreeExplainer(xgbm.named_steps["clf"])
sv_xgb = explainer_xgb.shap_values(X_test_t)
if isinstance(sv_xgb, list):
    sv_xgb = sv_xgb[1]
shap.summary_plot(sv_xgb, X_test_t, feature_names=feat_names, show=False, max_display=15)
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_xgboost.png", dpi=120)
plt.close()
print("saved shap_rf.png, shap_xgboost.png")

# ---------------------------------------------------------------------------
# PRIORITY 4a: Model comparison figures (3-way)
# ---------------------------------------------------------------------------
print("\n=== PRIORITY 4a: MODEL COMPARISON (3-way) ===")
# ROC curves 3-way
plt.figure(figsize=(6, 5))
for proba, name, c in [(proba_rf, "RandomForest", "#1f77b4"), (proba_xgb, "XGBoost", "#d62728"), (proba_graph, "Graph", "#2ca02c")]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, proba):.3f})", color=c, lw=2)
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve: Random Forest vs XGBoost vs Graph"); plt.legend(loc="lower right")
plt.tight_layout(); plt.savefig(OUT_DIR / "roc_curves.png", dpi=120); plt.close()

# metrics bar 3-way
metrics = ["roc_auc", "pr_auc", "f1", "precision", "recall"]
x = np.arange(len(metrics)); w = 0.25
plt.figure(figsize=(9, 5))
plt.bar(x - w, [res_rf[m] for m in metrics], w, label="RandomForest", color="#1f77b4")
plt.bar(x, [res_xgb[m] for m in metrics], w, label="XGBoost", color="#d62728")
plt.bar(x + w, [res_graph[m] for m in metrics], w, label="Graph", color="#2ca02c")
plt.xticks(x, metrics); plt.ylim(0, 1.05); plt.ylabel("Score"); plt.legend()
plt.title("Model Performance Comparison (3-way)"); plt.tight_layout()
plt.savefig(OUT_DIR / "metrics_bar.png", dpi=120); plt.close()
print("saved roc_curves.png (3-way), metrics_bar.png (3-way)")

# Graph similarity visualization (kNN graph on scaled features, sample 150)
try:
    np.random.seed(RANDOM_STATE)
    idx = np.random.choice(len(X_train_t), size=min(150, len(X_train_t)), replace=False)
    Xs = X_train_t[idx]
    ys = y_train[idx]
    # Convert to dense if sparse
    if hasattr(Xs, "toarray"):
        Xs_dense = Xs.toarray()
    else:
        Xs_dense = np.array(Xs)
    # Build kNN graph with k=10 for visualization (SAGE official uses k=15)
    best_k = K_SAGE
    A = kneighbors_graph(Xs_dense, n_neighbors=min(best_k, 10), mode='connectivity', include_self=False)
    G_vis = nx.from_scipy_sparse_array(A)
    # Use spring layout
    pos = nx.spring_layout(G_vis, seed=RANDOM_STATE)
    plt.figure(figsize=(7, 7))
    colors_vis = ["#d62728" if y==1 else "#1f77b4" for y in ys]
    nx.draw_networkx_nodes(G_vis, pos, node_color=colors_vis, node_size=60, alpha=0.9, linewidths=0.5, edgecolors="white")
    nx.draw_networkx_edges(G_vis, pos, edge_color="#7f8c8d", alpha=0.25, width=0.7)
    # counts for legend
    n_attr = int((ys==1).sum()); n_stay = int((ys==0).sum())
    plt.title(f"Graph Predictive - Employee Similarity Graph (kNN, sample 150)\nNodes={len(ys)} (Attrition={n_attr} Stay={n_stay})  Edges={A.nnz//2} (undirected)  k={min(best_k,10)}", fontsize=10, fontweight="bold")
    plt.axis("off"); plt.tight_layout(); plt.savefig(OUT_DIR / "graph_similarity.png", dpi=130); plt.close()
    print(f"saved graph_similarity.png  nodes={len(ys)} edges={A.nnz} k={min(best_k,10)}")
    graph_vis_stats = {"vis_nodes": int(len(ys)), "vis_edges_directed": int(A.nnz), "vis_edges_undirected": int(A.nnz//2), "vis_k": int(min(best_k,10)), "vis_attrition": int(n_attr)}
except Exception as e:
    print(f"[graph vis] warning: {e}")
    graph_vis_stats = {"error": str(e)}

# ---------------------------------------------------------------------------
# PRIORITY 4b: Causal DAG (DoWhy) OverTime -> Attrition (explanatory, not predictive)
# ---------------------------------------------------------------------------
causal_df = df.copy()
causal_df["OverTime"] = (causal_df["OverTime"] == "Yes").astype(int)
for col in ["BusinessTravel", "Department", "JobRole", "MaritalStatus"]:
    causal_df[col] = causal_df[col].astype("category").cat.codes
confounders = ["Age", "BusinessTravel", "Department", "Education", "JobLevel",
               "JobRole", "MaritalStatus", "MonthlyIncome", "TotalWorkingYears",
               "YearsAtCompany", "JobSatisfaction", "EnvironmentSatisfaction",
               "WorkLifeBalance", "StockOptionLevel", "TrainingTimesLastYear",
               "NumCompaniesWorked", "DistanceFromHome"]

# build DOT manually
dot = "digraph {\n"
for c in confounders:
    dot += f'  "{c}" -> "OverTime";\n'
    dot += f'  "{c}" -> "Attrition";\n'
dot += '  "OverTime" -> "Attrition";\n}\n'

model = CausalModel(
    data=causal_df[["OverTime", "Attrition"] + confounders],
    treatment="OverTime", outcome="Attrition", graph=dot,
)
identified = model.identify_effect()
estimate = model.estimate_effect(identified,
                                 method_name="backdoor.linear_regression")
ref_placebo = model.refute_estimate(identified, estimate,
                                     method_name="placebo_treatment_refuter")
ref_rcc = model.refute_estimate(identified, estimate,
                                method_name="random_common_cause")

ate_val = float(estimate.value)
print(f"[causal] ATE(OverTime -> Attrition) = {ate_val:.4f}")
print(f"[causal] placebo ATE ~ {ref_placebo.new_effect:.4f}  (expect ~0)")
print(f"[causal] random-common-cause ATE ~ {ref_rcc.new_effect:.4f}")

# DAG visualization (networkx)
G = nx.DiGraph()
for c in confounders:
    G.add_node(c, layer=0)
G.add_node("OverTime", layer=1)
G.add_node("Attrition", layer=2)
for c in confounders:
    G.add_edge(c, "OverTime"); G.add_edge(c, "Attrition")
G.add_edge("OverTime", "Attrition")

plt.figure(figsize=(11, 7))
pos = {}
for i, c in enumerate(confounders):
    pos[c] = (0.05 + (i % 6) * 0.16, 0.92 - (i // 6) * 0.32)
pos["OverTime"] = (0.70, 0.55)
pos["Attrition"] = (0.95, 0.55)
colors = ["#2b5c8f" if n in confounders else ("#d9534f" if n == "OverTime" else "#27ae60")
          for n in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=1900, alpha=0.9)
nx.draw_networkx_labels(G, pos, font_size=7, font_color="white", font_weight="bold")
nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>", arrowsize=12,
                       edge_color="#7f8c8d", alpha=0.4, connectionstyle="arc3,rad=0.05")
plt.title("Causal DAG: OverTime (Treatment) -> Attrition (Outcome)\n"
          f"ATE = {ate_val:+.4f}  |  Confounders = {len(confounders)}",
          fontsize=12, fontweight="bold")
plt.axis("off"); plt.tight_layout()
plt.savefig(OUT_DIR / "attrition_dag.png", dpi=130); plt.close()
print("saved attrition_dag.png")

# ---------------------------------------------------------------------------
# Save report JSON
# ---------------------------------------------------------------------------
report = {
    "dataset": "IBM HR Analytics Employee Attrition (WA_Fn-UseC_-HR-Employee-Attrition.csv)",
    "n_rows": int(len(df)),
    "positive_rate": float(y.mean()),
    "target": "Attrition (Yes=1)",
    "models": {"RandomForest": res_rf, "XGBoost": res_xgb, "Graph": res_graph},
    "best_model": best_name,
    "graph_predictive": {
        "kernel": "knn-cosine",
        "params": {"model": "GraphSAGE-end2end", "k": K_SAGE,
                   "sage": "64-32", "lr": 0.01, "wd": 5e-5, "drop": 0.3},
        "n_nodes_train_original": int(len(y_train)),
        "n_nodes_all": int(N_all),
        "vis_stats": graph_vis_stats if 'graph_vis_stats' in locals() else {},
        "feat_dim": int(X_train_t.shape[1]),
        "graph_type": "GraphSAGE end-to-end (kNN cosine + SAGE 64-32, no SVM)",
    },
    "causal": {
        "treatment": "OverTime",
        "outcome": "Attrition",
        "ate": ate_val,
        "placebo_ate": float(ref_placebo.new_effect),
        "random_common_cause_ate": float(ref_rcc.new_effect),
        "n_confounders": len(confounders),
        "confounders": confounders,
    },
}
with open(OUT_DIR / "ibm_results.json", "w") as f:
    json.dump(report, f, indent=2)
print("\n[done] saved reports/ibm/ibm_results.json")
