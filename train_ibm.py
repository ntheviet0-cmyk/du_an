"""
train_ibm.py - Employee Attrition Prediction & Causal Graph on the IBM HR Analytics
dataset (WA_Fn-UseC_-HR-Employee-Attrition.csv).

Report deliverables:
  PRIORITY 1 : Random Forest  (SMOTE + 5-fold CV tuning, threshold optimization)
  PRIORITY 2 : XGBoost        (SMOTE + scale_pos_weight + 5-fold CV, threshold optimization)
  PRIORITY 3 : Graph (predictive) - LabelSpreading (knn/rbf, SMOTE + 5-fold CV) as 3rd model
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
from sklearn.semi_supervised import LabelSpreading
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
# PRIORITY 2: XGBoost (SMOTE + scale_pos_weight - per user request)
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
# PRIORITY 3: GRAPH (predictive) - LabelSpreading with SMOTE + 5-fold CV
# ---------------------------------------------------------------------------
print("\n=== PRIORITY 3: GRAPH (LabelSpreading) ===")
pipe_graph = Pipeline([
    ("smote", SMOTE(random_state=RANDOM_STATE)),
    ("clf", LabelSpreading(kernel='knn', max_iter=30, n_jobs=None)),
])
# Conditional grid for knn vs rbf kernels
grid_graph = [
    {"clf__kernel": ["knn"], "clf__n_neighbors": [10, 15, 20], "clf__alpha": [0.1, 0.2, 0.5]},
    {"clf__kernel": ["rbf"], "clf__gamma": [10, 20], "clf__alpha": [0.1, 0.2, 0.5]},
]
gs_graph = GridSearchCV(pipe_graph, grid_graph, cv=5, scoring="f1")
gs_graph.fit(X_train_t, y_train)
graph_model = gs_graph.best_estimator_
print("best Graph params:", gs_graph.best_params_)
res_graph, proba_graph, pred_graph = evaluate(graph_model, X_test_t, y_test, "Graph")

# best among 3 by ROC-AUC (per user decision 3)
all_res = {"RandomForest": res_rf, "XGBoost": res_xgb, "Graph": res_graph}
best_name = max(all_res, key=lambda k: all_res[k]["roc_auc"])
best_model = {"RandomForest": rf, "XGBoost": xgbm, "Graph": graph_model}[best_name]
print(f"\n[summary] best model by ROC-AUC (3-way) = {best_name}")

# ---------------------------------------------------------------------------
# Persist artifacts for the Streamlit app
# ---------------------------------------------------------------------------
ART_DIR = Path("models/ibm")
ART_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(pre, ART_DIR / "preprocessor.joblib")
joblib.dump(rf, ART_DIR / "random_forest.joblib")
joblib.dump(xgbm, ART_DIR / "xgboost.joblib")
joblib.dump(graph_model, ART_DIR / "graph.joblib")
meta = {
    "features": CAT + NUM, "cat": CAT, "num": NUM,
    "feat_names": feat_names.tolist(), "best_model": best_name,
    "thr_rf": res_rf["best_threshold"], "thr_xgb": res_xgb["best_threshold"],
    "thr_graph": res_graph["best_threshold"],
    "metrics": {"RandomForest": res_rf, "XGBoost": res_xgb, "Graph": res_graph},
    "graph_params": gs_graph.best_params_,
}
joblib.dump(meta, ART_DIR / "meta.joblib")
print(f"[artifacts] saved models + preprocessor -> {ART_DIR}")
# Graph predictive node/edge stats (after SMOTE)
try:
    clf_g = graph_model.named_steps["clf"]
    # After SMOTE, X_train_resampled size
    from collections import Counter
    # Retrieve resampled y size via SMOTE fit (approx)
    sm = SMOTE(random_state=RANDOM_STATE)
    _, y_res = sm.fit_resample(X_train_t, y_train)
    n_nodes_train = len(y_res)
    n_pos_train = int((y_res==1).sum())
    print(f"[graph] predictive nodes: train_original={len(y_train)} pos={int(y_train.sum())} -> after SMOTE n_nodes={n_nodes_train} pos={n_pos_train}")
    if hasattr(clf_g, "affinity_matrix_") and clf_g.affinity_matrix_ is not None:
        am = clf_g.affinity_matrix_
        n_nodes = am.shape[0]
        # affinity_matrix_ may be sparse or dense
        try:
            n_edges = int(am.nnz) if hasattr(am, "nnz") else int((am>1e-9).sum())
        except Exception:
            n_edges = int((am>0).sum()) if hasattr(am, "sum") else -1
        print(f"[graph] affinity matrix: n_nodes={n_nodes} n_edges(nnz)={n_edges}")
    if hasattr(clf_g, "graph_matrix") and clf_g.graph_matrix is not None:
        gm = clf_g.graph_matrix
        try:
            n_edges_g = int(gm.nnz) if hasattr(gm, "nnz") else int((gm>1e-9).sum())
            print(f"[graph] graph_matrix n_edges={n_edges_g}")
        except Exception:
            pass
    print(f"[graph] kernel={clf_g.kernel} n_neighbors={getattr(clf_g,'n_neighbors', 'N/A')} alpha={clf_g.alpha} gamma={getattr(clf_g,'gamma', 'N/A')}")
except Exception as e:
    print(f"[graph] stats warning: {e}")

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
    # Build kNN graph with k=10 for visualization (use best n_neighbors if knn)
    best_k = gs_graph.best_params_.get("n_neighbors", 10)
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
        "kernel": gs_graph.best_params_.get("clf__kernel"),
        "params": gs_graph.best_params_,
        "n_nodes_train_original": int(len(y_train)),
        "n_nodes_train_after_smote": int(len(y_res)) if 'y_res' in locals() else None,
        "vis_stats": graph_vis_stats if 'graph_vis_stats' in locals() else {},
        "feat_dim": int(X_train_t.shape[1]),
        "graph_type": "LabelSpreading (employee similarity kNN graph)",
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
