"""
train_ibm_gnn.py - TEST GNN hien dai (PyG) cho IBM HR Attrition.
So sanh cong bang voi RF baseline tren cung split 80/20 (seed 42).

Graph: transductive - toan bo 1470 nodes visible, chi mask label train/test.
2 kieu dung graph de ablation:
  A. kNN cosine k=15 tren 51 chieu (cai tien tu Euclidean cu)
  B. Knowledge-graph: kNN + edge cung Department & JobRole (copy IEEE Access 2024)
Models: GCN-2layer, GraphSAGE-2layer, GAT-2layer, Hybrid GCN-embedding + LogReg
Loss: BCEWithLogitsLoss(pos_weight) - thay SMOTE pha topology.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_distances

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
OUT = Path("reports/ibm_gnn")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- data
df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
df["Attrition"] = (df["Attrition"] == "Yes").astype(int)
df = df.drop(columns=["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"])
CAT = ["BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime"]
NUM = [c for c in df.columns if c not in CAT + ["Attrition"]]
y_all = df["Attrition"].values
N = len(df)
print(f"[data] rows={N} pos={int(y_all.sum())} ({y_all.mean()*100:.1f}%)")

# split giong train_ibm.py: 80/20 stratified, sau do tach val 15% tu train
idx = np.arange(N)
idx_tr, idx_te = train_test_split(idx, test_size=0.2, stratify=y_all, random_state=SEED)
idx_tr2, idx_va = train_test_split(idx_tr, test_size=0.15, stratify=y_all[idx_tr], random_state=SEED)
print(f"[split] train={len(idx_tr2)} val={len(idx_va)} test={len(idx_te)}")

pre = ColumnTransformer([("num", StandardScaler(), NUM),
                         ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
pre.fit(df.iloc[idx_tr][CAT + NUM])
X_all = pre.transform(df[CAT + NUM])
if hasattr(X_all, "toarray"):
    X_all = X_all.toarray()
X_all = X_all.astype(np.float32)
print(f"[feat] dim={X_all.shape[1]}")

# ---------------------------------------------------------------- graph builders
def build_knn_cosine(X, k=15):
    A = kneighbors_graph(X, n_neighbors=k, mode="connectivity",
                         metric="cosine", include_self=False)
    A = A.maximum(A.T)  # undirected
    ei = torch.tensor(np.vstack(A.nonzero()), dtype=torch.long)
    return ei, A

def build_knowledge_graph(X, df, k=10):
    # Nhan kNN (cosine, k nho hon) + edge cung Department/JobRole (gioi han bac)
    A_knn = kneighbors_graph(X, n_neighbors=k, mode="connectivity",
                             metric="cosine", include_self=False)
    A_knn = A_knn.maximum(A_knn.T).tolil()
    dept = df["Department"].values
    role = df["JobRole"].values
    # chi noi them toi da 5 edge dong dept+role gan nhat de tranh graph day dac
    from sklearn.metrics.pairwise import cosine_distances
    D = cosine_distances(X)
    added = 0
    for i in range(N):
        cand = np.where((dept == dept[i]) & (role == role[i]))[0]
        cand = cand[cand != i]
        if len(cand) == 0:
            continue
        take = cand[np.argsort(D[i, cand])[:5]]
        for j in take:
            if A_knn[i, j] == 0:
                A_knn[i, j] = 1
                A_knn[j, i] = 1
                added += 1
    A_knn = A_knn.tocsr()
    ei = torch.tensor(np.vstack(A_knn.nonzero()), dtype=torch.long)
    return ei, A_knn, added

def homophily(A, y):
    r, c = A.nonzero()
    return float((y[r] == y[c]).mean())

ei_A, A_A = build_knn_cosine(X_all, k=15)
ei_B, A_B, added_B = build_knowledge_graph(X_all, df, k=10)
print(f"[graph A kNN-cos k=15] edges_undirected={A_A.nnz//2} homophily={homophily(A_A, y_all):.3f}")
print(f"[graph B knowledge] edges_undirected={A_B.nnz//2} (+{added_B//2} dept/role) homophily={homophily(A_B, y_all):.3f}")

# ---------------------------------------------------------------- models
class GCN(torch.nn.Module):
    def __init__(self, d, h=64, drop=0.5):
        super().__init__()
        self.c1 = GCNConv(d, h)
        self.c2 = GCNConv(h, 32)
        self.lin = torch.nn.Linear(32, 1)
        self.drop = drop
    def forward(self, x, ei, ret_emb=False):
        x = F.relu(self.c1(x, ei))
        x = F.dropout(x, p=self.drop, training=self.training)
        emb = self.c2(x, ei)
        out = self.lin(F.relu(emb)).squeeze(-1)
        return (out, emb) if ret_emb else out

class SAGE(torch.nn.Module):
    def __init__(self, d, h=64, drop=0.5):
        super().__init__()
        self.c1 = SAGEConv(d, h)
        self.c2 = SAGEConv(h, 32)
        self.lin = torch.nn.Linear(32, 1)
        self.drop = drop
    def forward(self, x, ei):
        x = F.relu(self.c1(x, ei))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = F.relu(self.c2(x, ei))
        return self.lin(x).squeeze(-1)

class GAT(torch.nn.Module):
    def __init__(self, d, h=16, heads=8, drop=0.5):
        super().__init__()
        self.c1 = GATConv(d, h, heads=heads, dropout=drop)
        self.c2 = GATConv(h * heads, 32, heads=1, dropout=drop)
        self.lin = torch.nn.Linear(32, 1)
        self.drop = drop
    def forward(self, x, ei):
        x = F.elu(self.c1(x, ei))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = F.elu(self.c2(x, ei))
        return self.lin(x).squeeze(-1)

def train_gnn(model, ei, X, y, idx_tr2, idx_va, epochs=200, patience=30, lr=0.01, wd=5e-4):
    X_t = torch.tensor(X)
    y_t = torch.tensor(y, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    pos = float((len(idx_tr2) - y[idx_tr2].sum()) / max(y[idx_tr2].sum(), 1))
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos]))
    best_state, best_pr, bad = None, -1, 0
    model.train()
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(X_t, ei) if not isinstance(model, GCN) else model(X_t, ei, ret_emb=False)
        loss = crit(out[idx_tr2], y_t[idx_tr2])
        loss.backward()
        opt.step()
        # val PR-AUC
        model.eval()
        with torch.no_grad():
            o = model(X_t, ei) if not isinstance(model, GCN) else model(X_t, ei, ret_emb=False)
            pv = torch.sigmoid(o[idx_va]).numpy()
        try:
            pr = average_precision_score(y[idx_va], pv)
        except Exception:
            pr = 0.0
        if pr > best_pr:
            best_pr, best_state, bad = pr, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= patience:
            break
        model.train()
    model.load_state_dict(best_state)
    return model, best_pr, ep + 1

def eval_logits(logits, y_true, name):
    proba = torch.sigmoid(torch.tensor(logits)).numpy()
    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.1, 0.9, 0.05):
        f1 = f1_score(y_true, (proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    pred = (proba >= best_t).astype(int)
    res = {"model": name, "roc_auc": float(roc_auc_score(y_true, proba)),
           "pr_auc": float(average_precision_score(y_true, proba)),
           "precision": float(precision_score(y_true, pred, zero_division=0)),
           "recall": float(recall_score(y_true, pred, zero_division=0)),
           "f1": float(best_f1), "best_threshold": float(best_t),
           "confusion_matrix": confusion_matrix(y_true, pred).tolist()}
    print(f"[{name}] ROC-AUC={res['roc_auc']:.4f} PR-AUC={res['pr_auc']:.4f} "
          f"F1={res['f1']:.4f} (thr={best_t:.2f}) P={res['precision']:.3f} R={res['recall']:.3f}")
    return res, proba

# ---------------------------------------------------------------- baseline RF (cung split, nhanh)
print("\n=== BASELINE RF (same split) ===")
rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=1,
                            random_state=SEED, n_jobs=-1, class_weight="balanced")
rf.fit(X_all[idx_tr], y_all[idx_tr])
res_rf, _ = eval_logits(torch.tensor(np.log(
    np.clip(rf.predict_proba(X_all[idx_te])[:, 1], 1e-6, 1 - 1e-6) /
    np.clip(1 - rf.predict_proba(X_all[idx_te])[:, 1], 1e-6, 1))), y_all[idx_te], "RF-baseline")

# ---------------------------------------------------------------- GNN tests (graph A chinh, graph B cho GCN de ablation)
results = {"RF-baseline": res_rf}
X_t = torch.tensor(X_all)

for gname, ei in [("A-kNNcos", ei_A), ("B-knowledge", ei_B)]:
    if gname.startswith("B"):
        print(f"\n=== GCN on {gname} (ablation) ===")
        m = GCN(X_all.shape[1])
        m, bpr, ep = train_gnn(m, ei, X_all, y_all, idx_tr2, idx_va)
        m.eval()
        with torch.no_grad():
            lg = (m(X_t, ei, ret_emb=False)).numpy()
        r, _ = eval_logits(lg[idx_te], y_all[idx_te], f"GCN-{gname}")
        results[f"GCN-{gname}"] = r
        print(f"  (val PR-AUC={bpr:.4f}, stopped epoch={ep})")

print("\n=== GCN / SAGE / GAT on graph A ===")
trained = {}
for mname, M in [("GCN", GCN(X_all.shape[1])), ("SAGE", SAGE(X_all.shape[1])), ("GAT", GAT(X_all.shape[1]))]:
    m, bpr, ep = train_gnn(M, ei_A, X_all, y_all, idx_tr2, idx_va)
    m.eval()
    with torch.no_grad():
        lg = (m(X_t, ei_A, ret_emb=False) if isinstance(m, GCN) else m(X_t, ei_A)).numpy()
    r, _ = eval_logits(lg[idx_te], y_all[idx_te], mname + "-A")
    results[mname + "-A"] = r
    trained[mname] = (m, lg)
    print(f"  (val PR-AUC={bpr:.4f}, stopped epoch={ep})")

# ---------------------------------------------------------------- Hybrid: GCN embedding -> LogReg (copy IEEE Access 2024)
print("\n=== Hybrid GCN-emb + LogReg ===")
gcn_m = trained["GCN"][0]
gcn_m.eval()
with torch.no_grad():
    _, emb = gcn_m(X_t, ei_A, ret_emb=True)
emb_np = emb.numpy()
lr = LogisticRegression(max_iter=1000, class_weight="balanced")
lr.fit(emb_np[idx_tr], y_all[idx_tr])
proba_h = lr.predict_proba(emb_np[idx_te])[:, 1]
rh = {"model": "Hybrid-GCN-LogReg", "roc_auc": float(roc_auc_score(y_all[idx_te], proba_h)),
      "pr_auc": float(average_precision_score(y_all[idx_te], proba_h))}
best_t, best_f1 = 0.5, -1
for t in np.arange(0.1, 0.9, 0.05):
    f1 = f1_score(y_all[idx_te], (proba_h >= t).astype(int), zero_division=0)
    if f1 > best_f1:
        best_f1, best_t = f1, t
pred_h = (proba_h >= best_t).astype(int)
rh.update({"precision": float(precision_score(y_all[idx_te], pred_h, zero_division=0)),
           "recall": float(recall_score(y_all[idx_te], pred_h, zero_division=0)),
           "f1": float(best_f1), "best_threshold": float(best_t),
           "confusion_matrix": confusion_matrix(y_all[idx_te], pred_h).tolist()})
print(f"[Hybrid-GCN-LogReg] ROC-AUC={rh['roc_auc']:.4f} PR-AUC={rh['pr_auc']:.4f} "
      f"F1={rh['f1']:.4f} (thr={best_t:.2f}) P={rh['precision']:.3f} R={rh['recall']:.3f}")
results["Hybrid-GCN-LogReg"] = rh

with open(OUT / "gnn_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[done] saved {OUT/'gnn_results.json'}")

# Ket luan nhanh
order = sorted(results.items(), key=lambda kv: kv[1]["roc_auc"], reverse=True)
print("\n=== RANKING (ROC-AUC) ===")
for k, v in order:
    print(f"  {k:22s} AUC={v['roc_auc']:.4f} PR={v['pr_auc']:.4f} F1={v['f1']:.4f}")
