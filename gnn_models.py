"""gnn_models.py - Pipeline A (Al Akasheh style) dung chung cho train_ibm.py va app_ibm.py.
Graph: weighted kNN (k=15, cosine) tren 15 SHAP-top features.
Model: GCN 2-layer (ReLU, BCE pos_weight, Adam, early-stop) -> embedding 32-dim
       -> concat[51 features goc + 32 emb] -> Linear SVM (C=0.1, Platt scaling cho proba).
Inference 1 mau moi (transductive-safe): gan node moi vao graph train (top-15 weighted),
forward GCN 1 pass, lay embedding node moi -> SVM predict_proba.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import to_undirected
from sklearn.metrics.pairwise import cosine_similarity

SEED = 42
K = 15
C_SVM = 0.1
# Union top-10 SHAP RF + top-10 SHAP XGB (reports/ibm/analyze_stats.json)
SHAP_FEATS = [
    "cat__OverTime_Yes", "cat__OverTime_No", "num__JobLevel",
    "cat__MaritalStatus_Single", "num__StockOptionLevel",
    "cat__JobRole_Laboratory Technician", "num__Age",
    "num__YearsWithCurrManager", "num__YearsAtCompany",
    "cat__BusinessTravel_Travel_Frequently",
    "num__NumCompaniesWorked", "num__MonthlyIncome",
    "num__DistanceFromHome", "num__EnvironmentSatisfaction",
    "num__JobSatisfaction",
]


class GCN(torch.nn.Module):
    def __init__(self, d, h=64, drop=0.5):
        super().__init__()
        self.c1 = GCNConv(d, h)
        self.c2 = GCNConv(h, 32)
        self.lin = torch.nn.Linear(32, 1)
        self.drop = drop

    def forward(self, x, ei, ew=None, ret_emb=False):
        x = F.relu(self.c1(x, ei, ew))
        x = F.dropout(x, p=self.drop, training=self.training)
        emb = self.c2(x, ei, ew)
        out = self.lin(F.relu(emb)).squeeze(-1)
        return (out, emb) if ret_emb else out


class SAGE(torch.nn.Module):
    """GraphSAGE end-to-end (official graph model): 51 -> 64 -> 32 -> logit."""

    def __init__(self, d, h=64, drop=0.3):
        super().__init__()
        self.c1 = SAGEConv(d, h)
        self.c2 = SAGEConv(h, 32)
        self.lin = torch.nn.Linear(32, 1)
        self.drop = drop

    def forward(self, x, ei):
        x = F.relu(self.c1(x, ei))
        x = F.dropout(x, p=self.drop, training=self.training)
        return self.lin(F.relu(self.c2(x, ei))).squeeze(-1)


def feat_index(feat_names):
    idx = []
    for f in SHAP_FEATS:
        j = np.where(np.array(feat_names) == f)[0]
        if len(j):
            idx.append(int(j[0]))
    return np.array(idx, dtype=int)


def build_weighted_graph(X, sidx, k=K):
    """X: (N,51) float32. Tra ve (edge_index LongTensor, edge_weight FloatTensor)."""
    Xs = X[:, sidx]
    S = cosine_similarity(Xs)
    np.fill_diagonal(S, 0.0)
    rows, cols, ws = [], [], []
    for i in range(len(Xs)):
        js = np.argsort(S[i])[::-1][:k]
        for j in js:
            rows.append(i); cols.append(int(j)); ws.append(float(max(S[i, j], 0.0)))
    ei = torch.tensor([rows, cols], dtype=torch.long)
    ew = torch.tensor(ws, dtype=torch.float32)
    return to_undirected(ei, ew, reduce="max")


def attach_node(X_train, ei, ew, x_new, sidx, k=K):
    """Gan 1 mau moi vao graph train. Tra ve (X_aug, ei_aug, ew_aug, new_idx)."""
    sims = cosine_similarity(x_new[sidx].reshape(1, -1), X_train[:, sidx])[0]
    sims = np.maximum(sims, 0.0)
    js = np.argsort(sims)[::-1][:k]
    n = X_train.shape[0]
    X_aug = np.vstack([X_train, x_new.reshape(1, -1)])
    er = ei[0].tolist() + [n] * k + js.tolist()
    ec = ei[1].tolist() + js.tolist() + [n] * k
    w = ew.tolist() + [float(sims[j]) for j in js] * 2
    ei_aug = torch.tensor([er, ec], dtype=torch.long)
    ew_aug = torch.tensor(w, dtype=torch.float32)
    return X_aug, ei_aug, ew_aug, n
