"""Train-only preprocessing/graphs and disjoint per-query inductive inference."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
CAT = ["BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime"]
EXCLUDED = ["Attrition", "EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]


def load_data():
    raw = pd.read_csv(DATA_PATH)
    y = raw.Attrition.eq("Yes").to_numpy(dtype=np.int64)
    numeric = [c for c in raw if c not in CAT + EXCLUDED]
    return raw, raw[CAT + numeric], y, numeric


def make_splits(y, seed, folds):
    # Keep the historical 294-case holdout outside all new tuning.
    development, historical = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = [(development[tr], development[va]) for tr, va in cv.split(development, y[development])]
    return development, historical, splits


def prepare_fold(frame, numeric, train_idx, val_idx):
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ])
    train = np.asarray(pre.fit_transform(frame.iloc[train_idx]), dtype=np.float32)
    val = np.asarray(pre.transform(frame.iloc[val_idx]), dtype=np.float32)
    return pre, train, val


def make_graph(train, k):
    if not 1 <= k < len(train):
        raise ValueError("k must be smaller than the number of training nodes")
    adjacency = kneighbors_graph(train, n_neighbors=k, metric="cosine", mode="connectivity", include_self=False)
    adjacency = adjacency.maximum(adjacency.T)
    return torch.tensor(np.vstack(adjacency.nonzero()), dtype=torch.long)


def query_neighbors(train, queries, k):
    search = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute").fit(train)
    return search.kneighbors(queries, return_distance=False)


@torch.no_grad()
def inductive_predict(model, train, edges, queries, k, batch_size=8, neighbors=None):
    """Each query has its own copy of the train graph, never other queries.

    Bidirectional edges match graph_shap_lib.py's app-style attachment. Disjoint
    batching changes only execution speed, not the graph of any prediction.
    """
    model.eval()
    js = query_neighbors(train, queries, k) if neighbors is None else neighbors
    n = len(train)
    block = n + 1
    train_tensor = torch.as_tensor(train, dtype=torch.float32)
    output = []
    for start in range(0, len(queries), batch_size):
        q = torch.as_tensor(queries[start:start + batch_size], dtype=torch.float32)
        count = len(q)
        x = torch.cat((train_tensor.unsqueeze(0).expand(count, -1, -1), q[:, None, :]), dim=1).reshape(-1, train.shape[1])
        neighbor = torch.as_tensor(js[start:start + count], dtype=torch.long)
        new_nodes = torch.full_like(neighbor, n)
        attach = torch.stack((torch.cat((new_nodes, neighbor), dim=1), torch.cat((neighbor, new_nodes), dim=1)), dim=1)
        base = edges.unsqueeze(0).expand(count, -1, -1)
        all_edges = torch.cat((base, attach), dim=2)
        offsets = (torch.arange(count) * block)[:, None, None]
        all_edges = (all_edges + offsets).permute(1, 0, 2).reshape(2, -1)
        query_pos = torch.arange(count) * block + n
        output.append(torch.sigmoid(model(x, all_edges)[query_pos]).numpy())
    return np.concatenate(output) if output else np.empty(0)
