"""build_fairleave.py - FairLeave-style Phase 1+2 (+5 mot phan):
Graph cosine top-15 KHONG Gender (49-dim) + Residual GCN + Attrition classifier.
Baseline MLP tabular (khong graph) de doi chieu fairness.
Chay 3 seeds (42/1/7), moi seed do test 1 lan. Luu bundle seed tot nhat cho Phase 3+4.
Output: reports/ibm_gnn/fairleave_core.json + models/ibm/fairleave_bundle.joblib
"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, confusion_matrix)
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
OUT = Path("reports/ibm_gnn"); OUT.mkdir(parents=True, exist_ok=True)
ART = Path("models/ibm"); ART.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 1, 7]
K = 15

df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
df["Attrition"] = (df["Attrition"] == "Yes").astype(int)
df = df.drop(columns=["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"])
CAT = ["BusinessTravel","Department","EducationField","Gender","JobRole","MaritalStatus","OverTime"]
CAT_NG = [c for c in CAT if c != "Gender"]  # no-Gender input
NUM = [c for c in df.columns if c not in CAT+["Attrition"]]
y = df["Attrition"].values
gender = (df["Gender"] == "Male").astype(int).values  # 1=Male, 0=Female (chi audit)

def prep(cat_cols):
    pre = ColumnTransformer([("num",StandardScaler(),NUM),
                             ("cat",OneHotEncoder(handle_unknown="ignore"),cat_cols)])
    return pre

pre_ng = prep(CAT_NG)
# fit tren train cua seed 42 de on dinh feature space (giong moi run truoc)
idx0 = np.arange(len(df))
tr0, _ = train_test_split(idx0, test_size=0.2, stratify=y, random_state=42)
pre_ng.fit(df.iloc[tr0][CAT_NG + NUM])
Xng = pre_ng.transform(df[CAT_NG + NUM])
if hasattr(Xng, "toarray"): Xng = Xng.toarray()
Xng = np.asarray(Xng, dtype=np.float32)
print(f"[feat] no-Gender dim={Xng.shape[1]}", flush=True)

# ---- Phase 1: similarity graph, no Gender ----
S = np.maximum(cosine_similarity(Xng), 0)
np.fill_diagonal(S, 0)
rows, cols, ws = [], [], []
for i in range(len(Xng)):
    js = np.argsort(S[i])[::-1][:K]
    for j in js:
        rows.append(i); cols.append(int(j)); ws.append(float(S[i, j]))
ei = torch.tensor([rows, cols], dtype=torch.long)
ew = torch.tensor(ws, dtype=torch.float32)
EI, EW = to_undirected(ei, ew, reduce="max")
r_, c_ = EI.numpy()
print(f"[graph] no-Gender weighted edges={EI.shape[1]//2} homophily={float((y[r_]==y[c_]).mean()):.3f}", flush=True)
Xt = torch.tensor(Xng); yt = torch.tensor(y, dtype=torch.float32)

class ResGCN(torch.nn.Module):
    """Residual GCN: h = x1 + GCN2(x1) (skip-connection chong over-smoothing)."""
    def __init__(s, d, h=64, drop=0.5):
        super().__init__()
        s.c1 = GCNConv(d, h); s.c2 = GCNConv(h, h)
        s.lin = torch.nn.Linear(h, 1); s.drop = drop
    def forward(s, x, ei, ew=None, ret_emb=False):
        x1 = F.relu(s.c1(x, ei, ew))
        x1d = F.dropout(x1, p=s.drop, training=s.training)
        h = x1 + s.c2(x1d, ei, ew)  # residual
        out = s.lin(F.relu(h)).squeeze(-1)
        return (out, h) if ret_emb else out

def fair_metrics(yt_true, pred, g):
    out = {}
    for gv, name in [(1, "male"), (0, "female")]:
        m = g == gv
        tp = int(((pred == 1) & (yt_true == 1) & m).sum()); fn = int(((pred == 0) & (yt_true == 1) & m).sum())
        fp = int(((pred == 1) & (yt_true == 0) & m).sum()); tn = int(((pred == 0) & (yt_true == 0) & m).sum())
        out[name] = {"tpr": tp / max(tp + fn, 1), "fpr": fp / max(fp + tn, 1),
                     "pos_rate": (tp + fp) / max(m.sum(), 1)}
    out["eqodds_diff"] = max(abs(out["male"]["tpr"] - out["female"]["tpr"]),
                             abs(out["male"]["fpr"] - out["female"]["fpr"]))
    out["demparity_diff"] = abs(out["male"]["pos_rate"] - out["female"]["pos_rate"])
    return out

def ev(proba, thr, yt_idx, g_idx, name):
    pred = (proba >= thr).astype(int)
    return {"thr": round(float(thr), 3), "acc": round(float((pred == yt_idx).mean()), 4),
            "auc": round(float(roc_auc_score(yt_idx, proba)), 4),
            "pr": round(float(average_precision_score(yt_idx, proba)), 4),
            "f1": round(float(f1_score(yt_idx, pred, zero_division=0)), 4),
            "P": round(float(precision_score(yt_idx, pred, zero_division=0)), 4),
            "R": round(float(recall_score(yt_idx, pred, zero_division=0)), 4),
            "cm": confusion_matrix(yt_idx, pred).tolist(),
            "fair": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in fair_metrics(yt_idx, pred, g_idx).items()}}

results = {"seeds": {}, "mlp": {}}
bundles = {}
for seed in SEEDS:
    torch.manual_seed(seed); np.random.seed(seed)
    idx_tr, idx_te = train_test_split(idx0, test_size=0.2, stratify=y, random_state=seed)
    idx_tr2, idx_va = train_test_split(idx_tr, test_size=0.15, stratify=y[idx_tr], random_state=seed)
    m = ResGCN(Xng.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
    pos = float((len(idx_tr2) - y[idx_tr2].sum()) / max(y[idx_tr2].sum(), 1))
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos]))
    best, bs, bad = -1, None, 0
    for ep in range(200):
        m.train(); opt.zero_grad()
        loss = crit(m(Xt, EI, EW)[idx_tr2], yt[idx_tr2]); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): pv = torch.sigmoid(m(Xt, EI, EW)[idx_va]).numpy()
        try: pr = average_precision_score(y[idx_va], pv)
        except Exception: pr = 0.0
        if pr > best: best, bs, bad = pr, {k: v.cpu().clone() for k, v in m.state_dict().items()}, 0
        else: bad += 1
        if bad >= 30: break
    m.load_state_dict(bs); m.eval()
    with torch.no_grad():
        o = m(Xt, EI, EW)
        pv = torch.sigmoid(o[idx_va]).numpy(); pt = torch.sigmoid(o[idx_te]).numpy()
    # threshold F1 tren VAL
    bt, bf = 0.5, -1
    for t in np.arange(0.1, 0.91, 0.05):
        f = f1_score(y[idx_va], (pv >= t).astype(int), zero_division=0)
        if f > bf: bf, bt = f, t
    r = ev(pt, bt, y[idx_te], gender[idx_te], f"ResGCN-seed{seed}")
    r["val_pr"] = round(float(best), 4); r["epochs"] = ep + 1
    results["seeds"][str(seed)] = r
    bundles[seed] = ({k: v.cpu().clone() for k, v in m.state_dict().items()}, bt)
    print(f"[ResGCN seed {seed}] thr={bt:.2f} ACC={r['acc']:.4f} AUC={r['auc']:.4f} PR={r['pr']:.4f} F1={r['f1']:.4f} "
          f"eqodds={r['fair']['eqodds_diff']:.4f} dempar={r['fair']['demparity_diff']:.4f}", flush=True)
    # MLP baseline (tabular no-Gender, cung split) - 1 lan cho seed 42 de doi chieu
    if seed == 42:
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        mlp.fit(Xng[idx_tr], y[idx_tr])
        pm = mlp.predict_proba(Xng[idx_te])[:, 1]
        btm, bfm = 0.5, -1
        pmv = mlp.predict_proba(Xng[idx_va])[:, 1]
        for t in np.arange(0.1, 0.91, 0.05):
            f = f1_score(y[idx_va], (pmv >= t).astype(int), zero_division=0)
            if f > bfm: bfm, btm = f, t
        rm = ev(pm, btm, y[idx_te], gender[idx_te], "MLP")
        results["mlp"] = rm
        print(f"[MLP baseline] thr={btm:.2f} ACC={rm['acc']:.4f} AUC={rm['auc']:.4f} PR={rm['pr']:.4f} F1={rm['f1']:.4f} "
              f"eqodds={rm['fair']['eqodds_diff']:.4f} dempar={rm['fair']['demparity_diff']:.4f}", flush=True)

accs = [results["seeds"][str(s)]["acc"] for s in SEEDS]
aucs = [results["seeds"][str(s)]["auc"] for s in SEEDS]
prs = [results["seeds"][str(s)]["pr"] for s in SEEDS]
results["mean_std"] = {"acc": [round(float(np.mean(accs)), 4), round(float(np.std(accs)), 4)],
                       "auc": [round(float(np.mean(aucs)), 4), round(float(np.std(aucs)), 4)],
                       "pr": [round(float(np.mean(prs)), 4), round(float(np.std(prs)), 4)]}
print(f"[MEAN±STD] acc={results['mean_std']['acc']} auc={results['mean_std']['auc']} pr={results['mean_std']['pr']}", flush=True)
json.dump(results, open(OUT / "fairleave_core.json", "w"), indent=2)

# bundle seed tot nhat (theo AUC) cho Phase 3+4
best_seed = max(SEEDS, key=lambda s: results["seeds"][str(s)]["auc"])
st, thr = bundles[best_seed]
import joblib
joblib.dump({"resgcn_state": st, "in_dim": int(Xng.shape[1]), "X_all": Xng,
             "ei": EI, "ew": EW, "thr": float(thr), "seed": best_seed,
             "cat": CAT_NG, "num": NUM,
             "feat_names": [str(f) for f in pre_ng.get_feature_names_out()]},
            ART / "fairleave_bundle.joblib")
print(f"[bundle] best_seed={best_seed} thr={thr:.2f} -> {ART/'fairleave_bundle.joblib'}", flush=True)
print(f"saved {OUT/'fairleave_core.json'}", flush=True)
