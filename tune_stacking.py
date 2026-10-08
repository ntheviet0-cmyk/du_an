"""tune_stacking.py - Vong 3: stacking SAGE-full + SAGE-top20 + GAT + GCN + RF -> LogReg.
Meta-learner fit TREN VAL, do TEST 1 lan duy nhat."""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, GCNConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import kneighbors_graph
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, confusion_matrix, accuracy_score)

warnings.filterwarnings("ignore")
SEED = 42
OUT = Path("reports/ibm_gnn")

df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
df["Attrition"] = (df["Attrition"] == "Yes").astype(int)
df = df.drop(columns=["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"])
CAT = ["BusinessTravel","Department","EducationField","Gender","JobRole","MaritalStatus","OverTime"]
NUM = [c for c in df.columns if c not in CAT+["Attrition"]]
y = df["Attrition"].values
idx = np.arange(len(df))
idx_tr, idx_te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
idx_tr2, idx_va = train_test_split(idx_tr, test_size=0.15, stratify=y[idx_tr], random_state=SEED)

pre = ColumnTransformer([("num",StandardScaler(),NUM),("cat",OneHotEncoder(handle_unknown="ignore"),CAT)])
pre.fit(df.iloc[idx_tr][CAT+NUM])
Xf = pre.transform(df[CAT+NUM])
if hasattr(Xf,"toarray"): Xf = Xf.toarray()
Xf = Xf.astype(np.float32)
Fs,_ = f_classif(Xf[idx_tr], y[idx_tr]); Fs = np.nan_to_num(Fs, nan=0.0)
top20 = np.argsort(Fs)[::-1][:20]
X20 = Xf[:, top20]
print(f"[feat] full={Xf.shape[1]} top20={X20.shape[1]}", flush=True)

def knn_ei(X, k=15):
    A = kneighbors_graph(X, n_neighbors=k, mode="connectivity", metric="cosine", include_self=False)
    A = A.maximum(A.T)
    return torch.tensor(np.vstack(A.nonzero()), dtype=torch.long), A

EI_F, A_F = knn_ei(Xf)
EI_20, A_20 = knn_ei(X20)

class SAGE(torch.nn.Module):
    def __init__(s,d,h=64,drop=0.3):
        super().__init__()
        s.c1=SAGEConv(d,h); s.c2=SAGEConv(h,32); s.lin=torch.nn.Linear(32,1); s.drop=drop
    def forward(s,x,ei):
        x=F.relu(s.c1(x,ei)); x=F.dropout(x,p=s.drop,training=s.training)
        return s.lin(F.relu(s.c2(x,ei))).squeeze(-1)

class GAT(torch.nn.Module):
    def __init__(s,d,h=16,heads=8,drop=0.5):
        super().__init__()
        s.c1=GATConv(d,h,heads=heads,dropout=drop); s.c2=GATConv(h*heads,32,heads=1,dropout=drop)
        s.lin=torch.nn.Linear(32,1); s.drop=drop
    def forward(s,x,ei):
        x=F.elu(s.c1(x,ei)); x=F.dropout(x,p=s.drop,training=s.training)
        return s.lin(F.elu(s.c2(x,ei))).squeeze(-1)

class GCN(torch.nn.Module):
    def __init__(s,d,h=64,drop=0.5):
        super().__init__()
        s.c1=GCNConv(d,h); s.c2=GCNConv(h,32); s.lin=torch.nn.Linear(32,1); s.drop=drop
    def forward(s,x,ei):
        x=F.relu(s.c1(x,ei)); x=F.dropout(x,p=s.drop,training=s.training)
        return s.lin(F.relu(s.c2(x,ei))).squeeze(-1)

def train_gnn(make, X, ei, tag, h=None, drop=None, lr=0.01, wd=5e-4):
    torch.manual_seed(SEED); np.random.seed(SEED)
    Xt = torch.tensor(X); yt = torch.tensor(y, dtype=torch.float32)
    m = make(X.shape[1]) if h is None else make(X.shape[1], h=h, drop=drop)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    pos = float((len(idx_tr2)-y[idx_tr2].sum())/max(y[idx_tr2].sum(),1))
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos]))
    best, bs, bad = -1, None, 0
    for ep in range(200):
        m.train(); opt.zero_grad()
        out = m(Xt, ei); loss = crit(out[idx_tr2], yt[idx_tr2]); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): pv = torch.sigmoid(m(Xt, ei)[idx_va]).numpy()
        try: pr = average_precision_score(y[idx_va], pv)
        except Exception: pr = 0.0
        if pr > best: best, bs, bad = pr, {k:v.cpu().clone() for k,v in m.state_dict().items()}, 0
        else: bad += 1
        if bad >= 30: break
    m.load_state_dict(bs); m.eval()
    with torch.no_grad():
        o = m(torch.tensor(X), ei)
        pv = torch.sigmoid(o[idx_va]).numpy(); pt = torch.sigmoid(o[idx_te]).numpy()
    print(f"[{tag}] val_pr={best:.4f} ep={ep+1}", flush=True)
    return pv, pt

pv1, pt1 = train_gnn(SAGE, Xf, EI_F, "SAGE-full", h=64, drop=0.3, lr=0.01, wd=5e-5)
pv2, pt2 = train_gnn(SAGE, X20, EI_20, "SAGE-top20", h=64, drop=0.3, lr=0.01, wd=5e-5)
pv3, pt3 = train_gnn(GAT, Xf, EI_F, "GAT")
pv4, pt4 = train_gnn(GCN, Xf, EI_F, "GCN")
rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=1,
                            random_state=SEED, n_jobs=-1, class_weight="balanced")
rf.fit(Xf[idx_tr], y[idx_tr])
pv5 = rf.predict_proba(Xf[idx_va])[:,1]; pt5 = rf.predict_proba(Xf[idx_te])[:,1]
print("[RF] done", flush=True)

Zva = np.vstack([pv1,pv2,pv3,pv4,pv5]).T
Zte = np.vstack([pt1,pt2,pt3,pt4,pt5]).T
meta = LogisticRegression(max_iter=1000)
meta.fit(Zva, y[idx_va])
pe = meta.predict_proba(Zte)[:,1]
print(f"[meta] coef={meta.coef_.round(3).tolist()}", flush=True)

# nguong toi uu ACCURACY tren VAL (duyet meta tren val)
pva_meta = meta.predict_proba(Zva)[:,1]
bt, ba = 0.5, -1
for t in np.arange(0.1, 0.91, 0.05):
    a = float(((pva_meta >= t).astype(int) == y[idx_va]).mean())
    if a > ba: ba, bt = a, t
pred = (pe >= bt).astype(int)
yt_te = y[idx_te]
names = ["SAGE-full","SAGE-top20","GAT","GCN","RF"]
out = {"members_test_acc":{}, "thr_val":float(bt), "val_acc":float(ba)}
for n, pt in zip(names, [pt1,pt2,pt3,pt4,pt5]):
    b2, a2 = 0.5, -1
    Zv_col = Zva[:, names.index(n)]
    for t in np.arange(0.1, 0.91, 0.05):
        a = float(((Zv_col >= t).astype(int) == y[idx_va]).mean())
        if a > a2: a2, b2 = a, t
    out["members_test_acc"][n] = float((((pt >= b2).astype(int)) == yt_te).mean())
ens = {"test_acc":float((pred==yt_te).mean()),
       "test_auc":float(roc_auc_score(yt_te,pe)),
       "test_pr":float(average_precision_score(yt_te,pe)),
       "test_f1":float(f1_score(yt_te,pred,zero_division=0)),
       "test_P":float(precision_score(yt_te,pred,zero_division=0)),
       "test_R":float(recall_score(yt_te,pred,zero_division=0)),
       "cm":confusion_matrix(yt_te,pred).tolist(),
       "meta_coef":meta.coef_.round(4).tolist()}
out["ensemble"] = ens
json.dump(out, open(OUT/"ensemble90.json","w"), indent=2)
print(f"[ENSEMBLE-TEST-1-LAN] thr_val={bt:.2f} ACC={ens['test_acc']:.4f} AUC={ens['test_auc']:.4f} PR={ens['test_pr']:.4f} F1={ens['test_f1']:.4f}", flush=True)
print(f"members: {out['members_test_acc']}", flush=True)
print(f"saved {OUT/'ensemble90.json'}", flush=True)
