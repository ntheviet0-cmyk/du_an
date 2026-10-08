"""build_li.py - Tai hien Liqin Li (CSMS 2026, DOI 10.23919/CSMS.2025.0035):
Employee Relationship Network (simulation) + GCN / GraphSAGE / GAT
+ stacked ensemble (LogisticRegression) -> 93.2% accuracy (paper).
Relationship score = quan he to chuc (dept/role/overtime/satisfaction...)
+ similarity profile. Early-stop theo val ACC (giong paper). Test do 1 lan.
"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
SEED = 42
OUT = Path("reports/ibm_gnn"); OUT.mkdir(parents=True, exist_ok=True)

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

# ---------------- Relationship network (simulation, kieu Li) ----------------
# diem quan he = cung dept / role / overtime / muc satisfaction gan nhau / manager-tenure gan
dept = df["Department"].values; role = df["JobRole"].values
ot = df["OverTime"].values
jsat = df["JobSatisfaction"].values; esat = df["EnvironmentSatisfaction"].values
wlb = df["WorkLifeBalance"].values; ym = df["YearsWithCurrManager"].values
S = np.maximum(cosine_similarity(Xf), 0)
N = len(df)
R = np.zeros((N, N), dtype=np.float32)
same_dept = (dept[:,None] == dept[None,:]).astype(np.float32)
same_role = (role[:,None] == role[None,:]).astype(np.float32)
same_ot = (ot[:,None] == ot[None,:]).astype(np.float32)
sat_close = (np.abs(jsat[:,None]-jsat[None,:]) + np.abs(esat[:,None]-esat[None,:]
             ) + np.abs(wlb[:,None]-wlb[None,:]) <= 2).astype(np.float32)
mgr_close = (np.abs(ym[:,None]-ym[None,:]) <= 1).astype(np.float32)
R = 0.30*same_dept + 0.25*same_role + 0.15*same_ot + 0.15*sat_close + 0.15*mgr_close
Score = 0.5*R + 0.5*S
np.fill_diagonal(Score, 0)
K = 15
rows, cols, ws = [], [], []
for i in range(N):
    js = np.argsort(Score[i])[::-1][:K]
    for j in js:
        rows.append(i); cols.append(int(j)); ws.append(float(Score[i, j]))
ei = torch.tensor([rows, cols], dtype=torch.long)
ew = torch.tensor(ws, dtype=torch.float32)
ei, ew = to_undirected(ei, ew, reduce="max")
r_, c_ = ei.numpy()
print(f"[li-graph] edges={ei.shape[1]//2} homophily={float((y[r_]==y[c_]).mean()):.3f} mean_w={float(ew.mean()):.3f}", flush=True)
Xt = torch.tensor(Xf); yt = torch.tensor(y, dtype=torch.float32)

class GCN(torch.nn.Module):
    def __init__(s,d,h=64,drop=0.5):
        super().__init__()
        s.c1=GCNConv(d,h); s.c2=GCNConv(h,32); s.lin=torch.nn.Linear(32,1); s.drop=drop
    def forward(s,x,ei,ew=None):
        x=F.relu(s.c1(x,ei,ew)); x=F.dropout(x,p=s.drop,training=s.training)
        return s.lin(F.relu(s.c2(x,ei,ew))).squeeze(-1)

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

def train_gnn(m, tag, use_w=False, lr=0.01, wd=5e-4, drop=None, h=None):
    torch.manual_seed(SEED); np.random.seed(SEED)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    pos = float((len(idx_tr2)-y[idx_tr2].sum())/max(y[idx_tr2].sum(),1))
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos]))
    fwd = lambda mm: mm(Xt, ei, ew) if use_w else mm(Xt, ei)
    best, bs, bad = -1, None, 0
    for ep in range(200):
        m.train(); opt.zero_grad()
        loss = crit(fwd(m)[idx_tr2], yt[idx_tr2]); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): pv = torch.sigmoid(fwd(m)[idx_va]).numpy()
        acc = float(((pv >= 0.5).astype(int) == y[idx_va]).mean())  # paper: accuracy
        if acc > best: best, bs, bad = acc, {k:v.cpu().clone() for k,v in m.state_dict().items()}, 0
        else: bad += 1
        if bad >= 30: break
    m.load_state_dict(bs); m.eval()
    with torch.no_grad():
        o = fwd(m)
        return torch.sigmoid(o[idx_va]).numpy(), torch.sigmoid(o[idx_te]).numpy(), best, ep+1

def ev(pva, pte, name):
    bt, ba = 0.5, float(((pva>=0.5).astype(int)==y[idx_va]).mean())
    for t in np.arange(0.1, 0.91, 0.05):
        a = float(((pva >= t).astype(int) == y[idx_va]).mean())
        if a > ba: ba, bt = a, t
    pred = (pte >= bt).astype(int)
    yt_te = y[idx_te]
    r = {"thr_val":round(float(bt),2),"val_acc":round(float(ba),4),
         "acc":round(float((pred==yt_te).mean()),4),
         "auc":round(float(roc_auc_score(yt_te,pte)),4),
         "pr":round(float(average_precision_score(yt_te,pte)),4),
         "f1":round(float(f1_score(yt_te,pred,zero_division=0)),4),
         "P":round(float(precision_score(yt_te,pred,zero_division=0)),4),
         "R":round(float(recall_score(yt_te,pred,zero_division=0)),4),
         "cm":confusion_matrix(yt_te,pred).tolist()}
    print(f"[{name}] thr={bt:.2f} ACC={r['acc']:.4f} AUC={r['auc']:.4f} PR={r['pr']:.4f} F1={r['f1']:.4f} P={r['P']:.3f} R={r['R']:.3f}", flush=True)
    return r

print("--- members (early-stop val ACC) ---", flush=True)
q1a,q1t,v1,e1 = train_gnn(GCN(Xf.shape[1]), "GCN", use_w=True)
q2a,q2t,v2,e2 = train_gnn(SAGE(Xf.shape[1],h=64,drop=0.3), "SAGE", lr=0.01, wd=5e-5)
q3a,q3t,v3,e3 = train_gnn(GAT(Xf.shape[1]), "GAT")
print(f"[val-acc] GCN={v1:.4f}({e1}) SAGE={v2:.4f}({e2}) GAT={v3:.4f}({e3})", flush=True)
m1 = ev(q1a,q1t,"Li-GCN"); m2 = ev(q2a,q2t,"Li-SAGE"); m3 = ev(q3a,q3t,"Li-GAT")

print("--- stacked ensemble (LogReg, standardized) ---", flush=True)
Zva = np.vstack([q1a,q2a,q3a]).T; Zte = np.vstack([q1t,q2t,q3t]).T
sc = StandardScaler().fit(Zva)
meta = LogisticRegression(max_iter=1000).fit(sc.transform(Zva), y[idx_va])
pva_m = meta.predict_proba(sc.transform(Zva))[:,1]
pte_m = meta.predict_proba(sc.transform(Zte))[:,1]
print(f"[meta] coef={meta.coef_.round(3).tolist()}", flush=True)
mE = ev(pva_m, pte_m, "Li-STACK-TEST-1-LAN")
mE["meta_coef"] = meta.coef_.round(4).tolist()

json.dump({"members":{"GCN":m1,"SAGE":m2,"GAT":m3},"ensemble":mE,
           "graph":{"edges":int(ei.shape[1]//2),"k":K}},
          open(OUT/"li_results.json","w"), indent=2)
print(f"saved {OUT/'li_results.json'}", flush=True)
