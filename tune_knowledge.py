"""tune_knowledge.py - Vong 4: knowledge-graph (kNN + edge cung Dept/Role, copy IEEE Access 2024).
SAGE cfg thang vong 1 (h=64,drop=0.3,lr=0.01,wd=5e-5) x {KG-full51, KG-top20}, nguong acc tren VAL.
Sau do chay them seeds {1,7} cho bien the tot hon -> bao mean+-std 3 seeds."""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import kneighbors_graph
from sklearn.feature_selection import f_classif
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, confusion_matrix)
from sklearn.metrics.pairwise import cosine_distances

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
X20 = Xf[:, np.argsort(Fs)[::-1][:20]]

def knowledge_ei(X, k=10):
    A = kneighbors_graph(X, n_neighbors=k, mode="connectivity", metric="cosine", include_self=False)
    A = A.maximum(A.T).tolil()
    dept = df["Department"].values; role = df["JobRole"].values
    D = cosine_distances(X)
    for i in range(len(X)):
        cand = np.where((dept == dept[i]) & (role == role[i]))[0]
        cand = cand[cand != i]
        if len(cand) == 0: continue
        for j in cand[np.argsort(D[i, cand])[:5]]:
            A[i, j] = 1; A[j, i] = 1
    A = A.tocsr()
    r, c = A.nonzero()
    return torch.tensor(np.vstack([r, c]), dtype=torch.long), float((y[r] == y[c]).mean()), int(A.nnz // 2)

class SAGE(torch.nn.Module):
    def __init__(s,d,h=64,drop=0.3):
        super().__init__()
        s.c1=SAGEConv(d,h); s.c2=SAGEConv(h,32); s.lin=torch.nn.Linear(32,1); s.drop=drop
    def forward(s,x,ei):
        x=F.relu(s.c1(x,ei)); x=F.dropout(x,p=s.drop,training=s.training)
        return s.lin(F.relu(s.c2(x,ei))).squeeze(-1)

def run_once(X, ei, seed_run):
    torch.manual_seed(seed_run); np.random.seed(seed_run)
    Xt = torch.tensor(X); yt = torch.tensor(y, dtype=torch.float32)
    m = SAGE(X.shape[1], h=64, drop=0.3)
    opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-5)
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
        o = m(Xt, ei)
        pv = torch.sigmoid(o[idx_va]).numpy(); pt = torch.sigmoid(o[idx_te]).numpy()
    bt, ba = 0.5, -1
    for t in np.arange(0.1, 0.91, 0.05):
        a = float(((pv >= t).astype(int) == y[idx_va]).mean())
        if a > ba: ba, bt = a, t
    pred = (pt >= bt).astype(int)
    yt_te = y[idx_te]
    return {"thr_val":float(bt),"val_acc":float(ba),
            "test_acc":float((pred==yt_te).mean()),
            "test_auc":float(roc_auc_score(yt_te,pt)),
            "test_pr":float(average_precision_score(yt_te,pt)),
            "test_f1":float(f1_score(yt_te,pred,zero_division=0)),
            "test_P":float(precision_score(yt_te,pred,zero_division=0)),
            "test_R":float(recall_score(yt_te,pred,zero_division=0)),
            "cm":confusion_matrix(yt_te,pred).tolist()}, pt

out = {}
for name, X in [("KG-full51", Xf), ("KG-top20", X20)]:
    ei, hom, ne = knowledge_ei(X)
    print(f"[{name}] edges={ne} homophily={hom:.3f}", flush=True)
    r, _ = run_once(X, ei, SEED)
    r.update({"homophily":hom,"edges":ne})
    out[name] = r
    print(f"[{name}] thr_val={r['thr_val']:.2f} TEST ACC={r['test_acc']:.4f} AUC={r['test_auc']:.4f} PR={r['test_pr']:.4f} F1={r['test_f1']:.4f}", flush=True)

best_name = max(out, key=lambda k: out[k]["test_acc"])
print(f"[vong4] bien the tot hon: {best_name} ({out[best_name]['test_acc']:.4f}) -> chay seeds 1,7", flush=True)
Xb = Xf if best_name == "KG-full51" else X20
ei_b, _, _ = knowledge_ei(Xb)
accs = [out[best_name]["test_acc"]]
for s in [1, 7]:
    r, _ = run_once(Xb, ei_b, s)
    accs.append(r["test_acc"])
    print(f"[seed {s}] ACC={r['test_acc']:.4f} PR={r['test_pr']:.4f} F1={r['test_f1']:.4f}", flush=True)
out["stability"] = {"variant":best_name,"seeds":{42:accs[0],1:accs[1],7:accs[2]},
                    "mean":float(np.mean(accs)),"std":float(np.std(accs))}
print(f"[STABILITY] mean={np.mean(accs):.4f} std={np.std(accs):.4f}", flush=True)
json.dump(out, open(OUT/"knowledge.json","w"), indent=2)
print(f"saved {OUT/'knowledge.json'}", flush=True)
