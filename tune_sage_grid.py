"""tune_sage_grid.py - Vong 1: grid-search SAGE 24 cau hinh, chon top-3 TREN VAL.
Test set KHONG duoc dung o vong nay (chi do 1 lan o buoc --final).
Chay theo chunk: python tune_sage_grid.py --start 0 --end 8  (lam 3 lan: 0-8, 8-16, 16-24)
Chot: python tune_sage_grid.py --final  (train lai top-3 + do test 1 lan)
"""
import argparse, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

warnings.filterwarnings("ignore")
SEED = 42
OUT = Path("reports/ibm_gnn"); OUT.mkdir(parents=True, exist_ok=True)
GRID_FILE = OUT / "sage_grid.json"

# ---------------- data (giong het cac script truoc, split seed 42)
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
X = pre.transform(df[CAT+NUM])
if hasattr(X,"toarray"): X = X.toarray()
X = X.astype(np.float32)

# graph k=15 cosine (da chot)
A = kneighbors_graph(X, n_neighbors=15, mode="connectivity", metric="cosine", include_self=False)
A = A.maximum(A.T)
EI = torch.tensor(np.vstack(A.nonzero()), dtype=torch.long)
Xt = torch.tensor(X); yt = torch.tensor(y, dtype=torch.float32)

class SAGE(torch.nn.Module):
    def __init__(s,d,h=64,drop=0.5):
        super().__init__()
        s.c1=SAGEConv(d,h); s.c2=SAGEConv(h,32); s.lin=torch.nn.Linear(32,1); s.drop=drop
    def forward(s,x,ei):
        x=F.relu(s.c1(x,ei)); x=F.dropout(x,p=s.drop,training=s.training)
        return s.lin(F.relu(s.c2(x,ei))).squeeze(-1)

def best_thr_acc(pv, yv):
    bt, ba = 0.5, -1
    for t in np.arange(0.1, 0.91, 0.05):
        a = float(((pv >= t).astype(int) == yv).mean())
        if a > ba: ba, bt = a, t
    return bt, ba

def train_cfg(h, drop, lr, wd, seed_run):
    torch.manual_seed(seed_run); np.random.seed(seed_run)
    m = SAGE(X.shape[1], h=h, drop=drop)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    pos = float((len(idx_tr2)-y[idx_tr2].sum())/max(y[idx_tr2].sum(),1))
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos]))
    best, bs, bad = -1, None, 0
    for ep in range(200):
        m.train(); opt.zero_grad()
        out = m(Xt, EI)
        loss = crit(out[idx_tr2], yt[idx_tr2]); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): pv = torch.sigmoid(m(Xt, EI)[idx_va]).numpy()
        try: pr = average_precision_score(y[idx_va], pv)
        except Exception: pr = 0.0
        if pr > best: best, bs, bad = pr, {k:v.cpu().clone() for k,v in m.state_dict().items()}, 0
        else: bad += 1
        if bad >= 30: break
    m.load_state_dict(bs)
    m.eval()
    with torch.no_grad():
        pv = torch.sigmoid(m(Xt, EI)[idx_va]).numpy()
    bt, ba = best_thr_acc(pv, y[idx_va])
    f1 = float(f1_score(y[idx_va], (pv>=bt).astype(int), zero_division=0))
    return {"h":h,"drop":drop,"lr":lr,"wd":wd,
            "val_pr":float(best),"val_acc":ba,"val_thr":float(bt),"val_f1":f1,"epochs":ep+1}

CONFIGS = [{"h":h,"drop":d,"lr":l,"wd":w}
           for h in [32,64,128] for d in [0.3,0.5] for l in [0.01,0.005] for w in [5e-4,5e-5]]

def load():
    if GRID_FILE.exists(): return json.load(open(GRID_FILE))
    return {"runs":[]}

def key(c): return (c["h"],c["drop"],c["lr"],c["wd"])

ap = argparse.ArgumentParser()
ap.add_argument("--start", type=int, default=None)
ap.add_argument("--end", type=int, default=None)
ap.add_argument("--final", action="store_true")
a = ap.parse_args()

if not a.final:
    data = load()
    done = {tuple(r["cfg"][k] for k in ["h","drop","lr","wd"]) for r in data["runs"]}
    for i in range(a.start, a.end):
        c = CONFIGS[i]
        if key(c) in done:
            print(f"[skip {i}] {c} da chay"); continue
        r = train_cfg(seed_run=SEED*1000+i, **c)
        r["cfg"] = c; r["idx"] = i
        data["runs"].append(r)
        json.dump(data, open(GRID_FILE,"w"), indent=2)
        print(f"[{i}/24] {c} -> val_acc={r['val_acc']:.4f} (thr={r['val_thr']:.2f}) val_pr={r['val_pr']:.4f} val_f1={r['val_f1']:.4f}", flush=True)
    print(f"saved {GRID_FILE}")
else:
    # chon top-3 theo val_acc (tiebreak val_pr), train lai + do TEST 1 lan duy nhat
    data = load()
    assert len(data["runs"]) == 24, f"moi {len(data['runs'])}/24 runs - chay het grid truoc"
    top3 = sorted(data["runs"], key=lambda r: (r["val_acc"], r["val_pr"]), reverse=True)[:3]
    print("TOP-3 tren VAL:")
    for r in top3: print(f"  cfg={r['cfg']} val_acc={r['val_acc']:.4f} val_pr={r['val_pr']:.4f}")
    from sklearn.metrics import precision_score, recall_score, confusion_matrix
    data["test_once"] = []
    for r in top3:
        c = r["cfg"]
        tm = train_cfg(seed_run=SEED*1000+r["idx"], **c)  # train lai y het (khong dung test de chon)
        # lay proba test 1 lan
        torch.manual_seed(SEED*1000+r["idx"]); np.random.seed(SEED*1000+r["idx"])
        m = SAGE(X.shape[1], h=c["h"], drop=c["drop"])
        # nap lai bang cach train lai full (giong train_cfg) roi do
        opt = torch.optim.Adam(m.parameters(), lr=c["lr"], weight_decay=c["wd"])
        pos = float((len(idx_tr2)-y[idx_tr2].sum())/max(y[idx_tr2].sum(),1))
        crit = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos]))
        best, bs, bad = -1, None, 0
        for ep in range(200):
            m.train(); opt.zero_grad()
            out = m(Xt, EI); loss = crit(out[idx_tr2], yt[idx_tr2]); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad(): pv = torch.sigmoid(m(Xt, EI)[idx_va]).numpy()
            try: pr = average_precision_score(y[idx_va], pv)
            except Exception: pr = 0.0
            if pr > best: best, bs, bad = pr, {k:v.cpu().clone() for k,v in m.state_dict().items()}, 0
            else: bad += 1
            if bad >= 30: break
        m.load_state_dict(bs); m.eval()
        with torch.no_grad():
            pv = torch.sigmoid(m(Xt, EI)[idx_va]).numpy()
            pt = torch.sigmoid(m(Xt, EI)[idx_te]).numpy()
        bt, _ = best_thr_acc(pv, y[idx_va])  # nguong tu VAL
        pred = (pt >= bt).astype(int)
        yt_te = y[idx_te]
        acc = float((pred == yt_te).mean())
        t = {"cfg":c,"thr_from_val":float(bt),"test_acc":acc,
             "test_auc":float(roc_auc_score(yt_te,pt)),
             "test_pr":float(average_precision_score(yt_te,pt)),
             "test_f1":float(f1_score(yt_te,pred,zero_division=0)),
             "test_P":float(precision_score(yt_te,pred,zero_division=0)),
             "test_R":float(recall_score(yt_te,pred,zero_division=0)),
             "cm":confusion_matrix(yt_te,pred).tolist()}
        data["test_once"].append(t)
        json.dump(data, open(GRID_FILE,"w"), indent=2)
        print(f"[TEST-1-LAN] cfg={c} thr_val={bt:.2f} -> ACC={acc:.4f} AUC={t['test_auc']:.4f} PR={t['test_pr']:.4f} F1={t['test_f1']:.4f}", flush=True)
    print(f"saved {GRID_FILE}")
