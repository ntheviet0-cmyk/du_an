"""tune_feat_select.py - Vong 2: X-full51 vs X-top20 (f_classif TREN TRAIN) + threshold acc tren VAL.
Cau hinh SAGE thang vong 1: h=64, drop=0.3, lr=0.01, wd=5e-5. Do test 1 lan moi ban (2 lans tong)."""
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

warnings.filterwarnings("ignore")
SEED = 42
OUT = Path("reports/ibm_gnn")
H, DROP, LR, WD = 64, 0.3, 0.01, 5e-5

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
fn = np.array([str(f) for f in pre.get_feature_names_out()])
Xf = pre.transform(df[CAT+NUM])
if hasattr(Xf,"toarray"): Xf = Xf.toarray()
Xf = Xf.astype(np.float32)

# chon top-20 bang f_classif CHI TREN TRAIN (idx_tr) -> khong leak val/test
Fs, _ = f_classif(Xf[idx_tr], y[idx_tr])
Fs = np.nan_to_num(Fs, nan=0.0)
top20 = np.argsort(Fs)[::-1][:20]
print("[feat] top20:", fn[top20].tolist())

class SAGE(torch.nn.Module):
    def __init__(s,d,h=64,drop=0.3):
        super().__init__()
        s.c1=SAGEConv(d,h); s.c2=SAGEConv(h,32); s.lin=torch.nn.Linear(32,1); s.drop=drop
    def forward(s,x,ei):
        x=F.relu(s.c1(x,ei)); x=F.dropout(x,p=s.drop,training=s.training)
        return s.lin(F.relu(s.c2(x,ei))).squeeze(-1)

def run(X, name):
    torch.manual_seed(SEED); np.random.seed(SEED)
    A = kneighbors_graph(X, n_neighbors=15, mode="connectivity", metric="cosine", include_self=False)
    A = A.maximum(A.T)
    ei = torch.tensor(np.vstack(A.nonzero()), dtype=torch.long)
    Xt = torch.tensor(X); yt = torch.tensor(y, dtype=torch.float32)
    m = SAGE(X.shape[1], h=H, drop=DROP)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
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
        pv = torch.sigmoid(m(Xt, ei)[idx_va]).numpy()
        pt = torch.sigmoid(m(Xt, ei)[idx_te]).numpy()
    bt, ba = 0.5, -1  # nguong toi uu ACCURACY tren VAL
    for t in np.arange(0.1, 0.91, 0.05):
        a = float(((pv >= t).astype(int) == y[idx_va]).mean())
        if a > ba: ba, bt = a, t
    pred = (pt >= bt).astype(int)
    yt_te = y[idx_te]
    r = {"variant":name,"dim":int(X.shape[1]),"edges":int(A.nnz//2),
         "thr_val":float(bt),"val_acc":float(ba),
         "test_acc":float((pred==yt_te).mean()),
         "test_auc":float(roc_auc_score(yt_te,pt)),
         "test_pr":float(average_precision_score(yt_te,pt)),
         "test_f1":float(f1_score(yt_te,pred,zero_division=0)),
         "test_P":float(precision_score(yt_te,pred,zero_division=0)),
         "test_R":float(recall_score(yt_te,pred,zero_division=0)),
         "cm":confusion_matrix(yt_te,pred).tolist()}
    print(f"[{name}] dim={X.shape[1]} thr_val={bt:.2f} -> TEST ACC={r['test_acc']:.4f} AUC={r['test_auc']:.4f} PR={r['test_pr']:.4f} F1={r['test_f1']:.4f}", flush=True)
    return r

out = {"full51": run(Xf, "full51"), "top20": run(Xf[:, top20], "top20"),
       "top20_features": fn[top20].tolist()}
json.dump(out, open(OUT/"feat_select.json","w"), indent=2)
print(f"saved {OUT/'feat_select.json'}")
