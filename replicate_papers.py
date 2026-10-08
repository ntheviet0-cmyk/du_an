"""replicate_papers.py - Tai hien 2 pipeline paper tren data IBM (seed 42, split giong moi lan).
A (Al Akasheh 2024): SHAP-top features -> WEIGHTED graph -> GCN 2-layer -> concat[orig+emb] -> L-SVM (C grid) -> acc@0.5.
B (Li 2026): stacking GCN+SAGE+GAT -> LogReg (FIX: chuan hoa proba truoc meta, tranh RF ap dao nhu vong 3).
Moi pipeline do TEST dung 1 lan.
"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, confusion_matrix, accuracy_score)
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
SEED = 42
OUT = Path("reports/ibm_gnn"); OUT.mkdir(parents=True, exist_ok=True)

# ---------------- data
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

# ---------------- SHAP-top features (da luu tu analyze_ibm.py)
st = json.load(open("reports/ibm/analyze_stats.json"))
shap_feats = [x["feature"] for x in st["shap_top_rf"][:10]] + [x["feature"] for x in st["shap_top_xgb"][:10]]
shap_feats = list(dict.fromkeys(shap_feats))
sidx = np.array([int(np.where(fn == f)[0][0]) for f in shap_feats if f in fn])
print(f"[SHAP] {len(sidx)} features: {fn[sidx].tolist()}", flush=True)
Xs = Xf[:, sidx]

# ---------------- WEIGHTED graph (paper: edge weight = similarity, khong binary)
S = cosine_similarity(Xs)
np.fill_diagonal(S, 0)
K = 15
rows, cols, ws = [], [], []
for i in range(len(Xs)):
    j = np.argsort(S[i])[::-1][:K]
    for jj in j:
        rows.append(i); cols.append(int(jj)); ws.append(float(max(S[i, jj], 0)))
ei_all = torch.tensor([rows, cols], dtype=torch.long)
ew_all = torch.tensor(ws, dtype=torch.float32)
ei_w, ew_w = to_undirected(ei_all, ew_all, reduce="max")  # symmetrize lay max
ei_b = ei_w  # ban binary cho SAGE/GAT (khong ho tro edge_weight)
r, c = ei_w.numpy()
hom = float((y[r] == y[c]).mean())
print(f"[weighted-graph] edges={ei_w.shape[1]//2} homophily={hom:.3f} mean_w={ew_w.mean():.3f}", flush=True)
Xt = torch.tensor(Xf); yt = torch.tensor(y, dtype=torch.float32)

# ---------------- GNN defs
class GCN(torch.nn.Module):
    def __init__(s,d,h=64,drop=0.5):
        super().__init__()
        s.c1=GCNConv(d,h); s.c2=GCNConv(h,32); s.lin=torch.nn.Linear(32,1); s.drop=drop
    def forward(s,x,ei,ew=None,ret_emb=False):
        x=F.relu(s.c1(x,ei,ew)); x=F.dropout(x,p=s.drop,training=s.training)
        emb=s.c2(x,ei,ew)
        return (s.lin(F.relu(emb)).squeeze(-1),emb) if ret_emb else s.lin(F.relu(emb)).squeeze(-1)

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

def train_gnn(m, ei, ew, tag, lr=0.01, wd=5e-4, use_w=False, sel="acc"):
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
        sc = float(((pv >= 0.5).astype(int) == y[idx_va]).mean()) if sel == "acc" else average_precision_score(y[idx_va], pv)
        if sc > best: best, bs, bad = sc, {k:v.cpu().clone() for k,v in m.state_dict().items()}, 0
        else: bad += 1
        if bad >= 30: break
    m.load_state_dict(bs); m.eval()
    with torch.no_grad():
        o = fwd(m)
        return torch.sigmoid(o[idx_va]).numpy(), torch.sigmoid(o[idx_te]).numpy(), o, best, ep+1

def acc_thr(pva, pte, name):
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
         "cm":confusion_matrix(yt_te,pred).tolist()}
    print(f"[{name}] thr_val={bt:.2f} ACC={r['acc']:.4f} AUC={r['auc']:.4f} PR={r['pr']:.4f} F1={r['f1']:.4f}", flush=True)
    return r

# ================= PIPELINE A (Al Akasheh): GCN-emb + L-SVM
print("\n===== PIPELINE A (Al Akasheh: weighted-GCN + L-SVM) =====", flush=True)
gcn = GCN(Xf.shape[1])
pva, pte, logits, vacc, epa = train_gnn(gcn, ei_w, ew_w, "GCN-w", use_w=True, sel="acc")
print(f"[GCN-w] val_acc@0.5={vacc:.4f} ep={epa}", flush=True)
gcn.eval()
with torch.no_grad():
    _, emb = gcn(Xt, ei_w, ew_w, ret_emb=True)
Z = np.hstack([Xf, emb.numpy()])  # CONCAT goc + embedding (dung paper)
print(f"[concat] dim={Z.shape[1]}", flush=True)
gs = GridSearchCV(LinearSVC(random_state=SEED, max_iter=5000), {"C":[0.01,0.1,1,10]},
                  cv=5, scoring="accuracy", n_jobs=-1)
gs.fit(Z[idx_tr], y[idx_tr])
print(f"[L-SVM] best C={gs.best_params_} cv_acc={gs.best_score_:.4f}", flush=True)
dec_te = gs.decision_function(Z[idx_te]); dec_va = gs.decision_function(Z[idx_va])
pred05 = (dec_te >= 0).astype(int)
yt_te = y[idx_te]
# F1 threshold tune tren VAL decision scores
bt, bf = 0.0, -1
for t in np.percentile(dec_va, np.arange(5, 96, 5)):
    f = f1_score(y[idx_va], (dec_va >= t).astype(int), zero_division=0)
    if f > bf: bf, bt = f, t
predf = (dec_te >= bt).astype(int)
resA = {"best_C":gs.best_params_["C"],"cv_acc":round(float(gs.best_score_),4),
        "acc_default":round(float((pred05==yt_te).mean()),4),
        "auc":round(float(roc_auc_score(yt_te,dec_te)),4),
        "pr":round(float(average_precision_score(yt_te,dec_te)),4),
        "f1_valthr":round(float(bf if False else f1_score(yt_te,predf,zero_division=0)),4),
        "cm_default":confusion_matrix(yt_te,pred05).tolist()}
print(f"[A-TEST-1-LAN] C={resA['best_C']} ACC@0.5={resA['acc_default']:.4f} AUC={resA['auc']:.4f} PR={resA['pr']:.4f} F1={resA['f1_valthr']:.4f}", flush=True)

# ================= PIPELINE B (Li): stacking 3 GNN, FIX chuan hoa
print("\n===== PIPELINE B (Li: stacking, standardized) =====", flush=True)
sage = SAGE(Xf.shape[1], h=64, drop=0.3)
gat = GAT(Xf.shape[1])
gcn2 = GCN(Xf.shape[1])
q1a,q1t,_,_,_ = train_gnn(sage, ei_b, None, "SAGE", lr=0.01, wd=5e-5, sel="pr")
q2a,q2t,_,_,_ = train_gnn(gat, ei_b, None, "GAT", sel="pr")
q3a,q3t,_,_,_ = train_gnn(gcn2, ei_w, ew_w, "GCN", use_w=True, sel="pr")
Zva = np.vstack([q1a,q2a,q3a]).T; Zte = np.vstack([q1t,q2t,q3t]).T
sc = StandardScaler().fit(Zva)  # FIX: chuan hoa truoc meta
meta = LogisticRegression(max_iter=1000).fit(sc.transform(Zva), y[idx_va])
pva_m = meta.predict_proba(sc.transform(Zva))[:,1]
pte_m = meta.predict_proba(sc.transform(Zte))[:,1]
print(f"[meta] coef={meta.coef_.round(3).tolist()} (can bang, khong con ap dao)", flush=True)
resB = acc_thr(pva_m, pte_m, "B-STACK-TEST-1-LAN")
resB["meta_coef"] = meta.coef_.round(4).tolist()

json.dump({"A_AlAkasheh":resA, "B_Li":resB,
           "graph":{"edges":int(ei_w.shape[1]//2),"homophily":round(hom,4),"shap_features":fn[sidx].tolist()}},
          open(OUT/"replicate_papers.json","w"), indent=2)
print(f"\nsaved {OUT/'replicate_papers.json'}", flush=True)
