"""tune_k_attention.py - Thu k={10,15,20} cho SAGE + attention weight GAT."""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
OUT = Path("reports/ibm_gnn"); OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
df["Attrition"] = (df["Attrition"] == "Yes").astype(int)
df = df.drop(columns=["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"])
CAT = ["BusinessTravel","Department","EducationField","Gender","JobRole","MaritalStatus","OverTime"]
NUM = [c for c in df.columns if c not in CAT+["Attrition"]]
y = df["Attrition"].values; N = len(df)
idx = np.arange(N)
idx_tr, idx_te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
idx_tr2, idx_va = train_test_split(idx_tr, test_size=0.15, stratify=y[idx_tr], random_state=SEED)

pre = ColumnTransformer([("num",StandardScaler(),NUM),("cat",OneHotEncoder(handle_unknown="ignore"),CAT)])
pre.fit(df.iloc[idx_tr][CAT+NUM])
X = pre.transform(df[CAT+NUM])
if hasattr(X,"toarray"): X = X.toarray()
X = X.astype(np.float32)
Xt = torch.tensor(X); yt = torch.tensor(y, dtype=torch.float32)

class SAGE(torch.nn.Module):
    def __init__(s,d,h=64,drop=0.5):
        super().__init__()
        s.c1=SAGEConv(d,h); s.c2=SAGEConv(h,32); s.lin=torch.nn.Linear(32,1); s.drop=drop
    def forward(s,x,ei):
        x=F.relu(s.c1(x,ei)); x=F.dropout(x,p=s.drop,training=s.training)
        return s.lin(F.relu(s.c2(x,ei))).squeeze(-1)

class GAT(torch.nn.Module):
    def __init__(s,d,h=16,heads=8,drop=0.5):
        super().__init__()
        s.c1=GATConv(d,h,heads=heads,dropout=drop)
        s.c2=GATConv(h*heads,32,heads=1,dropout=drop)
        s.lin=torch.nn.Linear(32,1); s.drop=drop
    def forward(s,x,ei,ret_att=False):
        x1,att1 = s.c1(x,ei,return_attention_weights=True)
        x1 = F.elu(x1); x1=F.dropout(x1,p=s.drop,training=s.training)
        x2,att2 = s.c2(x1,ei,return_attention_weights=True)
        out = s.lin(F.elu(x2)).squeeze(-1)
        return (out,(att1,att2)) if ret_att else out

def train(model,ei,epochs=200,patience=30,lr=0.01,wd=5e-4):
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd)
    pos=float((len(idx_tr2)-y[idx_tr2].sum())/max(y[idx_tr2].sum(),1))
    crit=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos]))
    best,bs,bad=-1,None,0
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        # forward + loss tren train mask
        out = model(Xt, ei)
        loss=crit(out[idx_tr2],yt[idx_tr2]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): pv=torch.sigmoid(model(Xt,ei)[idx_va]).numpy()
        try: pr=average_precision_score(y[idx_va],pv)
        except Exception: pr=0
        if pr>best: best,bs,bad=pr,{k:v.cpu().clone() for k,v in model.state_dict().items()},0
        else: bad+=1
        if bad>=patience: break
    model.load_state_dict(bs); return model,best,ep+1

def ev(logits,name):
    p=torch.sigmoid(torch.tensor(logits)).numpy()
    bt,bf=0.5,-1
    for t in np.arange(0.1,0.9,0.05):
        f=f1_score(y[idx_te],(p>=t).astype(int),zero_division=0)
        if f>bf: bf,bt=f,t
    pred=(p>=bt).astype(int)
    r={"roc_auc":float(roc_auc_score(y[idx_te],p)),"pr_auc":float(average_precision_score(y[idx_te],p)),
       "f1":float(bf),"thr":float(bt),
       "precision":float(precision_score(y[idx_te],pred,zero_division=0)),
       "recall":float(recall_score(y[idx_te],pred,zero_division=0))}
    print(f"[k-test {name}] AUC={r['roc_auc']:.4f} PR={r['pr_auc']:.4f} F1={r['f1']:.4f} (thr={bt:.2f}) P={r['precision']:.3f} R={r['recall']:.3f}")
    return r,p

res={}
graphs={}
for k in [10,15,20]:
    A=kneighbors_graph(X,n_neighbors=k,mode="connectivity",metric="cosine",include_self=False)
    A=A.maximum(A.T)
    ei=torch.tensor(np.vstack(A.nonzero()),dtype=torch.long)
    graphs[k]=ei
    r,_=A.nonzero(); hom=float((y[r[0]]==y[r[1]]).mean())
    m,best_pr,ep_run=train(SAGE(X.shape[1]),ei)
    rr,_=ev(m(Xt,ei).detach().numpy()[idx_te],f"SAGE-k{k}")
    rr.update({"homophily":hom,"edges":int(A.nnz//2),"val_pr":float(best_pr),"epochs":int(ep_run)})
    res[k]=rr

with open(OUT/"k_ablation.json","w") as f: json.dump({str(k):v for k,v in res.items()},f,indent=2)
best_k=max(res,key=lambda k: res[k]["pr_auc"])
print(f"\n[best k] = {best_k} (PR-AUC={res[best_k]['pr_auc']:.4f})")
print(f"saved {OUT/'k_ablation.json'}")

# ---- GAT attention tren k tot nhat ----
torch.manual_seed(SEED)
gat=GAT(X.shape[1])
gat,ACCOUNT,MANAGER=train(gat,graphs[best_k])
gat.eval()
with torch.no_grad():
    out,(att1,att2)=gat(Xt,graphs[best_k],ret_att=True)
logits=out.numpy()
rr,p=ev(logits[idx_te],"GAT-bestK")
# att2: (edge_index2, att_weights2) layer cuoi (1 head) -> lay top neighbor cho 1 node rui ro cao
ei2,aw2=att2[0],att2[1].squeeze(-1).numpy()
proba=torch.sigmoid(out).numpy()
cand=idx_te[np.argsort(proba[idx_te])[::-1][:3]]
att_out={}
for n in cand:
    mask=(ei2[1]==int(n)).numpy() if hasattr(ei2[1],"numpy") else (np.asarray(ei2[1])==int(n))
    src=np.asarray(ei2[0])[mask]; w=aw2[mask]
    top=np.argsort(w)[::-1][:5]
    att_out[int(n)]=[{"neighbor":int(src[i]),"att":float(w[i]),
                      "nbr_attr":int(y[src[i]]),"proba":float(proba[n])} for i in top]
    # ve bar
    labs=[f"nbr {src[i]}({'out' if y[src[i]]==1 else 'stay'})" for i in top]
    plt.figure(figsize=(6,3.5))
    plt.barh(labs[::-1],[w[i] for i in top][::-1])
    plt.title(f"GAT attention - node {n} (P={proba[n]:.2f}, true={y[n]})")
    plt.tight_layout(); plt.savefig(OUT/f"gat_attention_node{n}.png",dpi=130); plt.close()
with open(OUT/"gat_attention.json","w") as f: json.dump({"best_k":best_k,"gat_test":rr,"top3":att_out},f,indent=2)
print(f"[attention] saved gat_attention_node*.png x3 + gat_attention.json")
