"""draw_evidence.py - Hinh MINH CHUNG 2-panel:
Trai: 2D that (Income x Tenure) - 2 lop chong lan, duong thang bat luc.
Phai: phan phoi decision scores cua Pipeline A (83-dim) tren test - tach ro 2 cum.
Chung minh: can khong gian 83-dim + GCN moi tach duoc Stay/Leave.
Output: reports/ibm/svm_evidence.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from gnn_models import GCN as GCN_A, feat_index, build_weighted_graph

R = 42
df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
df["Attrition"] = (df["Attrition"] == "Yes").astype(int)
df = df.drop(columns=["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"])
y = df["Attrition"].values
idx = np.arange(len(df))
idx_tr, idx_te = train_test_split(idx, test_size=0.2, stratify=y, random_state=R)

# ---------- Panel trai: 2D that + LinearSVC 2D ----------
X2 = df[["MonthlyIncome", "YearsAtCompany"]].values.astype(float)
X2s = StandardScaler().fit_transform(X2)
svm2 = LinearSVC(C=1.0, random_state=R, max_iter=8000).fit(X2s, y)
w, b = svm2.coef_[0], svm2.intercept_[0]
acc2d = float(svm2.score(X2s, y))

# ---------- Panel phai: Pipeline A decision scores tren test ----------
pre = joblib.load("models/ibm/preprocessor.joblib")
g = joblib.load("models/ibm/graph.joblib")
CAT = ["BusinessTravel","Department","EducationField","Gender","JobRole","MaritalStatus","OverTime"]
NUM = [c for c in df.columns if c not in CAT + ["Attrition"]]
Xa = pre.transform(df[CAT + NUM])
if hasattr(Xa, "toarray"):
    Xa = Xa.toarray()
Xa = np.asarray(Xa, dtype=np.float32)
fn = list(pre.get_feature_names_out())
sidx = feat_index(fn)
ei, ew = build_weighted_graph(Xa, sidx, k=15)
gm = GCN_A(int(g["in_dim"]))
gm.load_state_dict({k: v for k, v in g["gcn_state"].items()})
gm.eval()
with torch.no_grad():
    _, emb = gm(torch.tensor(Xa), ei, ew, ret_emb=True)
Z = np.hstack([Xa, emb.numpy()])
dec = g["svm"].decision_function(Z)
d_te, y_te = dec[idx_te], y[idx_te]
auc83 = float(roc_auc_score(y_te, d_te))
print(f"[evidence] 2D-acc={acc2d:.4f} | 83dim-AUC={auc83:.4f}", flush=True)

# ---------- Ve ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))
fig.patch.set_facecolor("white")

# Trai
stay, leave = y == 0, y == 1
ax1.scatter(X2s[stay, 0], X2s[stay, 1], s=14, c="#1f77b4", alpha=0.5, edgecolors="none", label="Stay")
ax1.scatter(X2s[leave, 0], X2s[leave, 1], s=20, c="#d62728", alpha=0.75, edgecolors="black",
            linewidths=0.4, label="Leave")
ax1.set_xlim(-1.6, 3.0)
ax1.set_ylim(-1.5, 5.8)
ax1.set_xlabel("MonthlyIncome (std)")
ax1.set_ylabel("YearsAtCompany (std)")
ax1.set_title(f"2D that: 2 lop chong lan\nduong thang bat luc (acc={acc2d:.2f})",
              fontsize=11, fontweight="bold")
ax1.legend(fontsize=8, loc="upper right")

# Phai
ax2.hist(d_te[y_te == 0], bins=25, color="#1f77b4", alpha=0.6, label="Stay (test)")
ax2.hist(d_te[y_te == 1], bins=25, color="#d62728", alpha=0.7, label="Leave (test)")
ax2.axvline(0, color="black", lw=1.6, linestyle="--", label="SVM threshold")
ax2.set_xlabel("Pipeline A decision score (83-dim)")
ax2.set_ylabel("So nhan vien (test)")
ax2.set_title(f"83-dim: 2 cum tach ro\nAUC={auc83:.3f} (bang chung can GCN+SVM)",
              fontsize=11, fontweight="bold")
ax2.legend(fontsize=8)

fig.suptitle("Minh chung: 2 chieu khong tach duoc - can khong gian 83-dim",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("reports/ibm/svm_evidence.png", dpi=160, facecolor="white", bbox_inches="tight")
plt.close()
print("Saved reports/ibm/svm_evidence.png")
