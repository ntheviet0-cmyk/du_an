"""explain_graph_shap.py - Tang giai thich SHAP cho GNN frozen (GraphSAGE best-tune).

Nguyen tac: giu nguyen model GNN da train (models/ibm/graph.joblib), khong xay lai do thi,
khong them model du doan thu hai. SHAP chi phan tich dong gop dac trung vao dung
xac suat ma GNN du doan (inductive: gan node moi vao graph train co dinh).

4 tang: (1) Global importance  (2) Feature effects  (3) Interaction (mo ta cap doi)
        (4) Local & cohort (bar chart SHAP + trung binh nhom HR).

Output: reports/ibm/shap_graph_*.png, graph_shap_values.npz, graph_shap_cases.csv,
        graph_shap_cohort.csv, graph_shap_grouped.csv
Note: print ASCII-only (console Windows cp1252).
"""
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from sklearn.model_selection import train_test_split

from gnn_models import SAGE as SAGE_G

REPO = Path(__file__).resolve().parent
ART = REPO / "models" / "ibm"
REP = REPO / "reports" / "ibm"
SEED = 42
N_BACKGROUND = 30
NSAMPLES = 200
N_CASES = 14

# ---------------- 1. load frozen GNN (dung chung via graph_shap_lib) ----------------
from graph_shap_lib import (get_bundle, graph_predict_proba, grouped_shap,
                            plot_grouped_shap)

_B = get_bundle()
FEATS = _B["feat_names"]
THR = _B["thr_graph"]
print(f"[bundle] seed_run={_B.get('seed_run')} thr={THR:.2f} train_nodes={_B['n_train']}")


# ---------------- 2. test data (split seed 42, giong train_ibm) ----------------
df = pd.read_csv(REPO / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv")
df["Attrition"] = (df["Attrition"] == "Yes").astype(int)
df = df.drop(columns=["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"])
CAT = ["BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime"]
NUM = [c for c in df.columns if c not in CAT + ["Attrition"]]
y = df["Attrition"].values
idx = np.arange(len(df))
idx_tr, idx_te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
pre = joblib.load(ART / "preprocessor.joblib")
Xte_t = pre.transform(df.iloc[idx_te][CAT + NUM])
Xte_t = Xte_t.toarray().astype(np.float32) if hasattr(Xte_t, "toarray") else np.asarray(Xte_t, np.float32)
yte = y[idx_te]
raw_te = df.iloc[idx_te].reset_index(drop=True)
print(f"[test] n={len(yte)} pos={int(yte.sum())}")

t0 = time.time()
proba_te = graph_predict_proba(Xte_t)
print(f"[proba] {len(yte)} cases in {time.time()-t0:.1f}s")
pred_te = (proba_te >= THR).astype(int)

# ---------------- 3. chon ca giai thich ----------------
rng = np.random.RandomState(123)
order_hi = np.argsort(proba_te)[::-1]
order_lo = np.argsort(proba_te)
order_bd = np.argsort(np.abs(proba_te - THR))
sel = []
sel += [int(i) for i in order_hi[:6]]
sel += [int(i) for i in order_lo[:4]]
sel += [int(i) for i in order_bd[:4] if int(i) not in sel]
pos_idx = np.where(yte == 1)[0]
have_pos = sum(1 for i in sel if yte[i] == 1)
for i in pos_idx[np.argsort(proba_te[pos_idx])[::-1]]:
    if have_pos >= 4:
        break
    if int(i) not in sel:
        sel.append(int(i))
        have_pos += 1
sel = sel[:N_CASES]
print(f"[cases] {len(sel)} selected, pos_included={sum(int(yte[i]) for i in sel)}")

# ---------------- 4. background + KernelSHAP ----------------
pos_tr = np.where(y[idx_tr] == 1)[0]
neg_tr = np.where(y[idx_tr] == 0)[0]
n_pos_bg = max(4, int(round(N_BACKGROUND * y[idx_tr].mean())))
bg_idx = np.concatenate([
    rng.choice(pos_tr, size=n_pos_bg, replace=False),
    rng.choice(neg_tr, size=N_BACKGROUND - n_pos_bg, replace=False),
])
Xtr_t = pre.transform(df.iloc[idx_tr][CAT + NUM])
Xtr_t = Xtr_t.toarray().astype(np.float32) if hasattr(Xtr_t, "toarray") else np.asarray(Xtr_t, np.float32)
background = Xtr_t[rng.permutation(bg_idx)][:N_BACKGROUND]
print(f"[background] {len(background)} (pos={n_pos_bg})")

explainer = shap.KernelExplainer(graph_predict_proba, background)
EV = float(np.atleast_1d(explainer.expected_value)[0])
print(f"[explainer] expected_value={EV:.4f}")

SV = np.zeros((len(sel), Xte_t.shape[1]), dtype=np.float64)
PROBA = np.zeros(len(sel))
ADDERR = np.zeros(len(sel))
DONE = np.zeros(len(sel), dtype=np.int64)
NPZ = REP / "graph_shap_values.npz"
done = set()
if NPZ.exists():
    try:
        _old = np.load(NPZ, allow_pickle=True)
        if (list(_old["sel_idx"]) == list(sel) and _old["shap_values"].shape == SV.shape
                and abs(float(_old["expected_value"]) - EV) < 1e-9):
            SV[:] = _old["shap_values"]
            PROBA[:] = _old["proba"]
            ADDERR[:] = _old["add_err"]
            DONE[:] = _old["done"] if "done" in _old else 0
            if "done" not in _old and float(_old["add_err"].max()) < 1e-9 and len(sel) == _old["shap_values"].shape[0]:
                DONE[:] = 1  # npz cu (chua co truong done) nhung da du 14 ca
                print("[resume] backfill done=1 cho npz cu")
            done = {k for k in range(len(sel)) if DONE[k] == 1}
            print(f"[resume] {len(done)}/{len(sel)} cases san co")
    except Exception as e:
        print(f"[resume] bo qua npz cu: {e}")
for k, i in enumerate(sel):
    if k in done:
        print(f"[case {k+1}/{len(sel)}] da co -> bo qua")
        continue
    t1 = time.time()
    sv = explainer.shap_values(Xte_t[i].reshape(1, -1), nsamples=NSAMPLES)
    sv = np.asarray(sv).reshape(-1)
    if sv.shape[0] != Xte_t.shape[1]:
        sv = np.asarray(sv).reshape(Xte_t.shape[1])
    SV[k] = sv
    PROBA[k] = float(proba_te[i])
    ADDERR[k] = abs((EV + sv.sum()) - PROBA[k])
    DONE[k] = 1
    print(f"[case {k+1}/{len(sel)}] proba={PROBA[k]:.3f} true={yte[i]} add_err={ADDERR[k]:.2e} ({time.time()-t1:.0f}s)")
    np.savez(NPZ, shap_values=SV, expected_value=EV,
             proba=PROBA, add_err=ADDERR, done=DONE, sel_idx=np.array(sel),
             feat_names=np.array(FEATS))
print(f"[verify] max_add_err={ADDERR.max():.2e}")
print("[save] graph_shap_values.npz (incremental)")
if int(DONE.sum()) < len(sel):
    print("[partial] con thieu ca -> chay lai script de tiep tuc (resume), chua ve bieu do")
    raise SystemExit(0)

# ---------------- 5. Tang 1: Global importance ----------------
mean_abs = np.abs(SV).mean(axis=0)
top15 = np.argsort(mean_abs)[::-1][:15]
plt.figure(figsize=(8, 6))
plt.barh(range(15)[::-1], mean_abs[top15][::-1], color="#2b5c8f")
plt.yticks(range(15)[::-1], [FEATS[j] for j in top15[::-1]], fontsize=8)
plt.xlabel("mean(|SHAP|)")
plt.title("GraphSAGE SHAP - Global Importance", fontweight="bold")
plt.tight_layout()
plt.savefig(REP / "shap_graph_global.png", dpi=130)
plt.close()

grouped = {}
for j, f in enumerate(FEATS):
    if f.startswith("cat__"):
        base = f.split("__", 1)[1].split("_")[0]  # BusinessTravel, OverTime, ...
    else:
        base = f.split("__", 1)[1]
    grouped[base] = grouped.get(base, 0.0) + float(mean_abs[j])
grp = sorted(grouped.items(), key=lambda kv: kv[1], reverse=True)
with open(REP / "graph_shap_grouped.csv", "w", encoding="utf-8-sig") as f:
    f.write("feature_group,mean_abs_shap\n")
    for name, val in grp:
        f.write(f"{name},{val:.6f}\n")
print("[layer1] saved shap_graph_global.png + graph_shap_grouped.csv")

# ---------------- 6. Tang 2: Feature effects ----------------
shap.summary_plot(SV, Xte_t[sel], feature_names=FEATS, show=False, max_display=15)
plt.tight_layout()
plt.savefig(REP / "shap_graph_beeswarm.png", dpi=130, bbox_inches="tight")
plt.close()
for j in top15[:3]:
    plt.figure()
    shap.dependence_plot(j, SV, Xte_t[sel], feature_names=FEATS, show=False)
    plt.tight_layout()
    safe = FEATS[j].replace(" ", "").replace("&", "and")[:40]
    plt.savefig(REP / f"shap_graph_dep_{safe}.png", dpi=120, bbox_inches="tight")
    plt.close()
print("[layer2] saved beeswarm + 3 dependence plots")

# ---------------- 7. Tang 3: Interaction (mo ta cap doi) ----------------
pairs = [("cat__OverTime_Yes", "num__MonthlyIncome"),
         ("cat__OverTime_Yes", "num__JobSatisfaction"),
         ("num__MonthlyIncome", "num__JobLevel")]
ji = {f: FEATS.index(f) for f in FEATS}
lines = ["pair,corr_sv,note"]
for a, b in pairs:
    ca = np.corrcoef(SV[:, ji[a]], SV[:, ji[b]])[0, 1]
    lines.append(f"{a} x {b},{ca:.3f},tuong quan SHAP 2 yeu to tren {len(sel)} ca")
mask_ot = Xte_t[sel][:, ji["cat__OverTime_Yes"]] > 0.5
mask_lowinc = Xte_t[sel][:, ji["num__MonthlyIncome"]] < np.median(Xte_t[sel][:, ji["num__MonthlyIncome"]])
both = mask_ot & mask_lowinc
lines.append(f"OT Yes + thu nhap thap,{int(both.sum())}/{len(sel)} ca,mean_proba={PROBA[both].mean() if both.any() else 0:.3f}")
with open(REP / "graph_shap_interaction.csv", "w", encoding="utf-8-sig") as f:
    f.write("metric,value,note\n")
    for ln in lines:
        f.write(ln + "\n")
print("[layer3] saved graph_shap_interaction.csv")

# ---------------- 8. Tang 4: Local SHAP contributions + cohort ----------------
case_rows = ["test_pos,proba,pred,true_label,top_up_1,top_up_2,top_up_3,top_down_1,top_down_2,n_factors,factors_up,factors_down"]
try:
    HR_DICT = json.loads((REP / "hr_dictionary.json").read_text(encoding="utf-8"))
except Exception:
    HR_DICT = {}
for k, i in enumerate(sel):
    gnames, gvals, gdata = grouped_shap(
        SV[k], Xte_t[i], FEATS, raw_row=raw_te.iloc[i])
    fig, ax = plt.subplots(figsize=(9, 5))
    shown_idx = plot_grouped_shap(
        ax, gvals, gnames, gdata, hr=HR_DICT, base_value=EV, prediction=PROBA[k])
    plt.tight_layout()
    plt.savefig(REP / f"shap_graph_waterfall_{k:02d}.png", dpi=120, bbox_inches="tight")
    plt.close()
    up = [FEATS[j] for j in np.argsort(SV[k])[::-1][:3]]
    dn = [FEATS[j] for j in np.argsort(SV[k])[:2]]
    up_all = [gnames[j] for j in np.argsort(gvals)[::-1] if gvals[j] > 0]
    dn_all = [gnames[j] for j in np.argsort(gvals) if gvals[j] < 0]
    case_rows.append(f"{i},{PROBA[k]:.3f},{pred_te[i]},{yte[i]},{up[0]},{up[1]},{up[2]},{dn[0]},{dn[1]},{len(shown_idx)},{';'.join(up_all)},{';'.join(dn_all)}")
    print(f"[local {k:02d}] factors={len(shown_idx)}")
with open(REP / "graph_shap_cases.csv", "w", encoding="utf-8-sig") as f:
    f.write("\n".join(case_rows) + "\n")
print("[layer4] saved local SHAP charts + graph_shap_cases.csv")

cohorts = {
    "Sales": (raw_te["Department"] == "Sales").values,
    "OT_Yes": (raw_te["OverTime"] == "Yes").values,
    "Tenure_0_2": (raw_te["YearsAtCompany"] <= 2).values,
    "Age_u30": (raw_te["Age"] < 30).values,
}
sel_arr = np.array(sel)
out = ["cohort,n,mean_proba,top1,top2,top3"]
for name, mask in cohorts.items():
    sub = sel_arr[mask[sel_arr]]
    if len(sub) == 0:
        out.append(f"{name},0,,,," )
        continue
    m = np.abs(SV[[list(sel).index(int(i)) for i in sub]]).mean(axis=0)
    t3 = [FEATS[j] for j in np.argsort(m)[::-1][:3]]
    out.append(f"{name},{len(sub)},{proba_te[sub].mean():.3f},{t3[0]},{t3[1]},{t3[2]}")
with open(REP / "graph_shap_cohort.csv", "w", encoding="utf-8-sig") as f:
    f.write("\n".join(out) + "\n")
print("[layer4] saved graph_shap_cohort.csv")
print("[done] SHAP 4 tang hoan tat")
