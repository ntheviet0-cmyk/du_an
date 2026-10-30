"""export_web_data.py - Xuat MỌI thu web tinh can: weights SAGE goc, X_train, tham so
preprocessor (scaler + one-hot categories), risk list 294 nguoi (cham san), KPI, SHAP tom tat.

Output: web/data/*.js (gan vao window.HR) de mo file:// van chay (khong fetch).
Chu y print ASCII-only (console cp1252).

Dung:
  python export_web_data.py                 # xuat tat ca (can torch)
  python export_web_data.py --only policy   # chi lam moi kpi.js (khong can torch)
  python export_web_data.py --only parity   # chi lam moi parity.js (can torch)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
OUT = REPO / "web" / "data"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42

CAT = ["BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime"]
KEEP = ["Age", "Gender", "MaritalStatus", "DistanceFromHome", "Education", "Department", "JobRole",
        "JobLevel", "BusinessTravel", "MonthlyIncome", "YearsAtCompany", "TotalWorkingYears",
        "NumCompaniesWorked", "OverTime", "WorkLifeBalance", "JobSatisfaction", "EnvironmentSatisfaction",
        "StockOptionLevel", "JobInvolvement", "RelationshipSatisfaction", "TrainingTimesLastYear",
        "EducationField", "DailyRate", "HourlyRate", "MonthlyRate", "PercentSalaryHike", "PerformanceRating",
        "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"]


def wjs(path, name, obj):
    (OUT / path).write_text(f"window.HR = window.HR || {{}};\nwindow.HR.{name} = {json.dumps(obj)};",
                            encoding="utf-8")
    print(f"[js] {path} ({(OUT/path).stat().st_size/1024:.0f} KB)")


def load_frame():
    """Doc CSV, bo cot thua, tra ve (df, y, frame_te, idx_te)."""
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(REPO / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv")
    df["Attrition"] = (df["Attrition"] == "Yes").astype(int)
    df = df.drop(columns=["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"])
    y = df["Attrition"].values
    idx = np.arange(len(df))
    _, idx_te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    frame = df.iloc[idx_te].reset_index(drop=True)
    return df, y, frame


def row_dict(r):
    return {k: (int(r[k]) if isinstance(r[k], (np.integer, bool)) else (float(r[k]) if isinstance(r[k], np.floating) else r[k])) for k in KEEP}


def policy_rates(df):
    """Ty le giam nguy co quan sat duoc cho tung chinh sach (don vi %).
    ot: lay tu ATE nhan qua da kiem dinh trong ibm_results.json (khong tinh o day).
    buddy: chenh lech nhom tham nien <= 2 nam so voi phan con lai.
    pay: chenh lech nhom thu nhap duoi trung vi phong so voi nhom con lai.
    """
    p_ten_new = float(df.loc[df.YearsAtCompany <= 2, "Attrition"].mean())
    p_ten_old = float(df.loc[df.YearsAtCompany > 2, "Attrition"].mean())
    med = df.groupby("Department")["MonthlyIncome"].transform("median")
    p_pay_low = float(df.loc[df.MonthlyIncome < med, "Attrition"].mean())
    p_pay_high = float(df.loc[df.MonthlyIncome >= med, "Attrition"].mean())
    return {
        "ot_obs": round(float(df.loc[df.OverTime == "Yes", "Attrition"].mean() - df.loc[df.OverTime == "No", "Attrition"].mean()) * 100, 1),
        "buddy": round((p_ten_new - p_ten_old) * 100, 1),
        "pay": round((p_pay_low - p_pay_high) * 100, 1),
        "n_buddy_group": int((df.YearsAtCompany <= 2).sum()),
        "n_pay_group": int((df.MonthlyIncome < med).sum()),
    }


def write_kpi():
    """kpi.js: chi can CSV + ibm_results.json (khong can torch)."""
    import joblib
    df, y, _ = load_frame()
    rep = json.loads((REPO / "reports" / "ibm" / "ibm_results.json").read_text(encoding="utf-8"))
    meta = joblib.load(REPO / "models" / "ibm" / "meta.joblib")
    thr = float(meta.get("thr_graph", 0.8))
    pr = policy_rates(df)
    pr["ot"] = round(float(rep["causal"]["ate"]) * 100, 1)
    KPI = {"attrition_rate": round(float(y.mean()) * 100, 1), "n_total": int(len(df)),
           "ate_ot": pr["ot"],
           "auc": round(float(rep["models"]["Graph"]["roc_auc"]), 3),
           "thr": thr, "graph_cfg": "SAGE h64-drop0.3 seed 42009 kNN-cosine k=15",
           "avg_monthly_usd": round(float(df["MonthlyIncome"].mean()), 1),
           "avg_annual_usd": round(float(df["MonthlyIncome"].mean()) * 12, 1),
           "policy_rates": pr}
    wjs("kpi.js", "KPI", KPI)
    return KPI


def write_parity(frame):
    """parity.js: 20 ca test dau + Xt + proba that tu model (can torch)."""
    import joblib
    from graph_shap_lib import graph_predict_proba
    pre = joblib.load(REPO / "models" / "ibm" / "preprocessor.joblib")
    meta = joblib.load(REPO / "models" / "ibm" / "meta.joblib")
    Xt = pre.transform(frame[CAT + [c for c in meta["num"]]])
    Xt = Xt.toarray().astype(np.float32) if hasattr(Xt, "toarray") else np.asarray(Xt, np.float32)
    proba = graph_predict_proba(Xt)
    raw20 = [row_dict(frame.iloc[i]) for i in range(20)]
    wjs("parity.js", "PARITY", {"Xt": Xt[:20].round(5).tolist(),
                                "proba": [round(float(p), 5) for p in proba[:20]],
                                "raw": raw20})


def export_all():
    import joblib
    from graph_shap_lib import get_bundle, graph_predict_proba

    B = get_bundle()
    meta = joblib.load(REPO / "models" / "ibm" / "meta.joblib")
    pre = joblib.load(REPO / "models" / "ibm" / "preprocessor.joblib")
    FEATS = B["feat_names"]
    NUM = [c for c in meta["num"]]

    # ---- 1. weights SAGE (float32 -> round 6) ----
    st = joblib.load(REPO / "models" / "ibm" / "graph.joblib")["sage_state"]
    W = {k: np.asarray(v, dtype=np.float64).round(6).tolist() for k, v in st.items()}
    wjs("weights.js", "W", W)

    # ---- 2. X_train (round 5) ----
    Xtr = np.asarray(B["X_train"], dtype=np.float64).round(5)
    wjs("xtrain.js", "X_TRAIN", Xtr.tolist())

    # ---- 3. edge list train ----
    wjs("edges.js", "EDGES", {"rows": B["edge_rows"], "cols": B["edge_cols"], "n": B["n_train"]})

    # ---- 4. preprocessor params ----
    numt = pre.named_transformers_["num"]
    catt = pre.named_transformers_["cat"]
    df_mode = pd.read_csv(REPO / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv")
    SC = {"means": numt.mean_.round(6).tolist(), "scales": numt.scale_.round(6).tolist(), "num_cols": NUM,
          "cat_cols": CAT, "categories": [list(map(str, c)) for c in catt.categories_],
          "cat_defaults": {c: str(df_mode[c].mode().iloc[0]) for c in CAT},
          "feat_names": FEATS}
    wjs("preproc.js", "PRE", SC)

    # ---- 5. raw data + split + proba + risk ----
    df, y, frame = load_frame()
    Xt = pre.transform(frame[CAT + NUM])
    Xt = Xt.toarray().astype(np.float32) if hasattr(Xt, "toarray") else np.asarray(Xt, np.float32)
    proba = graph_predict_proba(Xt)
    thr = float(meta.get("thr_graph", 0.8))
    def band(p):
        return "red" if p >= thr else ("amber" if p >= thr * 0.6 else "green")
    RISK = []
    for i, r in frame.iterrows():
        d = row_dict(r)
        d["proba"] = round(float(proba[i]), 4)
        d["band"] = band(float(proba[i]))
        d["left"] = int(r["Attrition"])
        RISK.append(d)
    RISK.sort(key=lambda d: -d["proba"])
    wjs("risk.js", "RISK", RISK)

    # ---- 6. KPI (kem policy rates) + SHAP tom tat + tu dien ----
    try:
        write_kpi()
    except Exception as e:
        print(f"[warn] kpi.js chua lam moi duoc ({e}); dung ban cu neu co")

    SHAP = {"grouped": pd.read_csv(REPO / "reports" / "ibm" / "graph_shap_grouped.csv").head(12).to_dict("records"),
            "cohort": pd.read_csv(REPO / "reports" / "ibm" / "graph_shap_cohort.csv").to_dict("records"),
            "cases": pd.read_csv(REPO / "reports" / "ibm" / "graph_shap_cases.csv").to_dict("records")}
    wjs("shap.js", "SHAP", SHAP)
    HRD = json.loads((REPO / "reports" / "ibm" / "hr_dictionary.json").read_text(encoding="utf-8"))
    wjs("vocab.js", "VN", HRD)
    wjs("meta.js", "META", {"feats": FEATS, "cat": CAT, "num": NUM})

    # ---- 7. parity 20 ca ----
    raw20 = [row_dict(frame.iloc[i]) for i in range(20)]
    wjs("parity.js", "PARITY", {"Xt": Xt[:20].round(5).tolist(),
                                "proba": [round(float(p), 5) for p in proba[:20]],
                                "raw": raw20})
    print("[done] web/data san sang")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["all", "policy", "parity"], default="all")
    args = ap.parse_args()
    if args.only == "policy":
        write_kpi()
        print("[done] kpi.js lam moi (khong can torch)")
    elif args.only == "parity":
        _, _, frame = load_frame()
        write_parity(frame)
        print("[done] parity.js lam moi")
    else:
        export_all()
