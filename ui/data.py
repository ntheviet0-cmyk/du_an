"""ui/data.py - Lop du lieu dung chung (cache): tabular, model, risk list, bao cao."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parent.parent
REP = REPO / "reports" / "ibm"
ART = REPO / "models" / "ibm"
SEED = 42

CAT = ["BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime"]


@st.cache_resource
def load_tabular():
    df = pd.read_csv(REPO / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv")
    df["Attrition"] = (df["Attrition"] == "Yes").astype(int)
    df = df.drop(columns=["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"])
    NUM = [c for c in df.columns if c not in CAT + ["Attrition"]]
    pre = joblib.load(ART / "preprocessor.joblib")
    return df, pre, CAT, NUM


@st.cache_resource
def load_meta():
    return joblib.load(ART / "meta.joblib")


@st.cache_data
def load_results():
    return json.loads((REP / "ibm_results.json").read_text(encoding="utf-8"))


@st.cache_data
def get_test_frame():
    """Hang test (seed 42) + nhan that. Dung cho RiskList/Executive."""
    from sklearn.model_selection import train_test_split
    df, _, _, _ = load_tabular()
    y = df["Attrition"].values
    idx = np.arange(len(df))
    _, idx_te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    frame = df.iloc[idx_te].copy().reset_index(drop=True)
    frame["_test_pos"] = range(len(frame))
    return frame


@st.cache_data
def score_test_graph():
    """Cham toan bo test bang GraphSAGE best-tune (~vai giay)."""
    import sys
    sys.path.insert(0, str(REPO))
    from graph_shap_lib import get_bundle, graph_predict_proba
    df, pre, cat, num = load_tabular()
    frame = get_test_frame()
    Xt = pre.transform(frame[cat + num])
    Xt = Xt.toarray().astype(np.float32) if hasattr(Xt, "toarray") else np.asarray(Xt, np.float32)
    return graph_predict_proba(Xt)


@st.cache_data
def build_risk_list():
    """Bang risk: ho so + proba + band. Nguon duy nhat cho RiskList/Executive."""
    from ui.components import band_of
    meta = load_meta()
    thr = float(meta.get("thr_graph", 0.8))
    frame = get_test_frame().copy()
    frame["proba"] = score_test_graph()
    frame["band"] = [band_of(p, thr) for p in frame["proba"]]
    med = frame.groupby("Department")["MonthlyIncome"].median().to_dict()
    frame["_med_income_dept"] = frame["Department"].map(med)
    return frame.sort_values("proba", ascending=False).reset_index(drop=True)


@st.cache_data
def dept_income_median():
    df, _, _, _ = load_tabular()
    return df.groupby("Department")["MonthlyIncome"].median().to_dict()
