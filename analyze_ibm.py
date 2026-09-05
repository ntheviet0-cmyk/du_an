"""analyze_ibm.py - EDA stats + SHAP top features for the detailed report."""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")
R = 42
df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
df["Attrition"] = (df["Attrition"] == "Yes").astype(int)
df = df.drop(columns=["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"])
CAT = ["BusinessTravel","Department","EducationField","Gender","JobRole","MaritalStatus","OverTime"]
NUM = [c for c in df.columns if c not in CAT + ["Attrition"]]

stats = {}
stats["n"] = int(len(df))
stats["attrition_rate"] = float(df["Attrition"].mean())
stats["n_attr"] = int(df["Attrition"].sum())

def rate_by(col):
    g = df.groupby(col)["Attrition"].agg(["mean","count","sum"]).sort_values("mean", ascending=False)
    return {str(k): {"rate": float(v["mean"]), "n": int(v["count"]), "attr": int(v["sum"])} for k,v in g.iterrows()}

stats["by_OverTime"] = rate_by("OverTime")
stats["by_Department"] = rate_by("Department")
stats["by_JobRole"] = rate_by("JobRole")
stats["by_MaritalStatus"] = rate_by("MaritalStatus")
stats["by_JobLevel"] = rate_by("JobLevel")
stats["by_WorkLifeBalance"] = rate_by("WorkLifeBalance")
stats["by_StockOptionLevel"] = rate_by("StockOptionLevel")

# tenure buckets
df["TenureBucket"] = pd.cut(df["YearsAtCompany"], [-1,2,5,10,100], labels=["0-2","3-5","6-10","10+"])
stats["by_TenureBucket"] = rate_by("TenureBucket")
df["IncomeQ"] = pd.qcut(df["MonthlyIncome"], 4, labels=["Q1(low)","Q2","Q3","Q4(high)"])
stats["by_IncomeQ"] = rate_by("IncomeQ")
df["AgeBucket"] = pd.cut(df["Age"], [17,25,35,45,100], labels=["<25","25-35","35-45","45+"])
stats["by_AgeBucket"] = rate_by("AgeBucket")

# ---- models for SHAP ----
X = df[CAT+NUM]; y = df["Attrition"].values
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,stratify=y,random_state=R)
pre = ColumnTransformer([("num",StandardScaler(),NUM),("cat",OneHotEncoder(handle_unknown="ignore"),CAT)])
pre.fit(Xtr); fn = pre.get_feature_names_out()
Xtr_t, Xte_t = pre.transform(Xtr), pre.transform(Xte)
sm = SMOTE(random_state=R)
Xtr_s, ytr_s = sm.fit_resample(Xtr_t, ytr)

rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=1, random_state=R, n_jobs=-1).fit(Xtr_s, ytr_s)
xgbm = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.2, random_state=R, n_jobs=-1, eval_metric="logloss", scale_pos_weight=(len(ytr)-ytr.sum())/ytr.sum()).fit(Xtr_s, ytr_s)

def top_features(model, k=10):
    ex = shap.TreeExplainer(model)
    sv = ex.shap_values(Xte_t)
    if isinstance(sv, list): sv = sv[1]
    imp = np.abs(sv).mean(0)
    idx = np.argsort(imp)[::-1][:k]
    return [{"feature": fn[i], "mean_abs_shap": float(imp[i])} for i in idx]

stats["shap_top_rf"] = top_features(rf)
stats["shap_top_xgb"] = top_features(xgbm)

Path("reports/ibm").mkdir(parents=True, exist_ok=True)
with open("reports/ibm/analyze_stats.json","w") as f:
    json.dump(stats, f, indent=2)
print("saved analyze_stats.json")
print("OverTime:", stats["by_OverTime"])
print("Top SHAP RF:", [d["feature"] for d in stats["shap_top_rf"][:5]])
print("Top SHAP XGB:", [d["feature"] for d in stats["shap_top_xgb"][:5]])
