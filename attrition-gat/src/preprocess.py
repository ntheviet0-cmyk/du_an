"""Preprocess chuan cho ca 3 mo hinh (RF, XGBoost, GAT).

Quy tac bat bien:
- Moi fit (ordinal mapping co dinh, OHE, scaler, range Gower) chi fit tren train fold.
- Khong SMOTE. Khong Min-Max (dung z-score de tranh cosine tren vector khong am cham diem cao).
- Hai bo feature: base va base+engineered; ca 3 mo hinh dung cung ma tran de so sanh cong bang.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DROP_COLS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
OHE_COLS = ["Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime"]
ORDINAL_COL = "BusinessTravel"
ORDINAL_MAPPING = {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}
LOG1P_COLS = ["MonthlyIncome", "TotalWorkingYears", "YearsAtCompany",
              "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"]
TARGET = "Attrition"


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["IncomePerYearExp"] = df["MonthlyIncome"] / (df["TotalWorkingYears"] + 1.0)
    df["PromotionGap"] = df["YearsAtCompany"] - df["YearsSinceLastPromotion"]
    df["ManagerTenureRatio"] = df["YearsWithCurrManager"] / (df["YearsAtCompany"] + 1.0)
    return df


class AttritionPreprocessor:
    """Fit tren train, transform tren val/test. Luu so chieu thuc te sau encode."""

    def __init__(self, use_engineered: bool = True):
        self.use_engineered = use_engineered
        self.ohe: OneHotEncoder | None = None
        self.scaler: StandardScaler | None = None
        self.num_cols_: list[str] = []
        self.ohe_feature_names_: list[str] = []
        self.n_features_: int = 0

    def _prepare_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
        if self.use_engineered:
            df = add_engineered(df)
        # ordinal co dinh, khong fit
        df[ORDINAL_COL] = df[ORDINAL_COL].map(ORDINAL_MAPPING).astype(float)
        # log1p (khong can fit)
        for c in LOG1P_COLS + (["IncomePerYearExp"] if self.use_engineered else []):
            if c in df.columns:
                df[c] = np.log1p(np.clip(df[c].astype(float), 0, None))
        return df

    def fit(self, df_train: pd.DataFrame) -> "AttritionPreprocessor":
        frame = self._prepare_frame(df_train)
        num = [c for c in frame.columns if c not in OHE_COLS + [TARGET]]
        self.num_cols_ = num
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.ohe.fit(frame[OHE_COLS].astype(str))
        self.ohe_feature_names_ = list(self.ohe.get_feature_names_out(OHE_COLS))
        self.scaler = StandardScaler()
        self.scaler.fit(frame[num].astype(float).values)
        self.n_features_ = len(num) + len(self.ohe_feature_names_)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        assert self.ohe is not None and self.scaler is not None, "chua fit"
        frame = self._prepare_frame(df)
        Xn = self.scaler.transform(frame[self.num_cols_].astype(float).values)
        Xc = self.ohe.transform(frame[OHE_COLS].astype(str))
        return np.hstack([Xn, Xc]).astype(np.float32)

    def feature_names(self) -> list[str]:
        return list(self.num_cols_) + list(self.ohe_feature_names_)


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df
