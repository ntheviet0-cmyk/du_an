"""Gower tu viet (Giai doan 3) — khong them dependency.
- Bien so: |xi-xj| / range_f, range CHI tu pool train (fit). Khong clip.
- Bien danh muc: cot GOC (khong one-hot), khoang cach 0/1.
- d doi xung, d(x,x)=0, similarity = 1 - d (co the am nhe neu du lieu moi vuot range).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DROP_COLS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
CAT_COLS = ["Department", "EducationField", "Gender", "JobRole", "MaritalStatus",
            "OverTime", "BusinessTravel"]  # BusinessTravel: coi nhu danh muc (0/1 tren chuoi goc)
TARGET = "Attrition"


class GowerKNN:
    def __init__(self):
        self.num_cols_: list[str] = []
        self.ranges_: dict[str, float] = {}

    def fit(self, df_pool_train: pd.DataFrame) -> "GowerKNN":
        """Chi fit tren pool train. Luu range bien so."""
        df = df_pool_train.drop(columns=[c for c in DROP_COLS + [TARGET] if c in df_pool_train.columns])
        self.num_cols_ = [c for c in df.columns if c not in CAT_COLS]
        for c in self.num_cols_:
            v = pd.to_numeric(df[c], errors="coerce").astype(float).values
            r = float(np.nanmax(v) - np.nanmin(v))
            self.ranges_[c] = r if r > 0 else 1.0  # hang so -> dong gop 0
        return self

    def _num_frame(self, df: pd.DataFrame) -> np.ndarray:
        df = df.drop(columns=[c for c in DROP_COLS + [TARGET] if c in df.columns])
        return np.column_stack([pd.to_numeric(df[c], errors="coerce").astype(float).values
                                for c in self.num_cols_])

    def _cat_frame(self, df: pd.DataFrame) -> np.ndarray:
        return np.column_stack([df[c].astype(str).values for c in CAT_COLS])

    def distance(self, df_a: pd.DataFrame, df_b: pd.DataFrame) -> np.ndarray:
        """Ma tran khoang cach Gower (|A| x |B|)."""
        Xa = self._num_frame(df_a)
        Xb = self._num_frame(df_b)
        Ca = self._cat_frame(df_a)
        Cb = self._cat_frame(df_b)
        n_num, n_cat = len(self.num_cols_), len(CAT_COLS)
        D = np.zeros((len(df_a), len(df_b)), dtype=np.float64)
        for j, c in enumerate(self.num_cols_):
            D += np.abs(Xa[:, j, None] - Xb[:, j][None, :]) / self.ranges_[c]
        for j in range(n_cat):
            D += (Ca[:, j, None] != Cb[:, j][None, :]).astype(np.float64)
        return D / max(n_num + n_cat, 1)

    def pool_neighbors(self, df_pool: pd.DataFrame, k: int):
        """Top-k trong pool tru chinh no. Tra (idx, sim=1-d)."""
        D = self.distance(df_pool, df_pool)
        np.fill_diagonal(D, np.inf)
        N = len(df_pool)
        k = min(k, N - 1)
        part = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
        row = np.arange(N)[:, None]
        order = np.argsort(D[row, part], axis=1)
        idx = part[row, order].astype(int)
        return idx, (1.0 - D[row, idx]).astype(float)

    def query_neighbors(self, df_query: pd.DataFrame, df_pool: pd.DataFrame, k: int):
        """Moi query lay top-k trong pool. Tra (idx vao pool, sim)."""
        D = self.distance(df_query, df_pool)
        Nq, Np = D.shape
        k = min(k, Np)
        part = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
        row = np.arange(Nq)[:, None]
        order = np.argsort(D[row, part], axis=1)
        idx = part[row, order].astype(int)
        return idx, (1.0 - D[row, idx]).astype(float)
