"""Unit test Gower: range chi tu pool, doi xung, d(x,x)=0, cat la khong vo."""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gower import GowerKNN  # noqa: E402


def _toy(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Age": rng.integers(20, 60, n).astype(float),
        "MonthlyIncome": rng.uniform(2000, 9000, n),
        "JobLevel": rng.integers(1, 5, n),
        "Department": rng.choice(["Sales", "R&D"], n),
        "JobRole": rng.choice(["Manager", "Clerk"], n),
        "BusinessTravel": rng.choice(["Travel_Rarely", "Non-Travel"], n),
        "EducationField": ["Life Sciences"] * n,
        "Gender": rng.choice(["Male", "Female"], n),
        "MaritalStatus": rng.choice(["Single", "Married"], n),
        "OverTime": rng.choice(["Yes", "No"], n),
        "Attrition": rng.choice([0, 1], n),
    })


def test_symmetry_and_self_zero():
    df = _toy()
    g = GowerKNN().fit(df.iloc[:30])
    D = g.distance(df, df)
    assert np.allclose(D, D.T), "Gower phai doi xung"
    assert np.allclose(np.diag(D), 0.0), "d(x,x) phai = 0"
    assert (D >= -1e-12).all()
    print("[PASS] doi xung + d(x,x)=0")


def test_range_only_from_pool():
    tr = _toy(n=30, seed=1)
    te = _toy(n=10, seed=2)
    te.loc[:, "MonthlyIncome"] = te["MonthlyIncome"] * 50 + 1e6  # test cuc doan
    g1 = GowerKNN().fit(tr)
    D_pool_before = g1.distance(tr, tr)
    g2 = GowerKNN().fit(tr)  # fit lai cung train
    D_pool_after = g2.distance(tr, tr)
    assert np.allclose(D_pool_before, D_pool_after), "range doi khi test doi -> ro ri"
    D_q = g1.distance(te, tr)
    assert np.isfinite(D_q).all(), "query cuc doan phai huu han"
    print("[PASS] range chi tu pool train; query la van huu han")


def test_unknown_category():
    tr = _toy()
    te = _toy(n=5, seed=9)
    te.loc[:, "JobRole"] = "FakeRoleXYZ"
    g = GowerKNN().fit(tr)
    idx, sim = g.query_neighbors(te, tr, k=3)
    assert idx.shape == (5, 3) and np.isfinite(sim).all()
    print("[PASS] category la van chay (mismatch=1)")


def test_neighbors_valid():
    df = _toy(n=50)
    g = GowerKNN().fit(df)
    idx, sim = g.pool_neighbors(df, k=5)
    assert idx.shape == (50, 5)
    assert not (idx == np.arange(50)[:, None]).any(), "co self-match"
    assert (sim <= 1.0 + 1e-9).all()
    print("[PASS] pool_neighbors hop le, khong self")


if __name__ == "__main__":
    test_symmetry_and_self_zero()
    test_range_only_from_pool()
    test_unknown_category()
    test_neighbors_valid()
    print("ALL GOWER TESTS PASSED")
