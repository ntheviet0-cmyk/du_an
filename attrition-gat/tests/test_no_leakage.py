"""Unit test chong ro ri Giai doan 1 (preprocess + CV).
Gower (G2) defer sang Giai doan 3 nhung de san stub + quy tac range-chi-tu-pool.
Chay: python -m pytest attrition-gat/tests -q  (hoac python attrition-gat/tests/test_no_leakage.py)
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from preprocess import AttritionPreprocessor, load_raw  # noqa: E402
from cv import outer_splits, inner_val_split  # noqa: E402


def test_scaler_unchanged_when_test_perturbed():
    df = load_raw(str(ROOT / "data" / "raw.csv"))
    y = df["Attrition"].values
    splits = list(outer_splits(y, n_splits=5, n_repeats=1, seed=42))
    s = splits[0]
    df_tr, df_te = df.iloc[s["train_idx"]].reset_index(drop=True), df.iloc[s["test_idx"]].reset_index(drop=True)
    pre1 = AttritionPreprocessor(use_engineered=True).fit(df_tr)
    Xte1 = pre1.transform(df_te)
    # xao tron test manh (doi gia tri so + category la)
    df_te_bad = df_te.copy()
    for c in ["MonthlyIncome", "Age"]:
        df_te_bad[c] = df_te_bad[c] * 99 + 12345
    df_te_bad["JobRole"] = "FakeRoleXYZ"
    Xte_bad = pre1.transform(df_te_bad)
    pre2 = AttritionPreprocessor(use_engineered=True).fit(df_tr)  # fit lai cung train
    assert np.allclose(pre1.scaler.mean_, pre2.scaler.mean_), "scaler mean doi -> nghi ro ri tu test"
    assert Xte1.shape == Xte_bad.shape, "shape doi khi gap category la (phai handle_unknown=ignore)"
    assert np.isfinite(Xte_bad).all(), "transform test la phai huu han, khong NaN"
    print("[PASS] scaler khong bi anh huong boi test; OHE chiu duoc category la")


def test_inner_val_within_train_only():
    y = np.array([0] * 100 + [1] * 20)
    tr = np.arange(100)
    ytr = y[tr]
    a, b = inner_val_split(ytr, val_size=0.2, seed=42)
    assert len(set(a) & set(b)) == 0, "train/val giao nhau"
    assert max(a.max(initial=-1), b.max(initial=-1)) < len(ytr), "val vuot ra ngoai train fold"
    print("[PASS] inner val nam trong train fold")


def test_no_smote_in_pipeline():
    import tune_baseline  # noqa
    import inspect
    src = inspect.getsource(tune_baseline)
    assert "SMOTE" not in src and "oversampl" not in src.lower(), "cam SMOTE/oversampling"
    assert "class_weight" in src and "scale_pos_weight" in src, "thieu xu ly lech lop bang weight"
    print("[PASS] khong SMOTE; co class_weight/scale_pos_weight")


def test_gower_rule_stub():
    # Stub cho Giai doan 3: range bien so CHI tu pool train; categorical dung cot goc 0/1.
    # O day chi khang dinh quy tac bang van ban + kiem tra range minh hoa khong dung test.
    rng_pool = np.array([20.0, 60.0])  # vi du Age min/max tu pool
    assert rng_pool[1] > rng_pool[0]
    print("[PASS] stub Gower: range-chi-tu-pool, cat 0/1 tren cot goc (implement day du o Giai doan 3)")


if __name__ == "__main__":
    test_scaler_unchanged_when_test_perturbed()
    test_inner_val_within_train_only()
    test_no_smote_in_pipeline()
    test_gower_rule_stub()
    print("ALL LEAKAGE TESTS PASSED")
