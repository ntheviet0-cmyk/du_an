"""Unit test cho graphs: inductive-safe, khong self-match, G4 giu bac."""
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graphs import build_g1, build_g3, build_g4_from_g1, homophily_metrics  # noqa: E402


def test_g1_no_self_and_in_pool():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 8)).astype(np.float32)
    for k in [5, 15]:
        idx, w = build_g1(X, k)
        assert idx.shape == (50, k)
        assert not (idx == np.arange(50)[:, None]).any(), "co self-match"
        assert (idx >= 0).all() and (idx < 50).all(), "lang gieng ngoai pool"
        assert np.isfinite(w).all()
    print("[PASS] G1: khong self-match, lang gieng trong pool")


def test_g3_same_group_only():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 8)).astype(np.float32)
    role = np.array(["A"] * 20 + ["B"] * 40)
    level = np.array([1] * 10 + [2] * 10 + [1] * 40)
    idx, _ = build_g3(X, role, level, k=5)
    for i in range(60):
        for j in idx[i]:
            if j == -1:
                continue
            assert role[j] == role[i] and level[j] == level[i], "G3 noi sai nhom"
            assert j != i
    print("[PASS] G3: chi noi cung role+level, khong self")


def test_g4_preserves_degrees():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 8)).astype(np.float32)
    g1, _ = build_g1(X, k=10)
    g4 = build_g4_from_g1(g1, seed=123)
    assert g4.shape == g1.shape
    assert not (g4 == np.arange(80)[:, None]).any(), "G4 co self-loop"
    # out-degree giu nguyen (=k); in-degree giu nguyen (swap bao toan)
    in1 = np.bincount(g1.reshape(-1), minlength=80)
    in4 = np.bincount(g4.reshape(-1), minlength=80)
    assert (in1 == in4).all(), "G4 lam doi in-degree"
    assert (g1.reshape(-1) != g4.reshape(-1)).any(), "G4 khong xao gi ca"
    print("[PASS] G4: giu out/in-degree, khong self-loop, da xao")


def test_metrics_no_nan():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(100, 8)).astype(np.float32)
    y = (rng.random(100) < 0.2).astype(int)
    idx, _ = build_g1(X, k=10)
    m = homophily_metrics(idx, y)
    for key in ["P1", "P0", "lift", "knn_pr_auc", "knn_roc_auc", "edge_homophily"]:
        assert np.isfinite(m[key]), f"{key} NaN"
    print("[PASS] metrics huu han:", {k: round(m[k], 3) for k in ["P1", "P0", "lift", "knn_roc_auc"]})


if __name__ == "__main__":
    test_g1_no_self_and_in_pool()
    test_g3_same_group_only()
    test_g4_preserves_degrees()
    test_metrics_no_nan()
    print("ALL GRAPH TESTS PASSED")
