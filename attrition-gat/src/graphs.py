"""Graph builders + met do homophily (Giai doan 2).

Quy tac inductive (bat bien 3):
- Moi ham nhan X_pool (chi node train/val) + pool_idx; query/test chi nhan canh TU pool.
- Trong chan doan Giai doan 2: pool = outer-train, danh gia leave-one-out trong pool
  (moi node i lay lang gieng tu pool tru chinh no). Khong cham test.
- Bo self-match. edge_attr = do tuong dong cosine.
- G2 (Gower) HOAN sang Giai doan 3 — chua do o day.

Do thi:
- G1: kNN cosine, k in {5,10,15,20,30}
- G3: cung JobRole + JobLevel (cot goc, khong can fit), thua hoa: trong nhom cung
     role+level lay toi da k lang gieng gan nhat theo cosine (co the it hon k)
- G4: null — giu nguyen out-degree/in-degree cua G1 bang hoan vi co seed
  (Giai doan 3: them G4u null ngau nhien deu, G0 self-loop, G2 Gower o gower.py).

Quy uoc canh PyG (Giai doan 3+):
- nbr_idx[i] = danh sach lang gieng ma node i tong hop -> edge (nbr -> i),
  tuc edge_index = (src=nbr, dst=i). Message chay tu pool vao node dich.
- Suy luan inductive: giu nguyen canh pool-pool; them canh (pool_nbr -> test),
  KHONG co canh test->pool / test-test / self ngoai G0.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

K_LIST = [5, 10, 15, 20, 30]


def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    """S[i,j] = cosine(X[i],X[j]); cheo chinh = -inf (bo self-match)."""
    Xn = X.astype(np.float64)
    n = Xn / np.linalg.norm(Xn, axis=1, keepdims=True).clip(min=1e-12)
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)
    return S


def topk_from_sim(S: np.ndarray, k: int):
    """Tra (idx (N,k), w (N,k)) — top-k theo S, bo self (da -inf)."""
    N = S.shape[0]
    k = min(k, N - 1)
    part = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
    row = np.arange(N)[:, None]
    order = np.argsort(-S[row, part], axis=1)
    idx = part[row, order]
    w = S[row, idx]
    return idx.astype(int), w.astype(float)


def build_g1(X_pool: np.ndarray, k: int):
    S = cosine_sim_matrix(X_pool)
    return topk_from_sim(S, k)


def build_g3(X_pool: np.ndarray, role: np.ndarray, level: np.ndarray, k: int):
    """Cung JobRole + JobLevel; trong nhom lay toi da k gan nhat (cosine).
    Node khong co ban cung nhom -> hang toan -1 (co lap). Pad -1, w=0."""
    S = cosine_sim_matrix(X_pool)
    N = len(X_pool)
    k = min(k, N - 1)
    idx = np.full((N, k), -1, dtype=int)
    w = np.zeros((N, k), dtype=float)
    same = (role[:, None] == role[None, :]) & (level[:, None] == level[None, :])
    np.fill_diagonal(same, False)
    for i in range(N):
        cand = np.where(same[i])[0]
        if len(cand) == 0:
            continue
        take = cand[np.argsort(-S[i, cand])[:k]]
        idx[i, :len(take)] = take
        w[i, :len(take)] = S[i, take]
    return idx, w


def build_g4_from_g1(g1_idx: np.ndarray, seed: int = 42, max_rounds: int = 200):
    """Null giu bac: hoan vi vector hoa tren dst, giu nguyen out-degree (src co dinh)
    va in-degree (cung multiset dst). Sua self-loop + trung canh bang swap bao toan.
    g1_idx: (N,k) khong padding (G1 day du). Tra (N,k) da xao."""
    rng = np.random.default_rng(seed)
    N, k = g1_idx.shape
    src = np.repeat(np.arange(N), k)
    dst0 = g1_idx.reshape(-1).astype(int).copy()
    d2 = rng.permutation(dst0)
    E = len(src)
    for _ in range(max_rounds):
        codes = src * N + d2
        _, inv, counts = np.unique(codes, return_inverse=True, return_counts=True)
        bad = (d2 == src) | (counts[inv] > 1)
        if not bad.any():
            break
        bi = np.where(bad)[0]
        partners = rng.choice(E, size=min(len(bi), E), replace=False)
        involved = np.union1d(bi, partners)
        # hoan vi gia tri trong involved -> song anh tong the, giu multiset dst
        d2[involved] = d2[rng.permutation(involved)]
    return d2.reshape(N, k).astype(int)


def homophily_metrics(nbr_idx: np.ndarray, y_pool: np.ndarray) -> dict:
    """LOO trong pool. nbr_idx co the pad -1 (G3)."""
    y = np.asarray(y_pool).astype(int)
    N = len(y)
    P = int(y.sum())
    valid = nbr_idx >= 0
    deg = valid.sum(axis=1)
    # diem kNN-label: ti le lang gieng duong; node co lap -> lay nen pool
    bg_all = P / N
    scores = np.full(N, bg_all)
    nbr_pos_rate = np.full(N, np.nan)
    for i in range(N):
        nb = nbr_idx[i][valid[i]]
        if len(nb):
            r = float(y[nb].mean())
            scores[i] = r
            nbr_pos_rate[i] = r
    pos_mask = y == 1
    neg_mask = y == 0
    p1 = float(np.nanmean(nbr_pos_rate[pos_mask]))
    p0 = float(np.nanmean(nbr_pos_rate[neg_mask]))
    bg_excl_self_pos = (P - 1) / (N - 1)  # nen tru chinh node (cho node duong)
    lift = float(p1 / bg_excl_self_pos) if bg_excl_self_pos > 0 else float("nan")
    # ti le node duong khong co lang gieng duong (chi tinh node co >=1 lang gieng)
    has_nbr = deg > 0
    pos_with_nbr = pos_mask & has_nbr
    frac_pos_no_pos_nbr = float((((nbr_pos_rate == 0) & pos_with_nbr).sum() / max(pos_with_nbr.sum(), 1)))
    # edge homophily tren canh co huong hop le
    agree, total = 0, 0
    for i in range(N):
        nb = nbr_idx[i][valid[i]]
        if len(nb):
            agree += int((y[nb] == y[i]).sum())
            total += len(nb)
    edge_hom = float(agree / total) if total else float("nan")
    return {
        "P1": p1,
        "P0": p0,
        "background": float(bg_excl_self_pos),
        "lift": lift,
        "frac_pos_no_pos_nbr": frac_pos_no_pos_nbr,
        "edge_homophily": edge_hom,
        "knn_pr_auc": float(average_precision_score(y, scores)),
        "knn_roc_auc": float(roc_auc_score(y, scores)),
        "mean_degree": float(deg.mean()),
        "n_isolated": int((deg == 0).sum()),
        "pool_n": int(N),
        "pool_pos": int(P),
    }


# ---------------- Giai doan 3: G0 / G4u / lap rap PyG inductive ----------------

def build_g0(n_pool: int):
    """Doi chung 'khong do thi': khong canh tuong minh; self-loop do conv tu them
    (add_self_loops=True) cho moi node. Tra mang rong (N,0)."""
    return (np.zeros((n_pool, 0), dtype=int), np.zeros((n_pool, 0), dtype=float))


def build_g4u(n_pool: int, k: int, seed: int):
    """Null ngau nhien deu: moi node chon k lang gieng deu trong pool (tru chinh no)."""
    rng = np.random.default_rng(seed)
    k = min(k, n_pool - 1)
    idx = np.zeros((n_pool, k), dtype=int)
    for i in range(n_pool):
        cand = np.delete(np.arange(n_pool), i)
        idx[i] = rng.choice(cand, size=k, replace=False)
    return idx, np.ones((n_pool, k), dtype=float)


def nbr_to_edge_index(nbr_idx: np.ndarray, nbr_w: np.ndarray | None = None):
    """nbr_idx (N,k) [co the pad -1] -> edge_index (2,E) voi src=nbr, dst=i;
    edge_attr = w tuong ung. Bo padding."""
    src, dst, w = [], [], []
    N = nbr_idx.shape[0]
    for i in range(N):
        for j in range(nbr_idx.shape[1]):
            v = int(nbr_idx[i, j])
            if v < 0:
                continue
            src.append(v)
            dst.append(i)
            w.append(float(nbr_w[i, j]) if nbr_w is not None else 1.0)
    import torch
    ei = torch.tensor([src, dst], dtype=torch.long)
    ea = torch.tensor(w, dtype=torch.float32).unsqueeze(-1)  # (E,1) cho edge_dim=1
    return ei, ea


def attach_test_pool_to_test(pool_nbr_idx_for_test: np.ndarray,
                             pool_nbr_w_for_test: np.ndarray | None,
                             n_pool: int):
    """Canh (pool_nbr -> test_node t): test node danh so n_pool + t.
    Tra edge_index/edge_attr phan test (noi vao canh pool-pool khi suy luan)."""
    import torch
    src, dst, w = [], [], []
    for t in range(pool_nbr_idx_for_test.shape[0]):
        for j in range(pool_nbr_idx_for_test.shape[1]):
            v = int(pool_nbr_idx_for_test[t, j])
            if v < 0:
                continue
            src.append(v)
            dst.append(n_pool + t)
            w.append(float(pool_nbr_w_for_test[t, j]) if pool_nbr_w_for_test is not None else 1.0)
    ei = torch.tensor([src, dst], dtype=torch.long)
    ea = torch.tensor(w, dtype=torch.float32).unsqueeze(-1)
    return ei, ea


def cosine_query_topk(X_q: np.ndarray, X_p: np.ndarray, k: int):
    """Moi query lay top-k trong pool (cosine). Tra (idx vao pool, sim)."""
    Q = X_q.astype(np.float64)
    P = X_p.astype(np.float64)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True).clip(min=1e-12)
    P /= np.linalg.norm(P, axis=1, keepdims=True).clip(min=1e-12)
    S = Q @ P.T
    k = min(k, P.shape[0])
    part = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
    row = np.arange(Q.shape[0])[:, None]
    order = np.argsort(-S[row, part], axis=1)
    idx = part[row, order].astype(int)
    return idx, S[row, idx].astype(float)


def g3_query_topk(X_q: np.ndarray, X_p: np.ndarray, role_q: np.ndarray, role_p: np.ndarray,
                  level_q: np.ndarray, level_p: np.ndarray, k: int):
    """Test node lay toi da k pool node cung role+level gan nhat (cosine). Pad -1."""
    Q = X_q.astype(np.float64)
    P = X_p.astype(np.float64)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True).clip(min=1e-12)
    P /= np.linalg.norm(P, axis=1, keepdims=True).clip(min=1e-12)
    S = Q @ P.T
    Nq = Q.shape[0]
    idx = np.full((Nq, k), -1, dtype=int)
    w = np.zeros((Nq, k), dtype=float)
    for t in range(Nq):
        cand = np.where((role_p == role_q[t]) & (level_p == level_q[t]))[0]
        if len(cand) == 0:
            continue
        take = cand[np.argsort(-S[t, cand])[:k]]
        idx[t, :len(take)] = take
        w[t, :len(take)] = S[t, take]
    return idx, w


def g4u_query(n_test: int, n_pool: int, k: int, seed: int):
    """Test node noi k pool node ngau nhien deu (seeded). w=1."""
    rng = np.random.default_rng(seed)
    k = min(k, n_pool)
    idx = np.zeros((n_test, k), dtype=int)
    for t in range(n_test):
        idx[t] = rng.choice(n_pool, size=k, replace=False)
    return idx, np.ones((n_test, k), dtype=float)
