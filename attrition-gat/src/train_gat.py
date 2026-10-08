"""Vong train GAT co dinh Giai doan 3: full-batch, loss chi tren train,
BCEWithLogitsLoss(pos_weight), AdamW, ReduceLROnPlateau, early-stop val PR-AUC patience 30.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score


def train_gat(model, X_pool, y_pool, edge_index, edge_attr,
              train_mask: np.ndarray, val_mask: np.ndarray,
              lr=3e-3, wd=1e-4, max_epochs=300, patience=30, seed=42,
              device=None, verbose=False):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    X = torch.tensor(X_pool, dtype=torch.float32, device=device)
    y = torch.tensor(y_pool, dtype=torch.float32, device=device)
    ei = edge_index.to(device)
    ea = edge_attr.to(device) if edge_attr is not None else None
    tr = torch.tensor(train_mask, device=device)
    va = torch.tensor(val_mask, device=device)
    n_pos = float(y_pool[train_mask].sum())
    n_neg = float(train_mask.sum() - n_pos)
    pos_w = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=10)
    best_pr, best_state, bad, n_ep = -1.0, None, 0, 0
    for ep in range(max_epochs):
        model.train()
        opt.zero_grad()
        out = model(X, ei, ea)
        loss = crit(out[tr], y[tr])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(X, ei, ea)[va]).cpu().numpy()
        try:
            pr = float(average_precision_score(y_pool[val_mask], pv))
        except Exception:
            pr = 0.0
        sched.step(pr)
        n_ep = ep + 1
        if pr > best_pr:
            best_pr = pr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break
        if verbose and (ep + 1) % 50 == 0:
            print(f"  ep={ep + 1} loss={loss.item():.4f} valPR={pr:.4f} best={best_pr:.4f}", flush=True)
    model.load_state_dict(best_state)
    return model, {"best_val_pr": best_pr, "epochs": n_ep,
                   "pos_weight": float(pos_w.item()), "device": device}


@torch.no_grad()
def predict_proba(model, X_all, edge_index, edge_attr, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    X = torch.tensor(X_all, dtype=torch.float32, device=device)
    out = model(X, edge_index.to(device),
                edge_attr.to(device) if edge_attr is not None else None)
    return torch.sigmoid(out).cpu().numpy()
