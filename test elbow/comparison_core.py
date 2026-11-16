"""Pure selection and fixed-epoch outer fitting for fair method comparison."""
import itertools
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                             precision_score, recall_score, confusion_matrix, accuracy_score)
from sklearn.model_selection import StratifiedKFold

from data_pipeline import prepare_fold, make_graph, inductive_predict
from model import GraphSAGE
from run_experiments import candidate_id
from select_model import knee_candidate

METHODS = ["baseline", "grid_ap", "elbow_one_se"]


def candidate_pool(config):
    pool = [dict(k=k, h1=h[0], h2=h[1], dropout=d, lr=lr, weight_decay=wd)
            for k, h, d, lr, wd in itertools.product(config["k_values"], config["hidden_values"],
                                                    config["dropout_values"], config["learning_rates"], config["weight_decays"])]
    return list({candidate_id(c): c for c in pool + [config["baseline"]]}.values())


def nested_splits(development, y, outer_folds, inner_folds, seed):
    outer_cv = StratifiedKFold(outer_folds, shuffle=True, random_state=seed)
    result = []
    for fold, (tr, va) in enumerate(outer_cv.split(development, y[development])):
        outer_train, outer_val = development[tr], development[va]
        inner_cv = StratifiedKFold(inner_folds, shuffle=True, random_state=seed + fold + 1)
        inner = [(outer_train[a], outer_train[b]) for a, b in inner_cv.split(outer_train, y[outer_train])]
        result.append((outer_train, outer_val, inner))
    return result


def choose_methods(summary, baseline_id):
    """Same candidate scores/budget; different, explicitly defined selectors.

    Knee uses max-AP profile at each k/parameter count. If no monotone knee is
    found, use one-SE alone. No forced elbow or access to outer labels/scores.
    """
    if summary.empty or summary.n_folds.min() < 2 or summary.n_folds.nunique() != 1:
        raise ValueError("Need complete matching inner folds for every candidate")
    ranked = summary.sort_values(["mean_ap", "candidate"], ascending=[False, True])
    grid = ranked.iloc[0]
    cutoff = float(grid.mean_ap - grid.se_ap)
    eligible = summary[summary.mean_ap >= cutoff].copy()
    notes = {"cutoff": cutoff, "knee_k": None, "knee_parameters": None, "knee_restriction_applied": False}
    restricted = eligible.copy()
    for column, key in (("k", "knee_k"), ("parameters", "knee_parameters")):
        profile = summary.groupby(column, as_index=False).mean_ap.max().sort_values(column)
        found = knee_candidate(profile[column], profile.mean_ap)
        notes[key] = found
        if found:
            restricted = restricted[restricted[column] <= found["x"]]
    if not restricted.empty:
        notes["knee_restriction_applied"] = any(notes[k] for k in ("knee_k", "knee_parameters"))
        eligible = restricted
    else:
        notes["fallback"] = "Knee caps removed all one-SE candidates; retain global one-SE set."
    # Deterministic tie-break: dropout/lr variants do not imply different model size.
    elbow = eligible.sort_values(["mean_edges", "parameters", "candidate"]).iloc[0]
    if baseline_id not in set(summary.candidate):
        raise ValueError("Baseline must be evaluated in the same pool")
    return {"baseline": baseline_id, "grid_ap": str(grid.candidate), "elbow_one_se": str(elbow.candidate)}, notes


def choose_threshold(oof):
    """F1 threshold from INNER out-of-fold probabilities only; average seeds first."""
    labels = oof.groupby("source_index").actual.nunique()
    if not labels.eq(1).all():
        raise ValueError("Conflicting labels for the same OOF sample")
    pooled = oof.groupby("source_index", as_index=False).agg(actual=("actual", "first"), probability=("probability", "mean"))
    candidates = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scores = [f1_score(pooled.actual, pooled.probability >= t, zero_division=0) for t in candidates]
    # First maximum -> lower threshold on a tie; fixed before outer evaluation.
    return float(candidates[int(np.argmax(scores))])


def evaluate_predictions(y, probability, threshold):
    predicted = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return dict(ap=float(average_precision_score(y, probability)), auc=float(roc_auc_score(y, probability)),
                f1=float(f1_score(y, predicted, zero_division=0)), precision=float(precision_score(y, predicted, zero_division=0)),
                recall=float(recall_score(y, predicted, zero_division=0)), accuracy=float(accuracy_score(y, predicted)),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), threshold=float(threshold), n_eval=len(y))


def refit_predict(cfg, options, frame, numeric, train_idx, eval_idx, train_labels, seed, epochs):
    """Outer refit sees training labels ONLY. Never early-stop on outer validation."""
    if len(train_labels) != len(train_idx):
        raise ValueError("Only labels for outer training rows may be supplied")
    torch.manual_seed(seed)
    np.random.seed(seed)
    pre, train, query = prepare_fold(frame, numeric, train_idx, eval_idx)
    edges = make_graph(train, cfg["k"])
    model = GraphSAGE(train.shape[1], cfg["h1"], cfg["h2"], cfg["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    target = torch.tensor(train_labels, dtype=torch.float32)
    positive = target.sum().item()
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(target) - positive) / max(positive, 1)))
    x = torch.tensor(train)
    start = time.perf_counter()
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(x, edges), target)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite outer train loss")
        loss.backward()
        optimizer.step()
    train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    probability = inductive_predict(model, train, edges, query, cfg["k"], options["inference_batch_size"])
    inference_seconds = time.perf_counter() - start
    model.eval()
    info = dict(train_seconds=train_seconds, inference_seconds=inference_seconds,
                inference_ms_per_employee=1000 * inference_seconds / len(query),
                edges=edges.shape[1] // 2, parameters=sum(p.numel() for p in model.parameters()), epochs=epochs)
    checkpoint = dict(config=cfg, in_dim=train.shape[1], state_dict=model.state_dict(), seed=seed,
                      epochs=epochs, train_indices=train_idx.tolist(), eval_indices=eval_idx.tolist())
    return probability, info, checkpoint, pre


def outer_summary(results):
    # First average repeated seed measurements per fold, then summarize folds.
    metrics = ["ap", "auc", "f1", "precision", "recall", "accuracy", "parameters", "edges",
               "train_seconds", "inference_ms_per_employee", "search_seconds"]
    folds = results.groupby(["method", "outer_fold"], as_index=False)[metrics].mean()
    rows = []
    for method, frame in folds.groupby("method"):
        row = {"method": method, "n_outer_folds": len(frame)}
        for metric in metrics:
            row[f"mean_{metric}"] = float(frame[metric].mean())
            row[f"sd_{metric}"] = float(frame[metric].std()) if len(frame) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows), folds
