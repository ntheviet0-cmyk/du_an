"""Resumable inductive GraphSAGE tuning in an isolated experiment folder."""
import argparse
import hashlib
import itertools
import json
import os
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from data_pipeline import DATA_PATH, load_data, make_splits, prepare_fold, make_graph, query_neighbors, inductive_predict
from model import GraphSAGE
from select_model import PARAMS, summarize, select_one_se

HERE = Path(__file__).resolve().parent
STAGES = ["baseline", "k", "hidden", "dropout", "regularization", "finalists"]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def candidate_id(cfg):
    return digest(cfg)[:12]


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def complete_summary(runs, config, stage, folds):
    part = runs[runs.stage == stage] if not runs.empty else runs
    if part.empty:
        raise ValueError(f"Stage '{stage}' has no results. Run screening stages first.")
    seed_count = len(config["screen_seeds"])
    good = part.groupby("candidate").filter(lambda g: len(g) == folds * seed_count)
    result = summarize(good)
    expected = {"k": len(config["k_values"]), "hidden": len(config["hidden_values"]),
                "dropout": len(config["dropout_values"]),
                "regularization": len(config["learning_rates"]) * len(config["weight_decays"])}[stage]
    if len(result) != expected:
        raise ValueError(f"Stage '{stage}' incomplete: {len(result)}/{expected} complete candidates. Finish this stage before finalist selection.")
    return result.sort_values(["mean_ap", "candidate"], ascending=[False, True])


def stage_candidates(stage, config, runs, folds):
    baseline = config["baseline"]
    if stage == "baseline":
        return [dict(baseline)]
    if stage == "k":
        return [dict(baseline, k=k) for k in config["k_values"]]
    if stage == "hidden":
        return [dict(baseline, h1=h1, h2=h2) for h1, h2 in config["hidden_values"]]
    if stage == "dropout":
        return [dict(baseline, dropout=d) for d in config["dropout_values"]]
    if stage == "regularization":
        return [dict(baseline, lr=lr, weight_decay=wd) for lr, wd in itertools.product(config["learning_rates"], config["weight_decays"])]
    top = {name: complete_summary(runs, config, name, folds).head(2) for name in STAGES[1:5]}
    candidates = []
    for k, h, d, reg in itertools.product(*(top[name].to_dict("records") for name in STAGES[1:5])):
        candidates.append(dict(baseline, k=int(k["k"]), h1=int(h["h1"]), h2=int(h["h2"]),
                               dropout=float(d["dropout"]), lr=float(reg["lr"]), weight_decay=float(reg["weight_decay"])))
    # Always retain current baseline in the multi-seed comparison.
    return list({candidate_id(c): c for c in candidates + [dict(baseline)]}.values())


def train_run(cfg, options, frame, numeric, y, train_idx, val_idx, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    start = time.perf_counter()
    pre, train, val = prepare_fold(frame, numeric, train_idx, val_idx)
    edges = make_graph(train, cfg["k"])
    neighbors = query_neighbors(train, val, cfg["k"])
    x = torch.from_numpy(train)
    target = torch.tensor(y[train_idx], dtype=torch.float32)
    model = GraphSAGE(train.shape[1], cfg["h1"], cfg["h2"], cfg["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    positives = target.sum().item()
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(target) - positives) / max(positives, 1)))
    best_ap, best_state, best_epoch, bad_epochs = -1, None, 0, 0
    history = []
    for epoch in range(1, options["max_epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(x, edges), target)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch}")
        loss.backward()
        optimizer.step()
        if epoch % options["eval_every"] and epoch != options["max_epochs"]:
            continue
        p = inductive_predict(model, train, edges, val, cfg["k"], options["inference_batch_size"], neighbors)
        ap = float(average_precision_score(y[val_idx], p))
        history.append({"epoch": epoch, "train_loss": float(loss.item()), "val_ap": ap})
        if ap > best_ap:
            best_ap, best_epoch, bad_epochs = ap, epoch, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            bad_epochs = epoch - best_epoch
        if bad_epochs >= options["patience"]:
            break
    model.load_state_dict(best_state)
    probability = inductive_predict(model, train, edges, val, cfg["k"], options["inference_batch_size"], neighbors)
    result = dict(val_ap=float(average_precision_score(y[val_idx], probability)),
                  val_auc=float(roc_auc_score(y[val_idx], probability)),
                  best_epoch=best_epoch, epochs=epoch, parameters=sum(p.numel() for p in model.parameters()),
                  edges=edges.shape[1] // 2, seconds=time.perf_counter() - start,
                  train_nodes=len(train), val_nodes=len(val), in_dim=train.shape[1])
    checkpoint = {"state_dict": best_state, "config": cfg, "in_dim": train.shape[1], "seed": seed,
                  "train_indices": train_idx.tolist(), "val_indices": val_idx.tolist(), "best_epoch": best_epoch}
    return result, checkpoint, pre, probability, history


def validate_config(config):
    for key in ("folds", "max_epochs", "patience", "eval_every", "inference_batch_size", "torch_threads"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"Invalid positive integer: {key}")
    if config["folds"] < 2 or config["eval_every"] > config["max_epochs"]:
        raise ValueError("Need >=2 folds and eval_every <= max_epochs")
    for key in ("screen_seeds", "final_seeds"):
        if not config[key] or len(set(config[key])) != len(config[key]):
            raise ValueError(f"Empty/duplicate seeds: {key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES + ["all"], default="baseline")
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--smoke", action="store_true", help="2 folds / 1 seed / 3 epochs, separate outputs/smoke")
    parser.add_argument("--limit", type=int, help="Run at most this many NEW fold/seed jobs, then stop (resume later)")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.smoke:
        config.update(folds=2, screen_seeds=[42], final_seeds=[42], max_epochs=3, patience=3, eval_every=1)
        config.update(k_values=[5, 15, 30], hidden_values=[[16, 8], [64, 32], [128, 64]],
                      dropout_values=[0.0, 0.3, 0.6], learning_rates=[0.003, 0.01], weight_decays=[0.00001, 0.0005])
    validate_config(config)
    torch.set_num_threads(config["torch_threads"])
    torch.use_deterministic_algorithms(True)
    out = HERE / "outputs" / ("smoke" if args.smoke else "full")
    for name in ("splits", "checkpoints", "tables", "figures", "predictions", "history"):
        (out / name).mkdir(parents=True, exist_ok=True)
    sources = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
               for name in ("model.py", "data_pipeline.py", "run_experiments.py", "select_model.py")}
    experiment = {"config": config, "dataset_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
                  "sources": sources, "python": platform.python_version(), "torch": torch.__version__,
                  "sklearn": sklearn.__version__, "protocol": "inductive_train_only_v1", "smoke": args.smoke}
    manifest = out / "experiment_config.json"
    if manifest.exists() and digest(json.loads(manifest.read_text(encoding="utf-8"))) != digest(experiment):
        raise RuntimeError("Experiment configuration/code/data changed. Archive this output directory before starting a new experiment.")
    atomic_json(manifest, experiment)
    raw, frame, y, numeric = load_data()
    development, historical, splits = make_splits(y, config["split_seed"], config["folds"])
    split_doc = {"index_base": 0, "development": development.tolist(), "historical_holdout": historical.tolist(),
                 "folds": [{"fold": i, "train": tr.tolist(), "validation": va.tolist()} for i, (tr, va) in enumerate(splits)],
                 "historical_note": "Previously inspected benchmark, not a new unbiased test."}
    atomic_json(out / "splits" / "split_manifest.json", split_doc)
    runs_path = out / "tables" / "runs.csv"
    runs = pd.read_csv(runs_path) if runs_path.exists() else pd.DataFrame()
    done = set(runs.run_id) if not runs.empty else set()
    stages = STAGES if args.stage == "all" else [args.stage]
    new_runs = 0
    for stage in stages:
        candidates = stage_candidates(stage, config, runs, config["folds"])
        atomic_json(out / f"candidates_{stage}.json", candidates)
        seeds = config["final_seeds"] if stage == "finalists" else config["screen_seeds"]
        print(f"[{stage}] {len(candidates)} configs x {len(splits)} folds x {len(seeds)} seeds", flush=True)
        for cfg, (fold, (tr, va)), seed in itertools.product(candidates, enumerate(splits), seeds):
            candidate = candidate_id(cfg)
            run_id = f"{stage}_{candidate}_f{fold}_s{seed}"
            if run_id in done:
                continue
            result, checkpoint, pre, proba, history = train_run(cfg, config, frame, numeric, y, tr, va, seed)
            torch.save(checkpoint, out / "checkpoints" / f"{run_id}.pt")
            joblib.dump(pre, out / "checkpoints" / f"{run_id}_preprocessor.joblib")
            pd.DataFrame({"source_index": va, "EmployeeNumber": raw.iloc[va].EmployeeNumber.to_numpy(),
                          "actual": y[va], "probability": proba}).to_csv(out / "predictions" / f"{run_id}.csv", index=False)
            atomic_json(out / "history" / f"{run_id}.json", history)
            row = dict(stage=stage, candidate=candidate, fold=fold, seed=seed, run_id=run_id, **cfg, **result)
            runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True)
            temporary = runs_path.with_suffix(".csv.tmp")
            runs.to_csv(temporary, index=False)
            os.replace(temporary, runs_path)
            summarize(runs).to_csv(out / "tables" / "summary.csv", index=False)
            done.add(run_id)
            new_runs += 1
            print(f"  {run_id}: AP={result['val_ap']:.4f} AUC={result['val_auc']:.4f} epoch={result['best_epoch']} time={result['seconds']:.1f}s", flush=True)
            if args.limit and new_runs >= args.limit:
                print("Job limit reached. Re-run the same command to resume.")
                return
        if stage == "finalists":
            summary = summarize(runs[runs.stage == "finalists"])
            chosen, best, cutoff = select_one_se(summary)
            summary.to_csv(out / "tables" / "finalist_comparison.csv", index=False)
            atomic_json(out / "selected_candidate.json", {"candidate": chosen.candidate,
                        "config": {p: int(chosen[p]) if p in ("k", "h1", "h2") else float(chosen[p]) for p in PARAMS},
                        "best_ap_candidate": best.candidate, "one_se_cutoff": cutoff,
                        "note": "CV-selected candidate, not promoted to production. Same validation folds used for stopping and selection; estimates are optimistic."})
    print("Done. Run draw_results.py with the matching --smoke flag to generate figures.")


if __name__ == "__main__":
    main()
