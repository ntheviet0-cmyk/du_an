"""Nested CV: shared grid-AP search vs knee-assisted one-SE selection and baseline.

Outer labels are consumed only by evaluate_predictions after refit_predict.
Existing production/tuning artifacts are never overwritten.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch

from data_pipeline import DATA_PATH, load_data, make_splits
from run_experiments import atomic_json, candidate_id, digest, train_run
from select_model import PARAMS, summarize
from comparison_core import (candidate_pool, nested_splits, choose_methods, choose_threshold,
                             refit_predict, evaluate_predictions, METHODS)
from comparison_report import build_report, compare_existing

HERE = Path(__file__).resolve().parent


class JobLimitReached(Exception):
    pass


def persist_csv(path, frame):
    temporary = path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def completed_job(path):
    marker = path / "result.json"
    if not marker.exists():
        return None
    for name in ("model.pt", "preprocessor.joblib", "predictions.csv"):
        if not (path / name).exists():
            raise RuntimeError(f"Completed job is missing {name}: {path.name}")
    return json.loads(marker.read_text(encoding="utf-8"))


def nested_comparison(args):
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.smoke:
        config.update(outer_folds=2, inner_folds=2, search_seeds=[42], evaluation_seeds=[42],
                      max_epochs=2, eval_every=1, patience=2)
        # Deliberately tiny pool for end-to-end verification, never tuning evidence.
        pool = [dict(config["baseline"]), dict(config["baseline"], k=5, h1=16, h2=8, dropout=0.5)]
    else:
        pool = candidate_pool(config)
    for field in ("outer_folds", "inner_folds", "max_epochs", "patience", "eval_every", "inference_batch_size", "torch_threads"):
        if not isinstance(config[field], int) or config[field] < (2 if field.endswith("folds") else 1):
            raise ValueError(f"Invalid {field}")
    if config["eval_every"] > config["max_epochs"]:
        raise ValueError("eval_every must not exceed max_epochs")
    for field in ("search_seeds", "evaluation_seeds"):
        if not config[field] or len(set(config[field])) != len(config[field]):
            raise ValueError(f"Invalid seeds: {field}")
    torch.set_num_threads(config["torch_threads"])
    torch.use_deterministic_algorithms(True)
    output = HERE / "outputs" / ("comparison_smoke" if args.smoke else "comparison")
    for directory in ("jobs", "tables", "figures", "selections"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    source_names = ["compare_methods.py", "comparison_core.py", "run_experiments.py", "model.py", "data_pipeline.py", "select_model.py"]
    manifest = dict(config=config, pool=pool, smoke=args.smoke,
                    dataset_sha256=hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
                    sources={name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in source_names},
                    python=platform.python_version(), torch=torch.__version__, sklearn=sklearn.__version__,
                    protocol="nested_shared_grid_vs_elbow_one_se_v1")
    manifest_path = output / "experiment_config.json"
    if manifest_path.exists() and digest(json.loads(manifest_path.read_text(encoding="utf-8"))) != digest(manifest):
        raise RuntimeError("Comparison fingerprint changed. Archive comparison output before a new experiment; existing tuning outputs are separate.")
    atomic_json(manifest_path, manifest)
    raw, frame, y, numeric = load_data()
    development, holdout, _ = make_splits(y, config["split_seed"], config["outer_folds"])
    splits = nested_splits(development, y, config["outer_folds"], config["inner_folds"], config["split_seed"])
    atomic_json(output / "split_manifest.json", {
        "index_base": 0, "historical_holdout_excluded": holdout.tolist(),
        "outer": [{"fold": f, "train": tr.tolist(), "validation": va.tolist(),
                   "inner": [{"train": a.tolist(), "validation": b.tolist()} for a, b in inner]}
                  for f, (tr, va, inner) in enumerate(splits)]})
    pool_map = {candidate_id(cfg): cfg for cfg in pool}
    outer_path = output / "tables" / "outer_fold_results.csv"
    results = pd.read_csv(outer_path) if outer_path.exists() else pd.DataFrame()
    completed_outer = set(results.result_id) if not results.empty else set()
    new_jobs = 0

    def count_job():
        nonlocal new_jobs
        new_jobs += 1
        if args.limit and new_jobs >= args.limit:
            raise JobLimitReached()

    try:
        for outer_fold, (outer_train, outer_val, inner) in enumerate(splits):
            inner_rows, inner_predictions = [], {}
            print(f"[outer {outer_fold + 1}/{len(splits)}] {len(pool)} candidates x {len(inner)} inner folds x {len(config['search_seeds'])} seeds", flush=True)
            for cfg in pool:
                candidate = candidate_id(cfg)
                inner_predictions[candidate] = []
                for fold, (tr, va) in enumerate(inner):
                    assert not set(tr) & set(outer_val) and not set(va) & set(outer_val)
                    for seed in config["search_seeds"]:
                        run_id = f"inner_o{outer_fold}_{candidate}_f{fold}_s{seed}"
                        path = output / "jobs" / run_id
                        result = completed_job(path)
                        if result is None:
                            result, checkpoint, pre, probability, history = train_run(cfg, config, frame, numeric, y, tr, va, seed)
                            path.mkdir(exist_ok=True)
                            torch.save(checkpoint, path / "model.pt")
                            joblib.dump(pre, path / "preprocessor.joblib")
                            persist_csv(path / "predictions.csv", pd.DataFrame({"source_index": va, "actual": y[va], "probability": probability}))
                            atomic_json(path / "history.json", history)
                            # Commit marker last: interrupted jobs are safely recomputed.
                            atomic_json(path / "result.json", result)
                            print(f"  {run_id}: AP={result['val_ap']:.4f}", flush=True)
                            count_job()
                        inner_rows.append(dict(stage="inner", candidate=candidate, fold=fold, seed=seed, run_id=run_id, **cfg, **result))
                        inner_predictions[candidate].append(pd.read_csv(path / "predictions.csv"))
            inner_runs = pd.DataFrame(inner_rows)
            summary = summarize(inner_runs)
            expected_runs = len(inner) * len(config["search_seeds"])
            assert summary.n_runs.eq(expected_runs).all() and len(summary) == len(pool)
            persist_csv(output / "tables" / f"inner_runs_o{outer_fold}.csv", inner_runs)
            persist_csv(output / "tables" / f"inner_summary_o{outer_fold}.csv", summary)
            choices, notes = choose_methods(summary, candidate_id(config["baseline"]))
            plans = []
            for method, candidate in choices.items():
                rows = inner_runs[inner_runs.candidate == candidate]
                epoch = max(1, int(np.floor(rows.best_epoch.median() + 0.5)))
                threshold = choose_threshold(pd.concat(inner_predictions[candidate], ignore_index=True))
                plans.append(dict(method=method, outer_fold=outer_fold, candidate=candidate,
                                  config=pool_map[candidate], epochs=epoch, threshold=threshold,
                                  search_seconds=float(rows.seconds.sum()) if method == "baseline" else float(inner_runs.seconds.sum()),
                                  inner_mean_ap=float(summary.loc[summary.candidate == candidate, "mean_ap"].iloc[0])))
            # Persist choices BEFORE outer inference; no outer label is used here.
            atomic_json(output / "selections" / f"outer_{outer_fold}.json", {"plans": plans, "knee_notes": notes})
            for plan in plans:
                for seed in config["evaluation_seeds"]:
                    result_id = f"{plan['method']}_o{outer_fold}_s{seed}"
                    if result_id in completed_outer:
                        continue
                    run_id = f"outer_o{outer_fold}_{plan['candidate']}_s{seed}_e{plan['epochs']}"
                    path = output / "jobs" / run_id
                    info = completed_job(path)
                    fresh = info is None
                    if fresh:
                        probability, info, checkpoint, pre = refit_predict(
                            plan["config"], config, frame, numeric, outer_train, outer_val,
                            y[outer_train], seed, plan["epochs"])
                        path.mkdir(exist_ok=True)
                        torch.save(checkpoint, path / "model.pt")
                        joblib.dump(pre, path / "preprocessor.joblib")
                        persist_csv(path / "predictions.csv", pd.DataFrame({"source_index": outer_val,
                                    "EmployeeNumber": raw.iloc[outer_val].EmployeeNumber.to_numpy(), "probability": probability}))
                        atomic_json(path / "result.json", info)
                    predicted = pd.read_csv(path / "predictions.csv")
                    assert np.array_equal(predicted.source_index.to_numpy(), outer_val)
                    metrics = evaluate_predictions(y[outer_val], predicted.probability.to_numpy(), plan["threshold"])
                    row = dict(result_id=result_id, method=plan["method"], outer_fold=outer_fold, seed=seed,
                               candidate=plan["candidate"], search_seconds=plan["search_seconds"], **plan["config"], **info, **metrics)
                    results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
                    persist_csv(outer_path, results)
                    completed_outer.add(result_id)
                    print(f"  [{plan['method']}] AP={metrics['ap']:.4f} F1={metrics['f1']:.4f} threshold={plan['threshold']:.2f}", flush=True)
                    if fresh:
                        count_job()
    except JobLimitReached:
        print("Job limit reached; saved progress. Re-run the same command to resume.", flush=True)
    build_report(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "comparison_config.json")
    parser.add_argument("--mode", choices=["nested", "existing"], default="nested")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--limit", type=int, help="Maximum new inner/outer training jobs; same command resumes")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.mode == "existing":
        compare_existing(HERE, args.smoke)
    elif args.report_only:
        build_report(HERE / "outputs" / ("comparison_smoke" if args.smoke else "comparison"))
    else:
        nested_comparison(args)


if __name__ == "__main__":
    main()
