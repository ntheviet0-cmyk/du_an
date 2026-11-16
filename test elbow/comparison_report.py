"""Paired outer-fold results, resource plots, and non-independent legacy snapshots."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comparison_core import METHODS, outer_summary
from select_model import summarize
from run_experiments import candidate_id

LABELS = {"baseline": "Current config (refit)", "grid_ap": "Grid / max inner AP", "elbow_one_se": "Knee + one-SE"}
COLORS = {"baseline": "#667085", "grid_ap": "#175cd3", "elbow_one_se": "#067647"}


def save(fig, path, smoke):
    if smoke:
        fig.text(0.99, 0.99, "SMOKE TEST - NOT scientific evidence", ha="right", va="top", fontsize=9, color="#b42318")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def complete_outer_folds(results, config):
    if results.empty:
        return results.copy()
    if results.result_id.duplicated().any():
        raise ValueError("Duplicate outer results")
    valid = []
    for fold, frame in results.groupby("outer_fold"):
        if set(frame.method) != set(METHODS):
            continue
        if all(set(frame.loc[frame.method == method, "seed"]) == set(config["evaluation_seeds"])
               and len(frame[frame.method == method]) == len(config["evaluation_seeds"]) for method in METHODS):
            valid.append(fold)
    return results[results.outer_fold.isin(valid)].copy()


def build_report(output):
    output = Path(output)
    manifest = json.loads((output / "experiment_config.json").read_text(encoding="utf-8"))
    config, smoke = manifest["config"], manifest["smoke"]
    source = output / "tables" / "outer_fold_results.csv"
    raw = pd.read_csv(source) if source.exists() else pd.DataFrame()
    results = complete_outer_folds(raw, config)
    n_folds = results.outer_fold.nunique() if not results.empty else 0
    lines = ["# Nested CV: grid AP vs knee/one-SE", "",
             "**SMOKE TEST — technical verification only.**" if smoke else "**Method comparison on development data, historical 294-case holdout excluded.**", "",
             f"Completed comparable outer folds: **{n_folds}/{config['outer_folds']}**.",
             "Partial folds are excluded from comparisons; no winner is declared before all folds finish.", "",
             "## Definition of the methods",
             "- baseline: current hyperparameters, re-trained under the same inductive protocol.",
             "- grid_ap: maximum mean inner-CV average precision in the shared candidate pool.",
             "- elbow_one_se: one-SE AP region, optional monotone knee caps on k/parameter count, then smallest graph/parameter count; deterministic candidate-ID tie-break.",
             "- This is NOT an exact replay of the legacy accuracy/top-3/test-inspection workflow. It compares a standardized AP grid selector against a precisely defined knee/one-SE selector.", "",
             "## Fairness and leakage controls",
             "Both selectors see exactly the same candidate pool and inner folds/seeds. Inner jobs are shared, so neither method has a search-budget advantage.",
             "The elbow heuristic here uses max-AP profiles within the shared grid. It does not measure the efficiency of the earlier sequential axis-screening algorithm.",
             "Outer refit epochs are the median inner best epochs. F1 thresholds come only from selected-candidate inner OOF probabilities (averaged across search seeds).",
             "No early stopping, parameter selection, or threshold selection uses outer labels. Outer features are attached independently to train-only graphs.",
             "Average evaluation seeds within each fold first; summarize paired fold values. SD is descriptive, not a significance test.",
             "Whole-dataset insights were previously inspected, so nested CV controls this automated selection process but is not a claim of untouched external validation.", "",
             "## Resource accounting",
             "Search seconds are logical inner-search budgets. Both grid and elbow are charged the SAME complete shared search time; actual work is cached and performed once.",
             "Baseline is charged its inner epoch/threshold-selection jobs. Refit training/inference times are measured separately; identical selected configs reuse outer fits.", ""]
    if not results.empty:
        summary, fold_means = outer_summary(results)
        summary.to_csv(output / "tables" / "method_comparison.csv", index=False)
        fold_means.to_csv(output / "tables" / "fold_seed_averaged.csv", index=False)
        paired = fold_means.pivot(index="outer_fold", columns="method", values="ap")
        paired["elbow_minus_grid"] = paired.elbow_one_se - paired.grid_ap
        paired["elbow_minus_baseline"] = paired.elbow_one_se - paired.baseline
        paired.to_csv(output / "tables" / "paired_ap_differences.csv")
        lines += ["## Results", "", "| Method | Mean AP | SD AP | Mean ROC-AUC | Mean F1 | Mean recall | Mean parameters |", "|---|---:|---:|---:|---:|---:|---:|"]
        for row in summary.itertuples():
            lines.append(f"| {row.method} | {row.mean_ap:.4f} | {row.sd_ap:.4f} | {row.mean_auc:.4f} | {row.mean_f1:.4f} | {row.mean_recall:.4f} | {row.mean_parameters:.0f} |")
        lines += ["", f"Mean paired AP difference (elbow - grid): {paired.elbow_minus_grid.mean():+.5f}.",
                  "No equivalence or superiority claim is made from a small numeric difference alone."]
        if n_folds < config["outer_folds"] or smoke:
            lines += ["", "**Provisional/technical output: do not interpret as the final scientific result.**"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for method in METHODS:
            axes[0].plot(paired.index + 1, paired[method], marker="o", color=COLORS[method], label=LABELS[method])
        axes[0].set(xlabel="Outer fold", ylabel="Outer average precision", title="Paired AP on identical outer folds")
        axes[0].set_xticks(paired.index + 1)
        axes[0].legend(fontsize=8)
        axes[1].bar(paired.index + 1, paired.elbow_minus_grid, color=COLORS["elbow_one_se"])
        axes[1].axhline(0, color="#667085", linewidth=1)
        axes[1].set(xlabel="Outer fold", ylabel="AP(elbow) - AP(grid)", title="Per-fold difference")
        axes[1].set_xticks(paired.index + 1)
        save(fig, output / "figures" / "paired_ap_comparison.png", smoke)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for row in summary.itertuples():
            for ax, x, xlabel in ((axes[0], row.mean_parameters, "Mean model parameter count"),
                                  (axes[1], row.mean_inference_ms_per_employee, "Mean inference ms / employee (includes neighbor search)")):
                ax.errorbar(x, row.mean_ap, yerr=row.sd_ap if np.isfinite(row.sd_ap) else 0,
                            fmt="o", capsize=4, color=COLORS[row.method], label=LABELS[row.method])
                ax.set(xlabel=xlabel, ylabel="Mean outer AP", title="Predictive quality vs complexity/cost")
                ax.grid(alpha=0.2)
        axes[0].legend(fontsize=8)
        save(fig, output / "figures" / "performance_vs_complexity.png", smoke)
    selections = []
    for path in sorted((output / "selections").glob("outer_*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        for plan in doc["plans"]:
            selections.append({key: value for key, value in plan.items() if key != "config"} | plan["config"])
    if selections:
        pd.DataFrame(selections).to_csv(output / "tables" / "selected_configs.csv", index=False)
    (output / "COMPARISON_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Comparison report: {n_folds}/{config['outer_folds']} comparable outer folds ({output.name})", flush=True)


def compare_existing(here, smoke=False):
    """Tier 1: descriptive comparison of existing finalists, no new training."""
    source = Path(here) / "outputs" / ("smoke" if smoke else "full")
    selected_path = source / "selected_candidate.json"
    if not selected_path.exists():
        raise ValueError("No final elbow candidate exists. Complete run_experiments.py --stage all first; baseline-only results cannot compare models.")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    manifest = json.loads((source / "experiment_config.json").read_text(encoding="utf-8"))
    baseline_id = candidate_id(manifest["config"]["baseline"])
    runs = pd.read_csv(source / "tables" / "runs.csv")
    rows = []
    for label, candidate in (("baseline", baseline_id), ("elbow_selected", selected["candidate"])):
        part = runs[(runs.stage == "finalists") & (runs.candidate == candidate)].copy()
        expected = manifest["config"]["folds"] * len(manifest["config"]["final_seeds"])
        if len(part) != expected:
            raise ValueError("Both baseline and selected candidate need complete finalist fold/seed measurements")
        part["stage"] = label
        rows.append(part)
    merged = pd.concat(rows, ignore_index=True)
    output = Path(here) / "outputs" / ("existing_comparison_smoke" if smoke else "existing_comparison")
    output.mkdir(exist_ok=True)
    summary = summarize(merged)
    summary.to_csv(output / "model_comparison.csv", index=False)
    folds = merged.groupby(["stage", "fold"], as_index=False).val_ap.mean().pivot(index="fold", columns="stage", values="val_ap")
    folds["selected_minus_baseline"] = folds.elbow_selected - folds.baseline
    folds.to_csv(output / "paired_ap.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ("baseline", "elbow_selected"):
        ax.plot(folds.index + 1, folds[method], marker="o", label=method)
    ax.set(xlabel="Development validation fold", ylabel="AP", title="Existing finalists: descriptive comparison, not independent test")
    ax.legend()
    save(fig, output / "model_comparison.png", smoke)
    text = ["# Existing finalist configuration comparison", "",
            "**SMOKE TEST only.**" if smoke else "Descriptive development-CV comparison.", "",
            "These folds already participated in configuration selection and early stopping. Do not use them to claim independent superiority of elbow over grid.",
            "Use compare_methods.py --mode nested for independent outer-fold selection evaluation.", "",
            f"Mean paired AP difference (selected - baseline): {folds.selected_minus_baseline.mean():+.5f}.", "",
            "Both rows use the SAME finalist folds and seeds. No comparison to historical transductive test metrics is made."]
    (output / "COMPARISON_REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"Existing finalist snapshot saved ({output.name})")
