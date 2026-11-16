"""Render elbow/validation curves and an explicit selection report."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd

from select_model import summarize, select_one_se, knee_candidate

HERE = Path(__file__).resolve().parent
GREEN = "#067647"
BLUE = "#175cd3"
SMOKE = False


def save_figure(fig, path):
    if SMOKE:
        fig.text(0.99, 0.995, "SMOKE TEST • 3 epochs • NOT tuning evidence", ha="right", va="top",
                 fontsize=8, color="#b42318", bbox={"facecolor": "#fff1f3", "edgecolor": "none", "pad": 3})
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def curve(frame, x_column, title, xlabel, path, knee=False, cost=False):
    frame = frame.sort_values(x_column)
    x = frame[x_column].to_numpy()
    y = frame.mean_ap.to_numpy()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(x, y, yerr=frame.se_ap.fillna(0), color=GREEN, marker="o", capsize=4, label="Mean validation AP +/- fold SE")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Average precision (higher is better)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(alpha=0.2)
    notes = []
    if knee:
        detected = knee_candidate(x, y)
        if detected:
            ax.axvline(detected["x"], linestyle="--", color=BLUE, label=f"Knee candidate: {detected['x']:g}")
            notes.append("Knee is a geometric heuristic, not proof of optimality.")
        else:
            notes.append("No clear monotone knee detected; do not force an elbow.")
    if len(frame) and frame.n_folds.min() >= 2:
        chosen, best, cutoff = select_one_se(frame)
        ax.scatter([chosen[x_column]], [chosen.mean_ap], s=130, facecolors="none", edgecolors="#d97706", linewidth=2, label="One-SE candidate")
        ax.axhline(cutoff, color="#d97706", linestyle=":", alpha=0.6)
        notes.append(f"Best AP={best.mean_ap:.4f}; one-SE candidate={chosen.candidate}.")
    if cost:
        right = ax.twinx()
        right.plot(x, frame.mean_edges, color="#667085", linestyle="--", marker="x", alpha=0.6)
        right.set_ylabel("Mean undirected training edges", color="#667085")
    if x_column == "parameters":
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticks(x, [f"{int(r.h1)}→{int(r.h2)}\n{int(r.parameters):,}" for r in frame.itertuples()])
    ax.legend(loc="best", fontsize=8)
    fig.text(0.02, 0.02, "\n".join(notes), fontsize=8, color="#475467")
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    save_figure(fig, path)


def heatmap(frame, path):
    pivot = frame.pivot(index="weight_decay", columns="lr", values="mean_ap").sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    image = ax.imshow(pivot.to_numpy(), cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), [f"{v:g}" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [f"{v:g}" for v in pivot.index])
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.4f}", ha="center", va="center", color="black",
                        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2})
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Weight decay")
    ax.set_title("Validation AP: learning rate × weight decay", loc="left", fontweight="bold")
    fig.colorbar(image, ax=ax, label="Mean validation AP")
    fig.tight_layout()
    save_figure(fig, path)


def training_curve(output, runs):
    baseline = runs[runs.stage == "baseline"]
    if baseline.empty:
        return
    run = baseline.iloc[0]
    history = pd.DataFrame(json.loads((output / "history" / f"{run.run_id}.json").read_text(encoding="utf-8")))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(history.epoch, history.train_loss, marker=".", color=BLUE, label="Train weighted BCE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss", color=BLUE)
    second = ax.twinx()
    second.plot(history.epoch, history.val_ap, marker="o", color=GREEN, label="Validation AP")
    second.set_ylabel("Validation AP", color=GREEN)
    ax.axvline(run.best_epoch, linestyle="--", color="#667085")
    ax.set_title(f"Baseline learning curve — fold {int(run.fold)}, seed {int(run.seed)}", loc="left")
    fig.tight_layout()
    save_figure(fig, output / "figures" / "baseline_learning_curve.png")


def main():
    global SMOKE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    SMOKE = args.smoke
    output = HERE / "outputs" / ("smoke" if args.smoke else "full")
    manifest = json.loads((output / "experiment_config.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    runs = pd.read_csv(output / "tables" / "runs.csv")
    summary = summarize(runs)
    summary["complete"] = summary.apply(
        lambda r: r.n_folds == config["folds"] and r.n_runs == config["folds"] * len(config["final_seeds"] if r.stage == "finalists" else config["screen_seeds"]), axis=1)
    summary.to_csv(output / "tables" / "summary.csv", index=False)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    report = ["# GraphSAGE elbow experiment", "", "**SMOKE TEST — not scientific tuning evidence.**" if args.smoke else "Inductive development CV; historical holdout excluded.", "",
              "Validation folds are used for early stopping AND selection. These scores are selection estimates, not an unbiased final performance estimate.",
              "Use nested CV or new external data for an independent performance claim.", "",
              "Fold means average seeds first. Error bars are SE across folds; CV folds overlap in training and SE is descriptive.", "",
              "One-SE complexity order: fewer training edges, then fewer parameters, then lower mean execution time.", ""]
    for stage, frame in summary.groupby("stage"):
        complete = frame[frame.complete]
        report += [f"## {stage}", f"Complete candidates: {len(complete)}/{len(frame)}", ""]
        if complete.empty:
            report.append("Incomplete runs; no selection or curve emitted.")
            continue
        chosen, best, cutoff = select_one_se(complete)
        report += [f"Best mean AP: {best.mean_ap:.5f}; one-SE cutoff: {cutoff:.5f}.",
                   f"One-SE candidate: `{chosen.candidate}` (k={int(chosen.k)}, hidden={int(chosen.h1)}→{int(chosen.h2)}, dropout={chosen.dropout:g}, lr={chosen.lr:g}, wd={chosen.weight_decay:g}).", ""]
        if stage in ("k", "hidden", "dropout"):
            x_column = {"k": "k", "hidden": "parameters", "dropout": "dropout"}[stage]
            curve(complete, x_column, f"GraphSAGE — {stage} screening", {"k": "Neighbors k", "hidden": "Hidden architecture / parameter count (log scale)", "dropout": "Dropout probability"}[stage],
                  figures / {"k": "elbow_k.png", "hidden": "elbow_hidden.png", "dropout": "validation_dropout.png"}[stage], knee=stage != "dropout", cost=stage == "k")
            if stage != "dropout":
                found = knee_candidate(complete.sort_values(x_column)[x_column], complete.sort_values(x_column).mean_ap)
                report.append(f"Geometric knee candidate: {found}" if found else "No clear monotone elbow detected; use validation and complexity instead.")
        if stage == "regularization":
            heatmap(complete, figures / "lr_weight_decay_heatmap.png")
    training_curve(output, runs)
    report += ["", "## Protocol", "Train-only scaler/encoder and training graph; each query attaches bidirectionally to its own train graph copy.",
               "No validation-to-validation edges, no production model replacement, no threshold tuning on historical test cases.",
               "Screening changes one axis around the baseline. Finalists check interactions; this is not an exhaustive global optimum search.",
               "Predictions, source row indices, checkpoints, preprocessing and learning histories are persisted per run."]
    (output / "selection_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Saved outputs/{output.name}/figures and selection_report.md")


if __name__ == "__main__":
    main()
