"""draw_model_metrics.py - Bang chi so 3 model gon (RF/XGB/Graph), doc tu ibm_results.json."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

with open("reports/ibm/ibm_results.json", encoding="utf-8") as f:
    res = json.load(f)["models"]

models = ["Random Forest", "XGBoost", "Graph"]
keys = ["RandomForest", "XGBoost", "Graph"]
metrics = ["ROC-AUC", "PR-AUC", "F1", "Precision", "Recall", "Accuracy"]
mkeys = ["roc_auc", "pr_auc", "f1", "precision", "recall"]
vals = [[round(res[k][mk], 3) for mk in mkeys] for k in keys]
# Accuracy from confusion_matrix [[TN,FP],[FN,TP]]
for i, k in enumerate(keys):
    cm = res[k]["confusion_matrix"]
    acc = (cm[0][0] + cm[1][1]) / res[k]["n_test"]
    vals[i].append(round(acc, 3))

fig, axes = plt.subplots(1, 2, figsize=(17, 7))
fig.patch.set_facecolor("white")

# Left: grouped bar
ax = axes[0]
x = np.arange(len(metrics))
w = 0.24
colors = ["#1f77b4", "#d62728", "#2ca02c"]
for i, (m, c) in enumerate(zip(models, colors)):
    bars = ax.bar(x + (i - 1) * w, vals[i], w, label=m, color=c, edgecolor="black")
    for b, v in zip(bars, vals[i]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Metrics Comparison", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
fig.text(0.5, 0.01, "Note: Accuracy is high on majority class (stay 83.9%) — read with Recall/F1.",
         ha="center", fontsize=9, style="italic", color="#555555")

# Right: clean table (no imbalance/threshold/confusion/legend)
ax2 = axes[1]
ax2.axis("off")
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.text(5, 9.3, "Model Metrics (test = 294)", ha="center", fontsize=14, fontweight="bold")

header = ["Metric"] + models
rows = [[m] + [f"{v:.3f}" for v in col] for m, col in zip(metrics, zip(*vals))]
table_data = [header] + rows
n_rows, n_cols = len(table_data), 4
col_w = [2.4, 2.5, 2.5, 2.5]
row_h = 1.0
start_y = 8.4
for i, row in enumerate(table_data):
    y = start_y - i * row_h
    for j, cell in enumerate(row):
        x0 = sum(col_w[:j])
        fc = "#2c3e50" if i == 0 else ("#f2f4f4" if i % 2 == 0 else "white")
        tc = "white" if i == 0 else "#1a1a1a"
        fw = "bold" if i == 0 or j == 0 else "normal"
        r = mpatches.FancyBboxPatch((x0, y - row_h), col_w[j], row_h,
                                    boxstyle="round,pad=0.03", facecolor=fc,
                                    edgecolor="#bdc3c7", linewidth=1.0)
        ax2.add_patch(r)
        ax2.text(x0 + col_w[j] / 2, y - row_h / 2, cell, ha="center", va="center",
                 fontsize=11, fontweight=fw, color=tc)

plt.tight_layout()
plt.savefig("reports/ibm/model_metrics_slide.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved reports/ibm/model_metrics_slide.png")
