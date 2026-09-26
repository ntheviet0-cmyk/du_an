import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Data
models = ["Random Forest", "XGBoost", "LabelSpreading"]
roc_auc = [0.799, 0.770, 0.691]
pr_auc = [0.494, 0.513, 0.367]
f1 = [0.496, 0.451, 0.394]
recall = [0.681, 0.340, 0.532]
precision = [0.390, 0.667, 0.313]

colors = ["#1f77b4", "#d62728", "#2ca02c"]

# Panel 1: ROC-AUC, PR-AUC, F1 grouped bar
ax1 = axes[0]
x = np.arange(len(models))
w = 0.25
bars1 = ax1.bar(x - w, roc_auc, w, label="ROC-AUC", color="#1f77b4", edgecolor="black")
bars2 = ax1.bar(x, pr_auc, w, label="PR-AUC", color="#ff7f0e", edgecolor="black")
bars3 = ax1.bar(x + w, f1, w, label="F1", color="#9467bd", edgecolor="black")

for bars in [bars1, bars2, bars3]:
    for b in bars:
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                 f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax1.set_ylabel("Score", fontsize=11)
ax1.set_title("ROC-AUC / PR-AUC / F1", fontsize=13, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=10)
ax1.set_ylim(0, 0.95)
ax1.legend(fontsize=9)
ax1.grid(axis="y", alpha=0.3)
ax1.set_facecolor("#fafafa")

# Highlight best
ax1.annotate("Best ROC-AUC", xy=(0, 0.799), xytext=(0.3, 0.92),
             arrowprops=dict(arrowstyle="->", color="blue", lw=1.5),
             color="blue", fontsize=9, fontweight="bold")
ax1.annotate("Best PR-AUC", xy=(1, 0.513), xytext=(1.3, 0.65),
             arrowprops=dict(arrowstyle="->", color="orange", lw=1.5),
             color="orange", fontsize=9, fontweight="bold")
ax1.annotate("Best F1", xy=(0, 0.496), xytext=(-0.3, 0.65),
             arrowprops=dict(arrowstyle="->", color="purple", lw=1.5),
             color="purple", fontsize=9, fontweight="bold")

# Panel 2: Precision-Recall comparison
ax2 = axes[1]
x = np.arange(len(models))
w = 0.25
bars4 = ax2.bar(x - w, precision, w, label="Precision", color="#27ae60", edgecolor="black")
bars5 = ax2.bar(x, recall, w, label="Recall", color="#e74c3c", edgecolor="black")

for bars in [bars4, bars5]:
    for b in bars:
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                 f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax2.set_ylabel("Score", fontsize=11)
ax2.set_title("Precision vs Recall", fontsize=13, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=10)
ax2.set_ylim(0, 0.85)
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3)
ax2.set_facecolor("#fafafa")

# Add labels for each model's characteristic
ax2.text(0, 0.72, "High Recall\n(low threshold)", ha="center", fontsize=8, color="blue", style="italic")
ax2.text(1, 0.72, "High Precision\n(high threshold)", ha="center", fontsize=8, color="red", style="italic")
ax2.text(2, 0.72, "Balanced\n(medium threshold)", ha="center", fontsize=8, color="green", style="italic")

plt.suptitle("Model Comparison: Class Imbalance Handling\n(RF: SMOTE | XGB: SMOTE + scale_pos_weight | Graph: SMOTE)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(r"C:/Users/Public/model_comparison.png", dpi=150)
plt.close()
print("Done")
