import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(figsize=(14, 8))
ax.axis("off")
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)

# Colors
header_color = "#2c3e50"
row_colors = ["#ffffff", "#f8f9fa", "#ffffff"]
highlight = "#e8f5e9"

# Title
ax.text(7, 9.5, "Model Comparison: Class Imbalance Handling",
        ha="center", fontsize=16, fontweight="bold", color="#2c3e50")
ax.text(7, 9.0, "IBM HR Analytics Employee Attrition Dataset (n=1,470, test=294, positive rate=16.1%)",
        ha="center", fontsize=9, color="#7f8c8d", style="italic")

# Data
metrics = [
    ["Model", "Random Forest", "XGBoost", "Graph (LabelSpreading)"],
    ["Imbalance method", "SMOTE", "SMOTE + scale_pos_weight (5.1)", "SMOTE"],
    ["ROC-AUC", "0.799", "0.770", "0.691"],
    ["PR-AUC", "0.494", "0.513", "0.367"],
    ["F1", "0.496", "0.451", "0.394"],
    ["Precision", "0.390", "0.667", "0.313"],
    ["Recall", "0.681", "0.340", "0.532"],
    ["Best Threshold", "0.25", "0.45", "0.70"],
    ["TP / FP / FN / TN", "32/50/15/197", "16/8/31/239", "25/55/22/192"],
]

# Table layout
n_rows = len(metrics)
n_cols = 4
row_h = 0.75
start_y = 8.2
col_w = [3.0, 3.5, 3.5, 4.0]

for i, row_data in enumerate(metrics):
    y = start_y - i * row_h
    for j, cell in enumerate(row_data):
        x = sum(col_w[:j])
        if i == 0:
            color = header_color
            text_color = "white"
            fontsize = 10
            fontweight = "bold"
        elif i == 1:
            color = "#ecf0f1"
            text_color = "#2c3e50"
            fontsize = 8.5
            fontweight = "normal"
        else:
            if i == 3:  # ROC-AUC
                color = highlight if j > 0 else "#f0f0f0"
            elif i == 2:
                color = highlight if j == 1 else row_colors[i % 2]
            else:
                color = row_colors[i % 2]
            text_color = "#2c3e50"
            fontsize = 9.5
            fontweight = "normal"
            if i == 3 and j > 0:
                text_color = "#1b5e20"
                fontweight = "bold"
            if i == 6:  # Recall row - highlight
                color = "#fff3e0" if j > 0 else "#f0f0f0"

        # Draw cell
        rect = mpatches.FancyBboxPatch((x, y-row_h), col_w[j], row_h,
                                    boxstyle="round,pad=0.05",
                                    facecolor=color, edgecolor="#bdc3c7", linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x + col_w[j]/2, y - row_h/2, str(cell), ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight, color=text_color)

# Legend for highlight
ax.text(0.5, 0.3, "Green = Best metric | Orange = Best Recall", ha="left", fontsize=9,
        color="#2c3e50", style="italic")

plt.savefig(r"C:/Users/Public/model_comparison_table.png", dpi=150)
plt.close()
print("Done")
