import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BLUE = "#1f77b4"
GREEN = "#27ae60"
RED = "#e74c3c"
ORANGE = "#e67e22"
GRAY = "#7f8c8d"
LIGHT_BLUE = "#d6eaf8"
LIGHT_GREEN = "#d5f5e3"
LIGHT_RED = "#fadbd8"

fig = plt.figure(figsize=(8, 6))
fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 8)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("Label Propagation Process (kNN Graph)", fontsize=14, fontweight="bold", pad=5)

steps = [
    (4, 9.0, ["Step 1: Raw Data", "(1,470 employees, 17 features)", "Mixed: numeric + categorical"], LIGHT_BLUE, "black"),
    (4, 7.2, ["Step 2: Preprocessing", "StandardScaler + OneHotEncoder", "→ 51-dim feature matrix"], ORANGE, "black"),
    (4, 5.4, ["Step 3: Build kNN Graph (k=15)", "Nodes = employees", "Edges = similarity (Euclidean)", "Adjacency matrix W"], GREEN, "black"),
    (4, 3.6, ["Step 4: LabelSpreading (α=0.1)", "Semi-supervised learning", "Harmonic function", "min fᵀLf s.t. f_labelled=y"], RED, "white"),
    (4, 1.8, ["Step 5: Prediction", "P(Attrition=1)", "Threshold 0.70", "Binary: Yes/No"], "#8e44ad", "white"),
]

w, h = 5.5, 1.2
for x, y, lines, color, tc in steps:
    rect = mpatches.FancyBboxPatch((x-w/2, y-h/2), w, h,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color, edgecolor="black", linewidth=1.3)
    ax.add_patch(rect)
    n = len(lines)
    for i, line in enumerate(lines):
        offset = (n - 1) * 0.16 - i * 0.16
        ax.text(x, y + offset, line, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=tc)

for i in range(4):
    y1 = steps[i][1] - h/2 - 0.05
    y2 = steps[i+1][1] + h/2 + 0.05
    ax.annotate("", xy=(4, y2), xytext=(4, y1),
                  arrowprops=dict(arrowstyle="->", color="black", lw=2))
    ax.text(4.15, (y1+y2)/2, "→", ha="center", va="center", fontsize=13, fontweight="bold")

plt.savefig(r"C:/Users/Public/graph_architecture.png", dpi=150)
plt.close()
print("Done")
