"""draw_graph_steps.py - Step diagram Pipeline A theo style Image 1 (paper IEEE):
vien den mong, nen trang, o luoi xanh la, thanh tim, tieu de navy.
Khong badge tron. Output: reports/ibm/graph_model_steps.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(9, 13))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 10)
ax.set_ylim(5.6, 16.8)
ax.axis("off")

NAVY = "#1a5276"
PURPLE = "#7d3c98"
GREEN_EC = "#27ae60"
GREEN_FC = "#d5f5e3"

def hbox(x, y, w, h, title, sub=None):
    r = mpatches.FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02",
        facecolor="white", edgecolor="black", linewidth=1.0)
    ax.add_patch(r)
    ax.text(x, y + (0.13 if sub else 0), title, ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="black")
    if sub:
        ax.text(x, y - 0.24, sub, ha="center", va="center", fontsize=6.5, color="#333333")

def arrow(x1, y1, x2, y2, color="black", ls="-", lw=1.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=2))

def grid(x, y, w, h, cols=4):
    r = mpatches.Rectangle((x - w / 2, y - h / 2), w, h, facecolor=GREEN_FC,
                           edgecolor=GREEN_EC, linewidth=0.9)
    ax.add_patch(r)
    ax.plot([x - w / 2, x + w / 2], [y, y], color=GREEN_EC, lw=0.8)
    for j in range(1, cols):
        xx = x - w / 2 + j * w / cols
        ax.plot([xx, xx], [y - h / 2, y + h / 2], color=GREEN_EC, lw=0.8)

def step(x, y, w, h, label, hint=None, body_y=None):
    """Hop step style Image 1: nhan 'Step N' + title 1 dong, hint xam duoi."""
    r = mpatches.FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02",
        facecolor="white", edgecolor="black", linewidth=1.0)
    ax.add_patch(r)
    ax.text(x, y + h / 2 - 0.25, label, ha="center", va="center",
            fontsize=7.5, fontweight="bold", color=NAVY)
    if hint:
        ax.text(x, (body_y if body_y is not None else y - 0.25), hint, ha="center",
                va="center", fontsize=6.5, color="#333333")

CX, W = 5.0, 6.4

ax.text(CX, 16.2, "Pipeline A:  Graph  +  GCN  +  SVM", fontsize=12,
        fontweight="bold", ha="center", color=NAVY)

# ---- head ----
hbox(CX, 15.35, W, 0.7, "HR table", "employee records")
hbox(CX, 14.4, W, 0.8, "Preprocessing", "StandardScaler  +  OneHotEncoder")
r = mpatches.FancyBboxPatch((CX - W / 2, 13.5 - 0.3), W, 0.6, boxstyle="round,pad=0.02",
    facecolor="#d7bde2", edgecolor=PURPLE, linewidth=1.0)
ax.add_patch(r)
ax.text(CX, 13.5, "Feature matrix  (N x 51)", ha="center", va="center",
        fontsize=7.5, fontweight="bold", color="black")
arrow(CX, 15.0, CX, 14.8)
arrow(CX, 14.0, CX, 13.8)
arrow(CX, 13.2, CX, 12.95)

# ---- Step 1 ----
step(CX, 12.35, W, 0.95, "Step 1:  SHAP selection", "keep important features")
# ---- Step 2 ----
step(CX, 10.9, W, 1.5, "Step 2:  kNN graph")
gx, gy = 3.7, 10.6
for sx, sy, lw, col in [(gx - 0.7, gy + 0.3, 1.2, "#1f77b4"), (gx + 0.7, gy + 0.3, 0.9, "#1f77b4"),
                        (gx - 0.7, gy - 0.3, 0.7, "#d62728"), (gx + 0.7, gy - 0.3, 1.0, "#1f77b4")]:
    ax.plot([gx, sx], [gy, sy], color="#777777", lw=lw, alpha=0.8, zorder=2)
    ax.scatter([sx], [sy], s=55, c=col, edgecolors="black", linewidths=0.5, zorder=3)
ax.scatter([gx], [gy], s=90, c="#d62728", edgecolors="black", linewidths=0.6, zorder=3)
ax.text(6.4, 10.75, "node = employee", fontsize=6.5, ha="center")
ax.text(6.4, 10.45, "edge = similarity", fontsize=6.5, ha="center")
# ---- Step 3 ----
step(CX, 9.35, W, 1.1, "Step 3:  GCN")
grid(3.4, 9.52, 1.3, 0.28)
grid(3.4, 9.26, 1.6, 0.28)
grid(3.4, 9.0, 0.9, 0.28)
ax.text(6.3, 9.15, "neighbors -> embedding", fontsize=6.5, ha="center", color="#333333")
# ---- Step 4 ----
step(CX, 8.1, W, 1.0, "Step 4:  Concat", "original features  +  embedding")
# ---- Step 5 ----
step(CX, 6.85, W, 1.2, "Step 5:  Linear SVM")
ax.plot([2.5, 4.0], [6.95, 6.5], color=NAVY, lw=1.2)
ax.text(2.35, 7.0, "Stay", fontsize=6, color=NAVY)
ax.text(3.95, 6.4, "Leave", fontsize=6, color="#922b21")
ax.text(6.2, 6.75, "P(Attrition) -> Stay / Leave", fontsize=6.5, ha="center", color="#333333")

# mui ten chinh
arrow(CX, 13.2, CX, 12.92)
for y1, y2 in [(11.85, 11.68), (10.12, 9.93), (8.77, 8.63), (7.57, 7.48)]:
    arrow(7.3, y1, 7.3, y2)

# ---- nhanh dut: features goc -> concat ----
ax.plot([8.7, 8.7], [8.1, 13.45], color=PURPLE, lw=1.2, linestyle="--")
ax.annotate("", xy=(8.2, 8.1), xytext=(8.7, 8.1),
    arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.2, linestyle="--", shrinkA=0, shrinkB=1))
ax.annotate("", xy=(8.7, 13.45), xytext=(8.2, 13.5),
    arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.2, linestyle="--", shrinkA=0, shrinkB=1))
ax.text(8.9, 10.7, "original features", fontsize=6.5, ha="left", va="center",
        color=PURPLE, rotation=90)

plt.savefig("reports/ibm/graph_model_steps.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved reports/ibm/graph_model_steps.png")
