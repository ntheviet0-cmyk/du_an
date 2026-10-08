"""draw_graph_model.py - Diagram Pipeline A theo style Image 1 (paper):
Dataset -> Numerical/Categorical -> Feature Matrix -> Graph Construction
-> Employee Similarity Graph G=(V,E) -> GCN -> Concat+SVC -> Attrition Prediction.
Output: reports/ibm/graph_model.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(figsize=(9, 14))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 10)
ax.set_ylim(0, 20.5)
ax.axis("off")

GREEN_EC = "#27ae60"
GREEN_FC = "#d5f5e3"
PURPLE_BG = "#d7bde2"
PURPLE_EC = "#7d3c98"
LIGHT_BLUE = "#d6eaf8"
NAVY = "#1a5276"

def box(x, y, w, h, fc="white", ec="#555555", lw=1.0):
    r = mpatches.FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02",
        facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(r)
    return r

def title(x, y, txt, fs=7.5, color="black"):
    ax.text(x, y, txt, ha="center", va="center", fontsize=fs, fontweight="bold", color=color)

def arrow(x1, y1, x2, y2, lw=1.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color="black", lw=lw, shrinkA=0, shrinkB=2))

def green_grid(x, y, w=1.5, h=0.55, rows=2, cols=5):
    r = mpatches.Rectangle((x - w / 2, y - h / 2), w, h, facecolor=GREEN_FC,
                           edgecolor=GREEN_EC, linewidth=0.9)
    ax.add_patch(r)
    for i in range(1, rows):
        ax.plot([x - w / 2, x + w / 2], [y - h / 2 + i * h / rows] * 2, color=GREEN_EC, lw=0.7)
    for j in range(1, cols):
        xx = x - w / 2 + j * w / cols
        ax.plot([xx, xx], [y - h / 2, y + h / 2], color=GREEN_EC, lw=0.7)

CX = 5.0

# ---------- 1. Dataset ----------
box(CX, 19.3, 3.6, 1.2)
title(CX, 19.65, "IBM HR Analytics Dataset", 7.5)
ax.text(CX, 19.35, "Employee table  (1470 x 31)", ha="center", fontsize=6, color="#333333")
green_grid(CX - 0.35, 18.95, w=1.5, h=0.42)
ax.text(CX + 0.75, 18.95, "...", fontsize=10, ha="center", va="center", color="#333333")

# ---------- 2. Numerical / Categorical ----------
box(3.3, 17.35, 2.9, 1.7)
title(3.3, 17.95, "Numerical Features (24)", 7)
green_grid(3.3, 17.5, w=1.5, h=0.42)
sub = mpatches.FancyBboxPatch((3.3 - 0.9, 16.85), 1.8, 0.35, boxstyle="round,pad=0.02",
      facecolor="white", edgecolor="#555555", linewidth=0.9)
ax.add_patch(sub)
ax.text(3.3, 17.02, "StandardScaler", ha="center", fontsize=6, color="black")
box(6.7, 17.35, 2.9, 1.7)
title(6.7, 17.95, "Categorical Features (7)", 7)
green_grid(6.7, 17.5, w=1.5, h=0.42)
sub = mpatches.FancyBboxPatch((6.7 - 0.9, 16.85), 1.8, 0.35, boxstyle="round,pad=0.02",
      facecolor="white", edgecolor="#555555", linewidth=0.9)
ax.add_patch(sub)
ax.text(6.7, 17.02, "OneHot Encoding", ha="center", fontsize=6, color="black")
arrow(4.4, 18.7, 3.3, 18.2)
arrow(5.6, 18.7, 6.7, 18.2)
arrow(3.3, 16.5, 4.2, 16.1)
arrow(6.7, 16.5, 5.8, 16.1)

# ---------- 3. Feature Matrix ----------
box(CX, 15.35, 7.6, 1.1, fc=PURPLE_BG, ec=PURPLE_EC)
title(CX, 15.62, "Feature Matrix  X  (N x F)", 7.5)
green_grid(CX - 0.5, 15.15, w=1.7, h=0.4)
ax.text(CX + 0.8, 15.15, "...", fontsize=10, ha="center", va="center", color="#333333")

# ---------- 4. Graph Construction ----------
arrow(CX, 14.8, CX, 14.45)
box(CX, 13.5, 7.6, 1.7)
title(CX, 14.1, "Graph Construction", 7.5)
ax.text(CX, 13.78, "Cosine Distance  ->  top-15  ->  Weighted Adjacency Matrix  A",
        ha="center", fontsize=6, color="#333333")
green_grid(3.1, 13.15, w=1.2, h=0.4, cols=4)
arrow(3.85, 13.15, 4.25, 13.15)
# mini star
sx, sy = 5.0, 13.15
for px, py, lw in [(sx - 0.55, sy + 0.3, 1.4), (sx + 0.55, sy + 0.3, 1.0),
                   (sx - 0.55, sy - 0.3, 0.8), (sx + 0.55, sy - 0.3, 1.2)]:
    ax.plot([sx, px], [sy, py], color="#777777", lw=lw, zorder=2)
    ax.scatter([px], [py], s=45, c="#1f77b4", edgecolors="black", linewidths=0.5, zorder=3)
ax.scatter([sx], [sy], s=70, c="#d62728", edgecolors="black", linewidths=0.5, zorder=3)
arrow(5.75, 13.15, 6.15, 13.15)
# heatmap A
rng = np.random.RandomState(0)
A = rng.rand(6, 6)
A = (A + A.T) / 2 + np.eye(6) * 0.6
for i in range(6):
    for j in range(6):
        v = A[i, j]
        ax.add_patch(mpatches.Rectangle((6.5 + j * 0.16, 13.42 - i * 0.16 - 0.16), 0.16, 0.16,
                     facecolor=plt.cm.Blues(v), edgecolor="white", linewidth=0.3))
ax.text(7.75, 13.15, "A", ha="center", fontsize=8, fontweight="bold", color="black")

# ---------- 5. Similarity Graph ----------
arrow(CX, 12.65, CX, 12.3)
box(CX, 11.0, 7.6, 2.4)
title(CX, 11.95, "Employee Similarity Graph  G = (V, E)", 7.5)
pts = np.array([[3.4, 11.3], [4.3, 11.5], [5.2, 11.2], [3.1, 10.6], [4.0, 10.5],
                [4.9, 10.6], [3.6, 9.95], [4.6, 9.9], [5.4, 10.1], [4.4, 11.0]])
lab = np.array([0, 0, 1, 0, 1, 0, 0, 0, 1, 0])
edges = [(0, 1), (1, 2), (0, 3), (1, 4), (2, 5), (3, 4), (4, 5), (3, 6),
         (4, 7), (5, 8), (6, 7), (7, 8), (1, 9), (4, 9), (5, 9)]
for a, b in edges:
    ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], color="#999999", lw=0.8, zorder=2)
for (px, py), l in zip(pts, lab):
    ax.scatter([px], [py], s=60, c="#d62728" if l else "#1f77b4",
               edgecolors="black", linewidths=0.5, zorder=3)
ax.scatter([6.5], [10.9], s=60, c="#1f77b4", edgecolors="black", linewidths=0.5)
ax.text(6.75, 10.9, "Employee (node)", ha="left", va="center", fontsize=6)
ax.plot([6.42, 6.58], [10.6, 10.6], color="#999999", lw=1.2)
ax.text(6.75, 10.6, "Similarity (edge)", ha="left", va="center", fontsize=6)

# ---------- 6. GCN ----------
arrow(CX, 9.8, CX, 9.45)
box(CX, 8.85, 7.6, 1.1, fc=PURPLE_BG, ec=PURPLE_EC)
title(CX, 9.05, "GCN  ->  Employee Embedding", 7.5)
ax.text(CX, 8.68, "aggregate neighbor info  +  concat [ features  +  embedding ]",
        ha="center", fontsize=6, color="#333333")

# ---------- 7. SVM ----------
arrow(CX, 8.3, CX, 7.95)
box(CX, 7.4, 7.6, 1.1)
title(CX, 7.6, "Linear SVM", 7.5)
ax.text(CX, 7.22, "classify 83-dim vector", ha="center", fontsize=6, color="#333333")

# ---------- 8. Prediction ----------
arrow(CX, 6.85, CX, 6.5)
box(CX, 5.9, 3.2, 1.1, fc=LIGHT_BLUE, ec=NAVY)
title(CX, 6.1, "Attrition Prediction", 7.5, color=NAVY)
ax.text(CX, 5.72, "(Stay / Leave)", ha="center", fontsize=6.5, color=NAVY)

plt.savefig("reports/ibm/graph_model.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved reports/ibm/graph_model.png")
