"""draw_graph_paper.py - Diagram paper-style (multi-view -> feature conversion -> graph block) cho mô hình Graph."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(10, 14))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 10)
ax.set_ylim(0, 16.5)
ax.axis("off")

def box(x, y, w, h, text, fontsize=7, fc="white", ec="black", style="round,pad=0.02", fw="normal", ls=1.0):
    r = mpatches.FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle=style,
        facecolor=fc, edgecolor=ec, linewidth=ls)
    ax.add_patch(r)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, fontweight=fw, linespacing=1.3)
    return r

def arrow(x1, y1, x2, y2, lw=1.0, color="black"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=0, shrinkB=2))

def grid_box(x, y, w=1.1, h=0.6, rows=2, cols=3, fc="#d5f5e3", ec="#27ae60"):
    r = mpatches.Rectangle((x-w/2, y-h/2), w, h, facecolor=fc, edgecolor=ec, linewidth=1.0)
    ax.add_patch(r)
    for i in range(1, rows):
        yy = y-h/2 + i*h/rows
        ax.plot([x-w/2, x+w/2], [yy, yy], color=ec, lw=0.8)
    for j in range(1, cols):
        xx = x-w/2 + j*w/cols
        ax.plot([xx, xx], [y-h/2, y+h/2], color=ec, lw=0.8)

# ---- HR dataset only, chain via Employee table ----
box(5.0, 15.2, 2.0, 0.5, "HR_dataset", 7.5, "#ffffff", "black", fw="bold")

# MULTI-VIEW label
box(0.7, 12.6, 1.2, 0.45, "MULTI-VIEW", 7.5, "#d6eaf8", "#2e86c1", fw="bold")

# Employee table centered
ax.text(5.0, 14.45, "Employee table", fontsize=6.5, ha="center", color="black")
# mini grid for table
grid_box(5.0, 13.95, w=1.3, h=0.6, rows=2, cols=3, fc="white", ec="black")
ax.text(4.3, 13.95, "u", fontsize=7, ha="center", va="center")
ax.text(5.0, 14.35, "i", fontsize=7, ha="center", va="center")
ax.text(5.0, 13.45, "(31 features)", fontsize=6, ha="center", va="center")

box(3.8, 12.7, 1.6, 0.7, "Numeric\nfeatures (24)", 7, "white", "black")
ax.text(3.15, 12.17, "Scaler:\nStandard", fontsize=6, ha="center", va="center")
box(6.2, 12.7, 1.6, 0.7, "Categorical\nfeatures (7)", 7, "white", "black")
ax.text(6.85, 12.17, "Encoder:\nOneHot → 27", fontsize=6, ha="center", va="center")

# chain arrows: HR -> Employee table -> 2 streams (origin at grid bottom corners to avoid (31 features) text)
arrow(5.0, 14.95, 5.0, 14.5)
arrow(4.7, 13.65, 3.8, 13.05)
arrow(5.3, 13.65, 6.2, 13.05)

# 2 green vectors (keep N x 24 / N x 27 literally)
grid_box(3.8, 11.7, w=1.6, h=0.55, rows=2, cols=4)
ax.text(3.8, 11.05, "numeric vector\n(N x 24)", fontsize=6, ha="center", va="center")
grid_box(6.2, 11.7, w=1.6, h=0.55, rows=2, cols=4)
ax.text(6.2, 11.05, "categorical vector\n(N x 27)", fontsize=6, ha="center", va="center")
arrow(3.8, 12.35, 3.8, 12.0); arrow(6.2, 12.35, 6.2, 12.0)

# converging arrows to feature conversion (wide bar)
for xx in [3.8, 6.2]:
    arrow(xx, 10.85, 5.0, 10.65)

# Feature conversion - wide bar spanning all 3 vectors
box(4.9, 10.3, 7.8, 0.7, "Feature conversion", 8, "#d7bde2", "#7d3c98", fw="bold")

# feature vectors 1..N - longer arrows from wide bar
for i, (xx, t) in enumerate([(1.9, "feature\nvector 1"), (4.0, "feature\nvector 2"), (7.9, "feature\nvector N")]):
    box(xx, 9.2, 1.4, 0.6, t, 6.5, "#fdebd0", "#b7950b", style="round,pad=0.02,rounding_size=0.15")
    arrow(xx, 9.95, xx, 9.55, lw=1.4)
ax.text(5.9, 9.2, "...", fontsize=14, ha="center", va="center")
for xx in [1.9, 4.0, 7.9]:
    arrow(xx, 8.9, xx, 8.45, lw=2.0, color="#7fb3d5")

# Big encoder box -> Graph block (minimal, main ideas only)
big = mpatches.Rectangle((0.5, 2.2), 9.0, 6.2, facecolor="#aed6f1", edgecolor="#1a5276", linewidth=1.2)
ax.add_patch(big)
ax.text(5.0, 7.9, "kNN Graph + LabelSpreading", fontsize=16, fontweight="bold", ha="center", va="center")
ax.text(5.0, 7.1, "Node = employee (N)   •   Edge = 15 nearest neighbours", fontsize=8, ha="center")
ax.text(5.0, 6.4, "Train: label 0 / 1      Test: label −1 → P(Attrition)", fontsize=8, ha="center")
ax.text(5.0, 5.5, "F(t+1) = α·S·F(t) + (1−α)·Y,   α = 0.1", fontsize=9, ha="center", fontweight="bold")
ax.text(5.0, 4.6, "P(Attrition) per node from graph position", fontsize=8, ha="center", fontweight="bold")
ax.text(5.0, 3.8, "Similar profile → same risk", fontsize=8, ha="center", style="italic")

# red arrow down directly to output (no threshold)
ax.annotate("", xy=(4.0, 1.65), xytext=(4.0, 2.2),
    arrowprops=dict(arrowstyle="-|>", color="#e74c3c", lw=3, mutation_scale=18))

# final output (architecture only)
box(4.0, 1.05, 1.8, 0.7, "Output\nAttrition Yes/No", 7, "#d7bde2", "#7d3c98", fw="bold")

plt.savefig("reports/ibm/graph_paper_style.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved reports/ibm/graph_paper_style.png")
