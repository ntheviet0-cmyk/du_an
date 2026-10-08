"""draw_graph_paper.py - Paper-style CONCEPTUAL diagram cho Graph Pipeline A.
Minh hoa khai quat (khong chi tiet so): dataset -> features -> 5 steps
(SHAP select -> graph -> GCN -> SVM -> output). File: reports/ibm/graph_paper_style.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(10, 14))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 10)
ax.set_ylim(2.5, 18.5)
ax.axis("off")

NAVY = "#1a5276"
BLUE_BG = "#aed6f1"

def box(x, y, w, h, text, fontsize=7, fc="white", ec="black", fw="normal"):
    r = mpatches.FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02",
        facecolor=fc, edgecolor=ec, linewidth=1.0)
    ax.add_patch(r)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, fontweight=fw, linespacing=1.4)
    return r

def arrow(x1, y1, x2, y2, lw=1.2, color="black"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=0, shrinkB=2))

def grid_box(x, y, w, h, rows=2, cols=4, fc="#d5f5e3", ec="#27ae60"):
    r = mpatches.Rectangle((x - w / 2, y - h / 2), w, h, facecolor=fc, edgecolor=ec, linewidth=1.0)
    ax.add_patch(r)
    for i in range(1, rows):
        ax.plot([x - w / 2, x + w / 2], [y - h / 2 + i * h / rows] * 2, color=ec, lw=0.8)
    for j in range(1, cols):
        ax.plot([x - w / 2 + j * w / cols] * 2, [y - h / 2, y + h / 2], color=ec, lw=0.8)

def step_box(x, y, w, h, num, title, body):
    """Hop step khai quat: badge so + tieu de + 1-2 dong mo ta + (tuy chon) ve them."""
    r = mpatches.FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02",
        facecolor="white", edgecolor=NAVY, linewidth=1.4)
    ax.add_patch(r)
    c = plt.Circle((x - w / 2 + 0.35, y + h / 2 - 0.3), 0.22, color=NAVY, zorder=3)
    ax.add_patch(c)
    ax.text(x - w / 2 + 0.35, y + h / 2 - 0.3, str(num), ha="center", va="center",
            fontsize=8, fontweight="bold", color="white", zorder=4)
    ax.text(x - w / 2 + 0.68, y + h / 2 - 0.3, title, ha="left", va="center",
            fontsize=8, fontweight="bold", color=NAVY)
    for dy, txt, fs, bold in body:
        ax.text(x, y + dy, txt, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color="black")
    return r

# ================= NUA TREN =================
box(5.0, 17.5, 2.0, 0.5, "HR_dataset", 7.5, fw="bold")
ax.text(5.0, 16.9, "Employee table  (1470 x 31)", fontsize=6.5, ha="center")
grid_box(5.0, 16.4, w=1.4, h=0.5)
arrow(5.0, 17.25, 5.0, 17.1)
arrow(5.0, 16.1, 5.0, 15.95)

box(3.8, 15.4, 1.8, 0.6, "Numeric (24)\nStandardScaler", 6.5)
box(6.2, 15.4, 1.8, 0.6, "Categorical (7)\nOneHot", 6.5)
arrow(4.7, 16.1, 3.8, 15.7)
arrow(5.3, 16.1, 6.2, 15.7)
grid_box(3.8, 14.65, w=1.5, h=0.45)
grid_box(6.2, 14.65, w=1.5, h=0.45)
arrow(3.8, 15.1, 3.8, 14.9)
arrow(6.2, 15.1, 6.2, 14.9)

box(5.0, 13.9, 7.6, 0.6, "Feature matrix  (N x 51)", 7.5, "#d7bde2", "#7d3c98", fw="bold")
arrow(3.8, 14.35, 5.0, 14.2)
arrow(6.2, 14.35, 5.0, 14.2)

ax.text(5.0, 13.0, "Pipeline A:  Graph  +  GCN  +  SVM", fontsize=11, fontweight="bold",
        ha="center", color=NAVY)
arrow(5.0, 13.6, 5.0, 13.35)
arrow(5.0, 12.7, 5.0, 12.45)

# ================= KHOI XANH: 5 STEPS KHAI QUAT =================
ax.add_patch(mpatches.Rectangle((0.5, 3.0), 9.0, 9.2, facecolor=BLUE_BG, edgecolor=NAVY, linewidth=1.4))
BX, BW = 5.0, 8.0

# S1
step_box(BX, 11.35, BW, 1.15, 1, "Select features", [(0.05, "SHAP top features  ->  keep 15 / 51", 7, False)])
# S2: do thi khai quat - 1 node trung tam + 4 ve tinh
step_box(BX, 9.6, BW, 2.1, 2, "Build employee graph", [])
cx, cy = 3.0, 9.35
for sx, sy, w, col in [(2.0, 10.0, 2.2, "#1f77b4"), (4.0, 10.0, 1.6, "#1f77b4"),
                       (4.2, 8.9, 1.0, "#d62728"), (2.0, 8.8, 1.3, "#1f77b4")]:
    ax.plot([cx, sx], [cy, sy], color="#555555", lw=w, alpha=0.7, zorder=2)
    ax.scatter([sx], [sy], s=110, c=col, edgecolors="black", linewidths=0.7, zorder=3)
ax.scatter([cx], [cy], s=170, c="#d62728", edgecolors="black", linewidths=0.8, zorder=3)
ax.text(6.3, 9.75, "node = employee", fontsize=7, ha="center")
ax.text(6.3, 9.35, "edge = similarity", fontsize=7, ha="center")
ax.text(6.3, 8.95, "(thicker = more similar)", fontsize=6.5, ha="center", style="italic", color="#333333")
# S3
step_box(BX, 7.75, BW, 1.3, 3, "Learn with GCN",
         [(0.1, "aggregate neighbor info  ->  embedding / employee", 7, False),
          (-0.3, "2 layers", 6.5, False)])
# S4
step_box(BX, 6.35, BW, 1.2, 4, "Classify with SVM",
         [(0.0, "[ original features ]  +  [ embedding ]  ->  SVM", 7, False)])
# S5
step_box(BX, 4.6, BW, 1.9, 5, "Output",
         [(0.35, "P(Attrition)  ->  Stay / Leave", 7.5, True),
          (-0.45, "AUC 0.822   •   PR-AUC 0.599   •   ACC 0.867", 7.5, True)])

for y1, y2 in [(10.75, 10.65), (8.55, 8.4), (7.1, 6.95), (5.75, 5.55)]:
    arrow(8.5, y1, 8.5, y2, lw=1.4)

plt.savefig("reports/ibm/graph_paper_style.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved reports/ibm/graph_paper_style.png")
