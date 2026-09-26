import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ============================================
# COLORS
# ============================================
BLUE = "#1f77b4"
GREEN = "#27ae60"
RED = "#e74c3c"
ORANGE = "#e67e22"
GRAY = "#7f8c8d"

LIGHT_BLUE = "#d6eaf8"
LIGHT_GREEN = "#d5f5e3"
LIGHT_RED = "#fadbd8"

PURPLE = "#8e44ad"


# ============================================
# HELPER FUNCTIONS
# ============================================
def draw_box(
    ax,
    x,
    y,
    w,
    h,
    lines,
    color,
    text_color="white",
    fontsize=8,
    line_spacing=0.28
):
    """
    Draw rounded box with properly spaced text.
    """

    rect = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.08",
        facecolor=color,
        edgecolor="black",
        linewidth=1.2
    )

    ax.add_patch(rect)

    # Total number of visual lines
    visual_lines = []

    for line in lines:
        visual_lines.extend(line.split("\n"))

    n = len(visual_lines)

    # Calculate vertical positions
    total_height = (n - 1) * line_spacing
    start_offset = total_height / 2

    for i, line in enumerate(visual_lines):

        offset = start_offset - i * line_spacing

        ax.text(
            x,
            y + offset,
            line,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color=text_color
        )


def draw_arrow_between(ax, x1, y1, x2, y2):
    """
    Draw horizontal arrow between boxes.
    """

    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->",
            color="black",
            lw=1.5
        )
    )


# ============================================
# FIGURE
# ============================================
fig = plt.figure(figsize=(16, 12))


# ============================================================
# PANEL 1: DATA FLOW DIAGRAM
# ============================================================
ax1 = fig.add_axes([0.05, 0.55, 0.90, 0.40])

ax1.set_xlim(0, 16)
ax1.set_ylim(0, 5)
ax1.axis("off")

ax1.set_title(
    "GRAPH ARCHITECTURE: LabelSpreading on kNN Similarity Graph",
    fontsize=13,
    fontweight="bold",
    pad=10
)


# --------------------------------------------
# Box dimensions
# --------------------------------------------
w = 2.8
h = 1.6


# --------------------------------------------
# Main boxes
# --------------------------------------------
boxes = [

    (
        1.5,
        4,
        [
            "Raw Data",
            "(1,470 employees)",
            "17 features"
        ],
        BLUE,
        8
    ),

    (
        4.5,
        4,
        [
            "Preprocessing",
            "• StandardScaler (numeric)",
            "• OneHotEncoder (cat)",
            "→ 51-dim matrix"
        ],
        ORANGE,
        7.3
    ),

    (
        7.5,
        4,
        [
            "kNN Graph (k=15)",
            "Nodes = employees",
            "Edges = similarity"
        ],
        GREEN,
        8
    ),

    (
        10.5,
        4,
        [
            "LabelSpreading",
            "alpha = 0.1",
            "(semi-supervised)",
            "Harmonic function"
        ],
        RED,
        7.5
    ),

    (
        13.5,
        4,
        [
            "Prediction",
            "P(Attrition)",
            "Threshold: 0.70"
        ],
        PURPLE,
        8
    ),
]


# --------------------------------------------
# Draw main boxes
# --------------------------------------------
for x, y, lines, color, fs in boxes:

    draw_box(
        ax1,
        x,
        y,
        w,
        h,
        lines,
        color,
        fontsize=fs,
        line_spacing=0.22
    )


# --------------------------------------------
# Horizontal arrows
# --------------------------------------------
for i in range(len(boxes) - 1):

    x1 = boxes[i][0] + w / 2 + 0.05
    x2 = boxes[i + 1][0] - w / 2 - 0.05

    draw_arrow_between(
        ax1,
        x1,
        4,
        x2,
        4
    )


# --------------------------------------------
# Information below boxes
# --------------------------------------------
ax1.text(
    1.5,
    2.35,
    "Features:\nAge, MonthlyIncome,\nYearsAtCompany...",
    ha="center",
    va="center",
    fontsize=7.2,
    color=GRAY,
    linespacing=1.3
)

ax1.text(
    4.5,
    2.35,
    "Feature matrix:\nshape (1470, 51)",
    ha="center",
    va="center",
    fontsize=7.2,
    color=GRAY,
    linespacing=1.3
)

ax1.text(
    7.5,
    2.35,
    "Adjacency W:\nsparse matrix\n(1470 × 1470)",
    ha="center",
    va="center",
    fontsize=7.2,
    color=GRAY,
    linespacing=1.3
)

ax1.text(
    10.5,
    2.35,
    "Solve:\nmin fᵀLf\ns.t. f_labelled = y_labelled",
    ha="center",
    va="center",
    fontsize=7.2,
    color=GRAY,
    linespacing=1.3
)

ax1.text(
    13.5,
    2.35,
    "Output:\nBinary\nAttrition Yes / No",
    ha="center",
    va="center",
    fontsize=7.2,
    color=GRAY,
    linespacing=1.3
)


# ============================================================
# PANEL 2: kNN GRAPH VISUALIZATION
# ============================================================
ax2 = fig.add_axes([0.05, 0.05, 0.43, 0.43])

ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)

ax2.set_title(
    "kNN Similarity Graph",
    fontsize=11,
    fontweight="bold",
    pad=8
)


# --------------------------------------------
# Generate nodes
# --------------------------------------------
np.random.seed(42)

n_nodes = 50

angle = np.random.uniform(
    0,
    2 * np.pi,
    n_nodes
)

r = np.random.uniform(
    0.1,
    1.0,
    n_nodes
)

x = r * np.cos(angle)
y = r * np.sin(angle)


# --------------------------------------------
# Simulated labels
# --------------------------------------------
labels = np.random.binomial(
    1,
    0.16,
    n_nodes
)


# --------------------------------------------
# Build kNN edges
# --------------------------------------------
edges = []

for i in range(n_nodes):

    dists = np.sqrt(
        (x[i] - x) ** 2 +
        (y[i] - y) ** 2
    )

    nearest = np.argsort(dists)[1:4]

    for j in nearest:

        if (
            dists[j] < 0.55
            and (j, i) not in edges
        ):
            edges.append((i, j))


# --------------------------------------------
# Draw edges
# --------------------------------------------
for i, j in edges:

    ax2.plot(
        [x[i], x[j]],
        [y[i], y[j]],
        color=GRAY,
        alpha=0.25,
        lw=0.6
    )


# --------------------------------------------
# Draw nodes
# --------------------------------------------
ax2.scatter(
    x,
    y,
    c=[
        RED if l == 1 else BLUE
        for l in labels
    ],
    s=40,
    alpha=0.8,
    edgecolors="black",
    linewidth=0.4
)


# --------------------------------------------
# Legend
# --------------------------------------------
ax2.legend(
    handles=[
        mpatches.Patch(
            color=RED,
            label="Attrition (1)"
        ),
        mpatches.Patch(
            color=BLUE,
            label="Stay (0)"
        )
    ],
    loc="upper right",
    fontsize=7
)


# --------------------------------------------
# Graph information
# --------------------------------------------
ax2.text(
    0,
    -1.15,
    (
        f"{n_nodes} nodes | {len(edges)} edges\n"
        "Red = nghỉ việc | Blue = ở lại\n"
        "Edge weight = 1 (binary)"
    ),
    ha="center",
    va="center",
    fontsize=7,
    linespacing=1.4,
    bbox=dict(
        boxstyle="round,pad=0.25",
        facecolor=LIGHT_BLUE,
        alpha=0.8
    )
)


# ============================================================
# PANEL 3: LABEL PROPAGATION PROCESS
# ============================================================
ax3 = fig.add_axes([0.52, 0.05, 0.43, 0.43])

ax3.set_xlim(0, 16)
ax3.set_ylim(0, 10)

ax3.axis("off")

ax3.set_title(
    "Label Propagation Process",
    fontsize=11,
    fontweight="bold",
    pad=8
)


# --------------------------------------------
# Step box dimensions
# --------------------------------------------
cw = 13.5
ch = 1.45


# --------------------------------------------
# Step positions
# --------------------------------------------
steps = [

    (
        8.0,
        9.0,
        [
            "Step 1: Few labeled nodes",
            "Most nodes: label = ?"
        ],
        LIGHT_BLUE,
        "black"
    ),

    (
        8.0,
        7.2,
        [
            "Step 2: Build kNN graph",
            "W (adjacency) = similarity"
        ],
        LIGHT_GREEN,
        "black"
    ),

    (
        8.0,
        5.4,
        [
            "Step 3: Compute Laplacian",
            "L = D − W (D = degree matrix)"
        ],
        LIGHT_RED,
        "black"
    ),

    (
        8.0,
        3.6,
        [
            "Step 4: Optimize (α = 0.1)",
            "min fᵀLf  s.t.  f_labelled = y_labelled"
        ],
        PURPLE,
        "white"
    ),

    (
        8.0,
        1.8,
        [
            "Step 5: All nodes get labels",
            "Unlabeled → P(Attrition = 1)"
        ],
        LIGHT_BLUE,
        "black"
    ),
]


# --------------------------------------------
# Draw step boxes
# --------------------------------------------
for x_pos, y_pos, lines, color, tc in steps:

    draw_box(
        ax3,
        x_pos,
        y_pos,
        cw,
        ch,
        lines,
        color,
        text_color=tc,
        fontsize=8,
        line_spacing=0.30
    )


# --------------------------------------------
# Draw arrows between steps
# --------------------------------------------
for i in range(len(steps) - 1):

    # Bottom of current box
    y1 = steps[i][1] - ch / 2

    # Top of next box
    y2 = steps[i + 1][1] + ch / 2

    # Center x
    x_arrow = 8.0

    ax3.annotate(
        "",
        xy=(x_arrow, y2 + 0.03),
        xytext=(x_arrow, y1 - 0.03),
        arrowprops=dict(
            arrowstyle="->",
            color="black",
            lw=1.4
        )
    )


# ============================================================
# FINAL LAYOUT
# ============================================================
plt.savefig(
    r"C:/Users/Public/graph_architecture.png",
    dpi=200,
    bbox_inches="tight",
    facecolor="white"
)

plt.close()

print("Done")
print("Saved to: C:/Users/Public/graph_architecture.png")