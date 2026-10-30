"""Draw a presentation-ready KernelSHAP architecture for frozen GraphSAGE."""
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports" / "ibm" / "shap_architecture.png"
WEB = ROOT / "web" / "assets" / "shap_architecture.png"

BG = "#f3f6fa"
INK = "#14283d"
MUTED = "#5d7083"
NAVY = "#173f5f"
BLUE = "#247ba0"
CYAN = "#70c1b3"
AMBER = "#f3a712"
PURPLE = "#7656a3"
RED = "#e85d75"
GREEN = "#2a9d6f"
LINE = "#cbd5df"

fig, ax = plt.subplots(figsize=(20, 11.25))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 20)
ax.set_ylim(0, 11.25)
ax.axis("off")


def box(x, y, w, h, fc="white", ec=LINE, lw=1.2, radius=0.15, z=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.025,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
    )
    ax.add_patch(patch)
    return patch


def label(x, y, value, size=9, color=INK, weight="normal", ha="center", va="center", z=7):
    ax.text(x, y, value, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=z)


def arrow(x1, y1, x2, y2, color=NAVY, lw=1.7, z=5):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
        color=color, linewidth=lw, shrinkA=2, shrinkB=2, zorder=z,
    ))


def stage_header(x, number, title, color, width=3.35):
    ax.add_patch(Circle((x + 0.31, 8.78), 0.22, facecolor=color, edgecolor="none", zorder=6))
    label(x + 0.31, 8.78, str(number), 9, "white", "bold")
    label(x + 0.62, 8.78, title, 10, INK, "bold", ha="left")
    ax.plot([x + 0.18, x + width - 0.18], [8.43, 8.43], color=color, lw=2.2, zorder=5)


# Header
box(0.35, 9.67, 19.3, 1.2, fc=NAVY, ec=NAVY, radius=0.22)
label(0.85, 10.47, "KIẾN TRÚC GIẢI THÍCH GRAPHSAGE BẰNG KERNELSHAP",
      18, "white", "bold", ha="left")
label(0.86, 10.02,
      "Từ hồ sơ nhân viên đến đóng góp SHAP có thể đọc được bởi HR",
      10, "#d9e8f2", ha="left")
for x, text_value in [(15.25, "GraphSAGE frozen"), (17.15, "kNN k = 15"), (18.62, "51 features")]:
    width = 1.55 if x < 18 else 1.05
    box(x, 10.02, width, 0.39, fc="#285675", ec="#4f7892", lw=0.8, radius=0.18, z=4)
    label(x + width / 2, 10.215, text_value, 7.2, "white", "bold")

# Main flow cards
xs = [0.45, 4.35, 8.25, 12.15, 16.05]
colors = [BLUE, AMBER, CYAN, PURPLE, GREEN]
titles = ["Hồ sơ đầu vào", "Tạo coalition", "Mô hình hộp đen", "Ước lượng SHAP", "Diễn giải cho HR"]
for i, (x, color, title) in enumerate(zip(xs, colors, titles), 1):
    box(x, 5.05, 3.5, 4.12, fc="white", ec=color, lw=1.5, radius=0.20)
    stage_header(x, i, title, color, 3.5)
for left, right in zip(xs[:-1], xs[1:]):
    arrow(left + 3.53, 7.05, right - 0.05, 7.05)

# Stage 1: input and preprocessing
box(0.82, 7.42, 2.76, 0.68, fc="#e8f2f8", ec=BLUE, lw=1.0, radius=0.12)
label(2.20, 7.77, "1 nhân viên cần giải thích", 9, BLUE, "bold")
features = [("Tuổi", "25"), ("Làm thêm", "Có"), ("Thu nhập", "1.118"), ("Hài lòng", "4/4")]
for row, (name, value) in enumerate(features):
    y = 7.08 - row * 0.37
    label(0.88, y, name, 7.6, MUTED, ha="left")
    label(3.48, y, value, 7.8, INK, "bold", ha="right")
    ax.plot([0.88, 3.48], [y - 0.19, y - 0.19], color="#e5ebf0", lw=0.7)
box(0.82, 5.25, 2.76, 0.44, fc=NAVY, ec=NAVY, radius=0.11)
label(2.20, 5.47, "Tiền xử lý  →  vector 51 chiều", 8.2, "white", "bold")

# Stage 2: background and masks
box(4.73, 7.42, 2.76, 0.68, fc="#fff4dc", ec=AMBER, lw=1.0, radius=0.12)
label(6.11, 7.78, "Background: 30 mẫu train", 8.8, "#986600", "bold")
label(6.11, 7.51, "đại diện cả hai lớp", 7.2, MUTED)
label(6.11, 7.05, "1 = giữ giá trị thật", 7.6, INK)
label(6.11, 6.75, "0 = thay bằng background", 7.6, INK)
mask = [1, 0, 1, 1, 0, 0, 1]
for i, bit in enumerate(mask):
    x = 4.78 + i * 0.39
    box(x, 6.15, 0.31, 0.37, fc="#fff0cf" if bit else "#edf1f4",
        ec=AMBER if bit else "#aab6bf", lw=0.9, radius=0.06)
    label(x + 0.155, 6.335, str(bit), 7.5, "#9a6800" if bit else MUTED, "bold")
label(6.11, 5.76, "200 coalition / nhân viên", 8.0, AMBER, "bold")
label(6.11, 5.45, "≈ 6.000 lượt gọi mô hình", 7.5, MUTED)

# Stage 3: graph plus neural network
cx, cy = 9.22, 7.27
angles = np.linspace(0, 2 * np.pi, 7, endpoint=False) + 0.18
neighbors = [(cx + 0.63 * np.cos(a), cy + 0.52 * np.sin(a)) for a in angles]
for nx, ny in neighbors:
    ax.plot([cx, nx], [cy, ny], color="#9cb5bd", lw=1.0, zorder=3)
for nx, ny in neighbors:
    ax.add_patch(Circle((nx, ny), 0.10, facecolor="#76c8d0", edgecolor=NAVY, lw=0.65, zorder=4))
ax.add_patch(Circle((cx, cy), 0.17, facecolor=RED, edgecolor="#9e3549", lw=1.0, zorder=5))
label(9.22, 6.48, "Gắn node vào 15 hàng xóm", 7.5, MUTED)
layers = [("SAGEConv", "51 → 64"), ("SAGEConv", "64 → 32"), ("Sigmoid", "P(nghỉ việc)")]
for row, (name, dims) in enumerate(layers):
    y = 6.05 - row * 0.48
    box(8.60, y, 2.75, 0.36, fc="#e5f5f2", ec=CYAN, lw=0.9, radius=0.08)
    label(8.77, y + 0.18, name, 7.4, NAVY, "bold", ha="left")
    label(11.17, y + 0.18, dims, 7.4, INK, "bold", ha="right")
box(9.05, 7.92, 1.88, 0.30, fc="#e8f6ef", ec=GREEN, lw=0.8, radius=0.12)
label(9.99, 8.07, "FROZEN • không train lại", 6.7, GREEN, "bold")

# Stage 4: weighted local surrogate
label(13.90, 7.82, "mask  z", 7.4, MUTED, "bold")
label(15.20, 7.82, "f(x)", 7.4, MUTED, "bold")
rows = [("1110", "0.82"), ("1001", "0.45"), ("0011", "0.28")]
for row, (mask_value, proba) in enumerate(rows):
    y = 7.48 - row * 0.34
    box(12.62, y - 0.13, 1.35, 0.28, fc="#f5f0fa", ec="#d8c9e6", lw=0.6, radius=0.04)
    label(13.30, y, mask_value, 7.2, INK)
    label(15.18, y, proba, 7.2, PURPLE, "bold")
box(12.52, 6.05, 2.95, 0.72, fc="#f0e8f7", ec=PURPLE, lw=1.0, radius=0.12)
label(14.00, 6.49, r"$g(z)=\phi_0+\sum_j\phi_jz_j$", 11.5, PURPLE, "bold")
label(14.00, 6.19, "hồi quy có trọng số Kernel", 7.2, MUTED)
label(14.00, 5.67, "φj = đóng góp của feature j", 8.0, PURPLE, "bold")
label(14.00, 5.35, "dương: tăng  •  âm: giảm", 7.2, MUTED)

# Stage 5: grouped HR values
items = [("Làm thêm giờ", 0.22), ("Hài lòng", -0.10), ("Thu nhập", 0.07), ("Thâm niên", -0.04)]
zero = 18.04
for row, (name, value) in enumerate(items):
    y = 7.78 - row * 0.52
    label(16.35, y, name, 7.5, INK, ha="left")
    width = value * 4.2
    color = RED if value > 0 else BLUE
    x = zero if value > 0 else zero + width
    ax.add_patch(plt.Rectangle((x, y - 0.13), abs(width), 0.26, facecolor=color,
                               edgecolor="none", alpha=0.88, zorder=4))
    label(19.31, y, f"{value * 100:+.1f} đ.%", 7.3, color, "bold", ha="right")
ax.plot([zero, zero], [6.04, 8.03], color="#9aa9b5", lw=0.9, ls="--", zorder=3)
box(16.42, 5.35, 2.76, 0.52, fc="#e8f6ef", ec=GREEN, lw=1.0, radius=0.11)
label(17.80, 5.61, "51 features → 30 nhóm HR gốc", 7.8, GREEN, "bold")

# Output layer
label(0.55, 4.60, "ĐẦU RA GIẢI THÍCH", 10, NAVY, "bold", ha="left")
ax.plot([2.60, 19.45], [4.60, 4.60], color=LINE, lw=1.2)
outputs = [
    (0.55, BLUE, "GLOBAL", "Yếu tố quan trọng chung", "mean(|SHAP|) trên nhiều ca"),
    (5.30, CYAN, "HƯỚNG TÁC ĐỘNG", "Giá trị nào tăng / giảm", "beeswarm & dependence"),
    (10.05, PURPLE, "NHÓM NHÂN SỰ", "So sánh theo cohort", "Sales • OT • thâm niên"),
    (14.80, AMBER, "LOCAL", "Giải thích một nhân viên", "giá trị • chiều • điểm %"),
]
for x, color, tag, title, subtitle in outputs:
    box(x, 2.82, 4.15, 1.40, fc="white", ec=color, lw=1.35, radius=0.17)
    box(x + 0.22, 3.82, 1.18, 0.25, fc=color, ec=color, radius=0.10, z=4)
    label(x + 0.81, 3.945, tag, 6.6, "white", "bold")
    label(x + 0.24, 3.51, title, 9.2, INK, "bold", ha="left")
    label(x + 0.24, 3.16, subtitle, 7.5, MUTED, ha="left")

# Additivity and guardrail footer
box(0.55, 1.20, 12.35, 1.10, fc="#eaf1f6", ec=NAVY, lw=1.1, radius=0.15)
label(0.86, 1.92, "KIỂM TRA CỘNG TÍNH", 7.3, BLUE, "bold", ha="left")
label(0.86, 1.57, r"$P(nghỉ\ việc)=E[f(X)]+\sum_j SHAP_j$", 12.0, INK, "bold", ha="left")
label(7.44, 1.58, "Base 0.218  +  tổng đóng góp  =  dự báo cá nhân", 8.1, MUTED, ha="left")
box(13.25, 1.20, 6.20, 1.10, fc="#fff3e6", ec=AMBER, lw=1.1, radius=0.15)
label(13.57, 1.92, "CÁCH HIỂU ĐÚNG", 7.3, AMBER, "bold", ha="left")
label(13.57, 1.59, "SHAP giải thích hành vi của mô hình", 8.4, INK, "bold", ha="left")
label(13.57, 1.34, "Không phải bằng chứng nguyên nhân thực tế", 7.7, MUTED, ha="left")

label(0.55, 0.65,
      "KernelSHAP là phép xấp xỉ: kết quả phụ thuộc background và số coalition • GraphSAGE luôn ở chế độ eval/frozen",
      7.8, MUTED, ha="left")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor=BG)
plt.close(fig)
WEB.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(OUT, WEB)
print("[ok] saved reports/ibm/shap_architecture.png + web/assets/shap_architecture.png")
