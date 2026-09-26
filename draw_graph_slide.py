"""draw_graph_slide.py - Diagram slide-ready cho mô hình Graph dự đoán (thuyết trình)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# 16:9 slide format
fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")

# Title
ax.text(8, 8.4, "MÔ HÌNH GRAPH DỰ ĐOÁN NGHỈ VIỆC  |  LabelSpreading trên đồ thị tương đồng nhân viên",
        ha="center", va="center", fontsize=16, fontweight="bold", color="#1a1a1a")
ax.text(8, 7.9, "Mỗi nhân viên là 1 node  •  Cạnh nối nhân viên giống nhau (kNN, k=15)  •  Lan truyền nhãn bán giám sát",
        ha="center", va="center", fontsize=11, color="#555555")

colors = ["#1f77b4", "#e67e22", "#27ae60", "#e74c3c", "#8e44ad"]
boxes = [
    ("1. DỮ LIỆU", "1,470 nhân viên\n31 đặc trưng\nAttrition = 16.1%"),
    ("2. TIỀN XỬ LÝ", "StandardScaler\n+ OneHotEncoder\n→ ma trận 51 chiều"),
    ("3. ĐỒ THỊ kNN", "Node = nhân viên\nEdge = tương đồng\n1972 nodes (sau SMOTE)"),
    ("4. LAN TRUYỀN NHÃN", "LabelSpreading\nα = 0.1, k = 15\nmin fLf"),
    ("5. DỰ ĐOÁN", "P(Attrition)\nNgưỡng = 0.70\nAUC = 0.691"),
]

w, h = 2.7, 2.2
y = 5.6
xs = [1.7, 4.75, 7.8, 10.85, 13.9]

for i, (title, body) in enumerate(boxes):
    x = xs[i]
    rect = mpatches.FancyBboxPatch((x-w/2, y-h/2), w, h,
        boxstyle="round,pad=0.08", facecolor=colors[i],
        edgecolor="black", linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, y+0.55, title, ha="center", va="center",
            fontsize=12, fontweight="bold", color="white")
    for j, line in enumerate(body.split("\n")):
        ax.text(x, y-0.15-j*0.38, line, ha="center", va="center",
                fontsize=10.5, color="white", fontweight="normal")
    if i < 4:
        ax.annotate("", xy=(xs[i+1]-w/2-0.05, y), xytext=(x+w/2+0.05, y),
            arrowprops=dict(arrowstyle="->", color="black", lw=2.5))

# Bottom: 3 key ideas for speaking
bottom = [
    ("Ý TƯỞNG", "Nhân viên giống nhau\n→ cùng nguy cơ nghỉ", "#d6eaf8"),
    ("CÔNG THỨC", "W = kNN similarity\nL = D − W", "#d5f5e3"),
    ("KẾT QUẢ", "Recall = 0.53\nBổ trợ cho RF (0.80)", "#fadbd8"),
]
bw, bh = 4.4, 1.9
by = 2.5
bxs = [3.0, 8.0, 13.0]
for x, (t, b, c) in zip(bxs, bottom):
    r = mpatches.FancyBboxPatch((x-bw/2, by-bh/2), bw, bh,
        boxstyle="round,pad=0.08", facecolor=c, edgecolor="black", linewidth=1.2)
    ax.add_patch(r)
    ax.text(x, by+0.45, t, ha="center", va="center", fontsize=12, fontweight="bold")
    for j, line in enumerate(b.split("\n")):
        ax.text(x, by-0.2-j*0.35, line, ha="center", va="center", fontsize=11)

ax.text(8, 0.7, "Nói khi trình bày:  Dữ liệu → vector 51 chiều → nối mỗi người với 15 người giống nhất → lan nhãn từ người đã biết sang người chưa biết → ra xác suất nghỉ việc.",
        ha="center", va="center", fontsize=10, style="italic", color="#333333",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f2f3f4"))

plt.savefig("reports/ibm/graph_slide.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved reports/ibm/graph_slide.png")
