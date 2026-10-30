"""draw_confusion_3way.py - 1 hinh 3 confusion matrices (O lai / Roi di) cho RF, XGB, GraphSAGE.

Nguon: reports/ibm/ibm_results.json (Graph = ban best-tune h64-drop0.3 seed 42009,
do retrain_graph_best.py; RF/XGB official).
Output: reports/ibm/confusion_matrices_3way.png + reports/ibm/confusion_3way.csv
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent
REP = REPO / "reports" / "ibm"
DATA = json.loads((REP / "ibm_results.json").read_text(encoding="utf-8"))

MODELS = [
    ("RandomForest", "Random Forest"),
    ("XGBoost", "XGBoost"),
    ("Graph", "GraphSAGE"),
]
LABELS = ["Ở lại", "Rời đi"]  # 0 = No, 1 = Yes

# ---- gom so lieu + verify ----
rows_csv = []
cms = []
for key, label in MODELS:
    m = DATA["models"][key]
    cm = np.array(m["confusion_matrix"], dtype=int)
    assert cm.shape == (2, 2), f"{key}: shape {cm.shape}"
    assert int(cm.sum()) == m["n_test"] == 294, f"{key}: tong != 294"
    assert int(cm[0].sum()) == 247 and int(cm[1].sum()) == 47, f"{key}: hang phai 247/47"
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    # doi chieu P/R voi JSON (sai so lam tron)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    assert abs(p - m["precision"]) < 1e-6, f"{key}: precision lech"
    assert abs(r - m["recall"]) < 1e-6, f"{key}: recall lech"
    cms.append((key, label, cm, m))
    rows_csv.append({
        "model": label, "threshold": round(float(m["best_threshold"]), 2),
        "TN_o_lai_dung": tn, "FP_bao_dong_gia": fp,
        "FN_bo_sot": fn, "TP_bat_dung": tp,
        "precision": round(float(m["precision"]), 3),
        "recall": round(float(m["recall"]), 3),
        "f1": round(float(m["f1"]), 3),
    })
    print(f"[{label}] TN={tn} FP={fp} FN={fn} TP={tp} "
          f"(hang O lai={cm[0].sum()}, hang Roi di={cm[1].sum()})")

with open(REP / "confusion_3way.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
    w.writeheader()
    w.writerows(rows_csv)
print("[csv] saved confusion_3way.csv")

# ---- ve 1x3 ----
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
fig.suptitle("Ma trận nhầm lẫn — Test (n=294: 247 Ở lại, 47 Rời đi)", fontsize=13, fontweight="bold")
vmax = max(cm.max() for _, _, cm, _ in cms)

for ax, (key, label, cm, m) in zip(axes, cms):
    row_sum = cm.sum(axis=1, keepdims=True).astype(float)
    pct = cm / np.maximum(row_sum, 1) * 100.0
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Dự đoán:\nỞ lại", "Dự đoán:\nRời đi"], fontsize=9)
    if ax is axes[0]:
        ax.set_yticklabels(["Thực tế:\nỞ lại", "Thực tế:\nRời đi"], fontsize=9)
    else:
        ax.set_yticklabels(["Ở lại", "Rời đi"], fontsize=9)
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > vmax * 0.55 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.1f}%)",
                    ha="center", va="center", fontsize=11, fontweight="bold", color=color)
    ax.set_title(label, fontsize=11, fontweight="bold")

fig.tight_layout(rect=[0, 0, 1, 0.90])
out = REP / "confusion_matrices_3way.png"
fig.savefig(out, dpi=150)
plt.close()
print("[png] saved confusion_matrices_3way.png")
