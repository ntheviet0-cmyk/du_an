"""Render one auditable history table for all recorded GraphSAGE experiments."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
REP = ROOT / "reports" / "ibm"
GNN = ROOT / "reports" / "ibm_gnn"
OUT = REP / "graph_tuning_history.png"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value):
    return "-" if value is None else f"{float(value) * 100:.1f}"


def num(value):
    return "-" if value is None else f"{float(value):.2f}"


def config(cfg):
    if not cfg:
        return "SAGE 64-32"
    return "h{h}, d{drop:g}, lr{lr:g}, wd{wd:g}".format(**cfg)


def row(stage, variant, cfg="", seed="-", k="15", features="51", split="Test",
        auc=None, pr=None, f1=None, precision=None, recall=None, acc=None, threshold=None,
        note=""):
    return [stage, variant, cfg, str(seed), str(k), str(features), split, pct(auc), pct(pr),
            pct(f1), f"{pct(precision)}/{pct(recall)}", pct(acc), num(threshold), note]


results = load(REP / "ibm_results.json")
grid = load(GNN / "sage_grid.json")
k_grid = load(GNN / "k_ablation.json")
features = load(GNN / "feat_select.json")
knowledge = load(GNN / "knowledge.json")
rows = []
styles = []

official = results["models"]["Graph_official"]
rows.append(row("Baseline", "Official GraphSAGE", "SAGE 64-32", 42, 15, 51, "Test",
                official["roc_auc"], official["pr_auc"], official["f1"], official["precision"],
                official["recall"], official["accuracy_default"], official["best_threshold"],
                "Official seed"))
styles.append("official")

best = results["models"]["Graph"]
rows.append(row("Chọn cuối", "Best-tune (app)", config(best["cfg"]), best["seed_run"], 15, 51, "Test",
                best["roc_auc"], best["pr_auc"], best["f1"], best["precision"], best["recall"],
                best["test_acc_valthr"], best["best_threshold"], "Đang dùng trong app"))
styles.append("best")

for run in sorted(grid["runs"], key=lambda item: item["idx"]):
    rows.append(row("Grid (VAL)", f"Grid #{run['idx'] + 1:02d}", config(run["cfg"]), 42_000 + run["idx"],
                    15, 51, "Validation", pr=run["val_pr"], f1=run["val_f1"],
                    acc=run["val_acc"], threshold=run["val_thr"], note=f"{run['epochs']} epochs"))
    styles.append("validation")

for index, run in enumerate(grid["test_once"], 1):
    cfg = run["cfg"]
    matching = next(item for item in grid["runs"] if item["cfg"] == cfg)
    rows.append(row("Grid (TEST)", f"Top-{index} tu VAL", config(cfg), 42_000 + matching["idx"], 15, 51,
                    "Test (1 lan)", run["test_auc"], run["test_pr"], run["test_f1"], run["test_P"],
                    run["test_R"], run["test_acc"], run["thr_from_val"], "Chon theo validation"))
    styles.append("test")

for k, result in sorted(k_grid.items(), key=lambda item: int(item[0])):
    rows.append(row("k ablation", f"k = {k}", "SAGE 64-32", 42, k, 51, "Test",
                    result["roc_auc"], result["pr_auc"], result["f1"], result["precision"],
                    result["recall"], threshold=result["thr"], note=f"{result['epochs']} epochs"))
    styles.append("test")

for name, result in features.items():
    if name == "top20_features":
        continue
    rows.append(row("Feature", name, "SAGE 64-32", 42, 15, result["dim"], "Test",
                    result["test_auc"], result["test_pr"], result["test_f1"], result["test_P"],
                    result["test_R"], result["test_acc"], result["thr_val"], "Feature selection"))
    styles.append("test")

for name in ("KG-full51", "KG-top20"):
    result = knowledge[name]
    rows.append(row("Knowledge graph", name, "SAGE 64-32", 42, 10, "51" if name.endswith("51") else "20",
                    "Test", result["test_auc"], result["test_pr"], result["test_f1"], result["test_P"],
                    result["test_R"], result["test_acc"], result["thr_val"], "Knowledge edges"))
    styles.append("test")

headers = ["Giai đoạn", "Biến thể", "Cấu hình", "Seed", "k", "Features", "Tập",
           "ROC-AUC", "PR-AUC", "F1", "P / R", "ACC", "Ngưỡng", "Ghi chú"]
widths = [0.90, 1.20, 2.35, 0.65, 0.35, 0.55, 0.95, 0.68, 0.68, 0.55, 0.85, 0.55, 0.60, 1.30]
height = len(rows) * 0.32 + 2.25
fig, ax = plt.subplots(figsize=(18, height))
ax.axis("off")
ax.set_title("Lịch sử tuning GraphSAGE - IBM HR Attrition", loc="left", fontsize=18,
             fontweight="bold", pad=10)


table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", colLoc="center",
                 colWidths=[width / sum(widths) for width in widths], bbox=[0, 0.035, 1, 0.94])
table.auto_set_font_size(False)
table.set_fontsize(7.1)
table.scale(1, 1.15)

colors = {"official": "#dbeafe", "best": "#dcfce7", "validation": "#f8fafc", "test": "#fff7ed"}
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#d1d5db")
    cell.set_linewidth(0.35)
    if r == 0:
        cell.set_facecolor("#1e3a5f")
        cell.get_text().set_color("white")
        cell.get_text().set_fontweight("bold")
    else:
        cell.set_facecolor(colors[styles[r - 1]])
        if c in (0, 1, 13):
            cell.get_text().set_ha("left")


fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
print(f"saved {OUT} ({len(rows)} rows)")
