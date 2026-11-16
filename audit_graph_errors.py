"""Build an auditable casebook for current GraphSAGE test errors.

Sources:
- reports/ibm/roc_proba.npz: persisted transductive test probabilities
- reports/ibm/ibm_results.json: deployed threshold and confusion matrix
- reports/ibm/graph_shap_cases.csv: local inductive SHAP where already available

This script does not train or score the model again. Cases without persisted local
SHAP are described as observed profiles, never as model attributions.
"""
from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
REP = ROOT / "reports" / "ibm"
ART = ROOT / "models" / "ibm"
SEED = 42
CAT = ["BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime"]
DROP = ["EmployeeCount", "Over18", "StandardHours"]

COLORS = {
    "ink": "#17202a", "muted": "#667085", "line": "#d0d5dd",
    "fp": "#d97706", "fn": "#c2415d", "green": "#067647", "blue": "#175cd3",
    "soft_fp": "#fff7ed", "soft_fn": "#fff1f3", "navy": "#16324f",
}


def fmt_probability(value: float) -> str:
    return f"{value * 100:.1f}%".replace(".", ",")


def observed_profile(row: pd.Series) -> str:
    return (
        f"OT={row.OverTime}; {row.JobRole}; {row.BusinessTravel}; {row.MaritalStatus}; "
        f"tuổi {int(row.Age)}; HL={int(row.JobSatisfaction)}; WLB={int(row.WorkLifeBalance)}; "
        f"MT={int(row.EnvironmentSatisfaction)}; xa {int(row.DistanceFromHome)}; "
        f"SO={int(row.StockOptionLevel)}; lương {int(row.MonthlyIncome)}; "
        f"level {int(row.JobLevel)}; thâm niên {int(row.YearsAtCompany)}"
    )


def review_priority(error_type: str, probability: float, threshold: float) -> tuple[str, str]:
    if error_type == "FN":
        if probability < 0.20:
            return "Critical", "Model rất tự tin dự đoán ở lại nhưng thực tế nghỉ; cần kiểm tra thiếu feature hoặc graph neighborhood."
        if probability < 0.50:
            return "High", "Không thể sửa chỉ bằng thay đổi nhỏ quanh threshold; cần kiểm tra representation và dữ liệu đầu vào."
        return "Near threshold", "Có thể nhạy với chính sách threshold, nhưng phải đánh giá lại precision/recall toàn tập."
    if probability >= 0.95:
        return "Critical", "Model rất tự tin báo nghỉ nhưng thực tế ở lại; cần kiểm tra pattern proxy và hàng xóm kNN."
    if probability >= 0.90:
        return "High", "Báo động giả có độ tự tin cao; nên rà feature và neighborhood trước khi hành động."
    return "Near threshold", "Sai gần threshold; không nên dùng điểm đơn lẻ để ra quyết định cá nhân."


def load_inputs():
    raw = pd.read_csv(DATA)
    y = (raw["Attrition"] == "Yes").astype(int).to_numpy()
    all_idx = np.arange(len(raw))
    idx_train, idx_test = train_test_split(all_idx, test_size=0.2, stratify=y, random_state=SEED)

    saved = np.load(REP / "roc_proba.npz")
    y_test = saved["y_test"].astype(int)
    proba = saved["proba_graph_ft"].astype(float)
    assert np.array_equal(y_test, y[idx_test]), "Saved y_test no longer matches seed-42 split"

    results = json.loads((REP / "ibm_results.json").read_text(encoding="utf-8"))
    graph = results["models"]["Graph"]
    threshold = float(graph["best_threshold"])
    expected_cm = np.asarray(graph["confusion_matrix"], dtype=int)
    pred = (proba >= threshold).astype(int)
    actual_cm = confusion_matrix(y_test, pred, labels=[0, 1])
    assert np.array_equal(actual_cm, expected_cm), f"CM mismatch: {actual_cm.tolist()} != {expected_cm.tolist()}"
    assert actual_cm.tolist() == [[235, 12], [25, 22]], actual_cm.tolist()
    return raw, y, idx_train, idx_test, y_test, proba, pred, threshold, actual_cm


def transformed_all(raw: pd.DataFrame, idx_train: np.ndarray, idx_test: np.ndarray):
    model_frame = raw.drop(columns=DROP + ["EmployeeNumber"]).copy()
    model_frame["Attrition"] = (model_frame["Attrition"] == "Yes").astype(int)
    numerical = [c for c in model_frame.columns if c not in CAT + ["Attrition"]]
    pre = joblib.load(ART / "preprocessor.joblib")
    x_train = pre.transform(model_frame.iloc[idx_train][CAT + numerical])
    x_test = pre.transform(model_frame.iloc[idx_test][CAT + numerical])
    x_all = np.vstack([x_train, x_test])
    if hasattr(x_all, "toarray"):
        x_all = x_all.toarray()
    return np.asarray(x_all, dtype=np.float32)


def shap_lookup() -> dict[int, dict]:
    path = REP / "graph_shap_cases.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    return {int(row["test_pos"]): row for row in frame.to_dict("records")}


def build_error_table():
    raw, y, idx_train, idx_test, y_test, proba, pred, threshold, cm = load_inputs()
    x_all = transformed_all(raw, idx_train, idx_test)
    ordered_source_idx = np.concatenate([idx_train, idx_test])
    ordered_y = y[ordered_source_idx]
    ordered_emp = raw.iloc[ordered_source_idx]["EmployeeNumber"].astype(int).to_numpy()
    n_train = len(idx_train)
    local_shap = shap_lookup()
    errors = np.where(pred != y_test)[0]
    rows = []
    neighbor_rows = []

    for test_pos in errors:
        source_idx = int(idx_test[test_pos])
        source = raw.iloc[source_idx]
        probability = float(proba[test_pos])
        error_type = "FP" if y_test[test_pos] == 0 else "FN"
        logit = math.log(np.clip(probability, 1e-12, 1 - 1e-12) / np.clip(1 - probability, 1e-12, 1))
        priority, review_note = review_priority(error_type, probability, threshold)

        graph_pos = n_train + int(test_pos)
        similarities = cosine_similarity(x_all[graph_pos:graph_pos + 1], x_all)[0]
        similarities[graph_pos] = -np.inf
        neighbor_pos = np.argsort(similarities)[::-1][:15]
        neighbor_leave_rate = float(ordered_y[neighbor_pos].mean())
        neighbor_ids = [int(v) for v in ordered_emp[neighbor_pos]]
        neighbor_sims = [float(v) for v in similarities[neighbor_pos]]
        neighbor_labels = [int(v) for v in ordered_y[neighbor_pos]]
        for rank, (pos, emp, sim, label) in enumerate(zip(neighbor_pos, neighbor_ids, neighbor_sims, neighbor_labels), 1):
            neighbor_rows.append({
                "test_pos": int(test_pos), "EmployeeNumber": int(source.EmployeeNumber),
                "error_type": error_type, "rank": rank, "neighbor_graph_pos": int(pos),
                "neighbor_EmployeeNumber": emp, "cosine_similarity": sim,
                "neighbor_true_attrition": label,
            })

        shap_case = local_shap.get(int(test_pos))
        if shap_case:
            explanation_status = "local_shap_verified_inductive"
            shap_probability = float(shap_case["proba"])
            top_up = "; ".join(str(shap_case.get(f"top_up_{i}", "")) for i in range(1, 4))
            top_down = "; ".join(str(shap_case.get(f"top_down_{i}", "")) for i in range(1, 3))
        else:
            explanation_status = "observed_profile_only"
            shap_probability = np.nan
            top_up = ""
            top_down = ""

        rows.append({
            "error_type": error_type,
            "test_pos": int(test_pos),
            "EmployeeNumber": int(source.EmployeeNumber),
            "source_row": source_idx + 2,
            "actual": int(y_test[test_pos]),
            "predicted": int(pred[test_pos]),
            "probability": probability,
            "logit": logit,
            "threshold": threshold,
            "signed_margin": probability - threshold,
            "absolute_margin": abs(probability - threshold),
            "review_priority": priority,
            "review_note": review_note,
            "observed_profile": observed_profile(source),
            "Age": int(source.Age),
            "OverTime": source.OverTime,
            "JobRole": source.JobRole,
            "BusinessTravel": source.BusinessTravel,
            "MaritalStatus": source.MaritalStatus,
            "JobSatisfaction": int(source.JobSatisfaction),
            "EnvironmentSatisfaction": int(source.EnvironmentSatisfaction),
            "WorkLifeBalance": int(source.WorkLifeBalance),
            "DistanceFromHome": int(source.DistanceFromHome),
            "StockOptionLevel": int(source.StockOptionLevel),
            "MonthlyIncome": int(source.MonthlyIncome),
            "JobLevel": int(source.JobLevel),
            "YearsAtCompany": int(source.YearsAtCompany),
            "neighbor_leave_rate": neighbor_leave_rate,
            "neighbor_ids": ";".join(map(str, neighbor_ids)),
            "neighbor_similarities": ";".join(f"{v:.5f}" for v in neighbor_sims),
            "neighbor_labels": ";".join(map(str, neighbor_labels)),
            "explanation_status": explanation_status,
            "shap_inductive_probability": shap_probability,
            "shap_top_up": top_up,
            "shap_top_down": top_down,
        })

    cases = pd.DataFrame(rows).sort_values(["error_type", "probability"], ascending=[True, False])
    neighbors = pd.DataFrame(neighbor_rows)
    assert len(cases) == 37 and (cases.error_type == "FP").sum() == 12 and (cases.error_type == "FN").sum() == 25
    assert cases.EmployeeNumber.is_unique
    assert cases.probability.between(0, 1).all()
    assert cases.threshold.nunique() == 1 and abs(float(cases.threshold.iloc[0]) - 0.8) < 1e-9
    assert len(neighbors) == 37 * 15
    assert neighbors.cosine_similarity.notna().all()
    assert neighbors.groupby("test_pos").size().eq(15).all()
    return cases, neighbors, cm, threshold


def save_csv(cases: pd.DataFrame, neighbors: pd.DataFrame):
    cases.to_csv(REP / "graph_error_cases.csv", index=False, encoding="utf-8-sig")
    cases[cases.error_type == "FP"].to_csv(REP / "graph_false_positives.csv", index=False, encoding="utf-8-sig")
    cases[cases.error_type == "FN"].to_csv(REP / "graph_false_negatives.csv", index=False, encoding="utf-8-sig")
    neighbors.to_csv(REP / "graph_error_neighbors.csv", index=False, encoding="utf-8-sig")


def draw_overview(cases: pd.DataFrame, cm: np.ndarray, threshold: float):
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    grid = fig.add_gridspec(2, 2, width_ratios=[0.72, 1.28], height_ratios=[1, 1], hspace=0.38, wspace=0.25)
    ax_cm = fig.add_subplot(grid[:, 0])
    ax_prob = fig.add_subplot(grid[0, 1])
    ax_margin = fig.add_subplot(grid[1, 1])
    fig.suptitle("GraphSAGE Error Audit — Test set (n=294)", x=0.04, ha="left", fontsize=20, fontweight="bold", color=COLORS["ink"])
    fig.text(0.04, 0.935, "Threshold 0.80 • 12 false positives • 25 false negatives", fontsize=10.5, color=COLORS["muted"])

    image = ax_cm.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    del image
    row_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() * 0.5 else COLORS["ink"]
            ax_cm.text(j, i, f"{cm[i, j]}\n({row_pct[i, j]:.1f}%)", ha="center", va="center", fontsize=17, fontweight="bold", color=color)
    ax_cm.set_xticks([0, 1], ["Predicted stay", "Predicted leave"])
    ax_cm.set_yticks([0, 1], ["Actual stay", "Actual leave"])
    ax_cm.set_title("A. Confusion matrix", loc="left", fontsize=13, fontweight="bold", pad=14)
    ax_cm.text(0.5, 1.75, "FN=25 is the largest operational risk: 53.2% of actual leavers were missed.", ha="center", fontsize=10, color=COLORS["fn"])

    ordered = cases.sort_values("probability").reset_index(drop=True)
    colors = ordered.error_type.map({"FP": COLORS["fp"], "FN": COLORS["fn"]})
    ax_prob.scatter(np.arange(len(ordered)), ordered.probability, c=colors, s=38, edgecolor="white", linewidth=0.6, zorder=3)
    ax_prob.axhline(threshold, color=COLORS["navy"], linestyle="--", linewidth=1.5, label="Threshold = 0.80")
    ax_prob.fill_between([-1, len(ordered)], threshold, 1, color=COLORS["soft_fp"], alpha=0.7)
    ax_prob.fill_between([-1, len(ordered)], 0, threshold, color=COLORS["soft_fn"], alpha=0.55)
    for _, row in ordered.iloc[[0, 1, -2, -1]].iterrows():
        x = ordered.index[ordered.EmployeeNumber == row.EmployeeNumber][0]
        ax_prob.annotate(str(int(row.EmployeeNumber)), (x, row.probability), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    ax_prob.set_ylim(-0.03, 1.04)
    ax_prob.set_xlim(-1, len(ordered))
    ax_prob.set_ylabel("Predicted leave probability")
    ax_prob.set_xlabel("37 misclassified employees, ordered by probability")
    ax_prob.set_title("B. Probability of every error", loc="left", fontsize=13, fontweight="bold")
    ax_prob.grid(axis="y", color="#e4e7ec", linewidth=0.7)
    ax_prob.legend(frameon=False, loc="upper left")

    critical = pd.concat([
        cases[cases.error_type == "FN"].nlargest(7, "absolute_margin"),
        cases[cases.error_type == "FP"].nlargest(7, "absolute_margin"),
    ]).sort_values("signed_margin")
    y_pos = np.arange(len(critical))
    bar_colors = critical.error_type.map({"FP": COLORS["fp"], "FN": COLORS["fn"]})
    ax_margin.barh(y_pos, critical.signed_margin, color=bar_colors)
    ax_margin.axvline(0, color=COLORS["ink"], linewidth=1)
    ax_margin.set_yticks(y_pos, [f"{r.error_type} • #{int(r.EmployeeNumber)}" for r in critical.itertuples()])
    ax_margin.set_xlabel("Probability − threshold (distance from decision boundary)")
    ax_margin.set_title("C. Most confident errors (7 FN + 7 FP)", loc="left", fontsize=13, fontweight="bold", pad=14)
    ax_margin.grid(axis="x", color="#e4e7ec", linewidth=0.7)

    for ax in (ax_cm, ax_prob, ax_margin):
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(0.04, 0.02, "Source: persisted transductive GraphSAGE probabilities. EmployeeNumber is an identifier, not a model feature.", fontsize=9, color=COLORS["muted"])
    fig.savefig(REP / "graph_error_overview.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_feature_map(cases: pd.DataFrame):
    numeric = ["Age", "MonthlyIncome", "JobLevel", "YearsAtCompany", "DistanceFromHome", "JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance", "StockOptionLevel"]
    raw = pd.read_csv(DATA)
    matrix = []
    for _, row in cases.iterrows():
        vals = []
        source = raw.loc[raw.EmployeeNumber == row.EmployeeNumber].iloc[0]
        for field in numeric:
            std = raw[field].std(ddof=0) or 1
            vals.append((float(source[field]) - raw[field].mean()) / std)
        vals += [1 if source.OverTime == "Yes" else -1,
                 1 if source.BusinessTravel == "Travel_Frequently" else -1,
                 1 if source.MaritalStatus == "Single" else -1]
        matrix.append(vals)
    matrix = np.asarray(matrix)
    labels = numeric + ["OverTime=Yes", "Travel frequently", "Single"]
    row_labels = [f"{r.error_type} #{int(r.EmployeeNumber)}  p={r.probability:.3f}" for r in cases.itertuples()]
    fig, ax = plt.subplots(figsize=(14, 14), facecolor="white")
    norm = TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=2.5)
    image = ax.imshow(np.clip(matrix, -2.5, 2.5), aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=7.5)
    split = int((cases.error_type == "FN").sum())
    if 0 < split < len(cases):
        ax.axhline(split - 0.5, color=COLORS["ink"], linewidth=2)
    fig.suptitle("Observed feature map of GraphSAGE errors", x=0.09, y=0.995, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.09, 0.972, "Standardized numeric fields; +1/-1 binary indicators. Profile context only, not SHAP attribution.", fontsize=10, color=COLORS["muted"])
    colorbar = fig.colorbar(image, ax=ax, fraction=0.02, pad=0.02)
    colorbar.set_label("Relative feature value")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(REP / "graph_error_feature_map.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_neighbor_audit(cases: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), facecolor="white")
    fig.suptitle("Cosine-kNN neighborhood audit", x=0.04, ha="left", fontsize=19, fontweight="bold")
    fig.text(0.04, 0.91, "Top-15 nearest employees in the same 51-feature space used to construct the evaluation graph.", fontsize=10, color=COLORS["muted"])

    ax = axes[0]
    for error_type, color, label in (("FN", COLORS["fn"], "False negative"), ("FP", COLORS["fp"], "False positive")):
        subset = cases[cases.error_type == error_type]
        ax.scatter(subset.neighbor_leave_rate, subset.probability, s=55, color=color,
                   edgecolor="white", linewidth=0.7, alpha=0.9, label=label)
        for row in subset.itertuples():
            if row.review_priority == "Critical" or row.absolute_margin < 0.06:
                ax.annotate(str(int(row.EmployeeNumber)), (row.neighbor_leave_rate, row.probability),
                            xytext=(4, 4), textcoords="offset points", fontsize=7.5)
    ax.axhline(0.80, color=COLORS["navy"], linestyle="--", linewidth=1.3)
    ax.plot([0, 1], [0, 1], color=COLORS["line"], linestyle=":", linewidth=1)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Actual attrition rate among top-15 neighbors")
    ax.set_ylabel("GraphSAGE predicted leave probability")
    ax.set_title("A. Prediction vs neighbor labels", loc="left", fontsize=13, fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(color="#e4e7ec", linewidth=0.7)

    ax = axes[1]
    ordered = cases.sort_values("neighbor_leave_rate")
    y_pos = np.arange(len(ordered))
    colors = ordered.error_type.map({"FN": COLORS["fn"], "FP": COLORS["fp"]})
    ax.barh(y_pos, ordered.neighbor_leave_rate, color=colors)
    ax.set_yticks(y_pos, [f"{r.error_type} • #{int(r.EmployeeNumber)}" for r in ordered.itertuples()], fontsize=7.5)
    ax.axvline(0.5, color=COLORS["navy"], linestyle="--", linewidth=1.2)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Actual leavers / 15 nearest neighbors")
    ax.set_title("B. Neighborhood composition for every error", loc="left", fontsize=13, fontweight="bold")
    ax.grid(axis="x", color="#e4e7ec", linewidth=0.7)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(REP / "graph_error_neighbors.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def wrap(value, width):
    return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=False))


def draw_case_table(frame: pd.DataFrame, title: str, filename: str):
    rows = []
    for row in frame.itertuples():
        if row.explanation_status.startswith("local_shap"):
            explanation = f"SHAP up: {row.shap_top_up}; SHAP down: {row.shap_top_down}"
        else:
            explanation = "Profile only; SHAP not available"
        rows.append([
            str(int(row.EmployeeNumber)), fmt_probability(row.probability), row.review_priority,
            wrap(row.observed_profile, 48), wrap(explanation, 32), wrap(row.review_note, 48),
        ])
    headers = ["Employee", "P(leave)", "Review", "Observed profile", "Explanation", "Audit note"]
    fig_height = max(5.5, len(rows) * 0.62 + 1.8)
    fig, ax = plt.subplots(figsize=(18, fig_height), facecolor="white")
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=18, fontweight="bold", pad=12)
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="left", colLoc="left",
                     colWidths=[0.07, 0.07, 0.09, 0.31, 0.15, 0.31], bbox=[0, 0, 1, 0.94])
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1, 1.25)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(COLORS["line"])
        cell.set_linewidth(0.45)
        if r == 0:
            cell.set_facecolor(COLORS["navy"])
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#ffffff" if r % 2 else "#f8fafc")
            cell.get_text().set_va("center")
    fig.savefig(REP / filename, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_casebooks(cases: pd.DataFrame):
    fp = cases[cases.error_type == "FP"].sort_values("probability", ascending=False)
    fn = cases[cases.error_type == "FN"].sort_values("probability")
    draw_case_table(fp, "False positives — predicted leave, actually stayed (12 cases)", "graph_false_positives.png")
    draw_case_table(fn.iloc[:13], "False negatives — predicted stay, actually left (1/2)", "graph_false_negatives_01.png")
    draw_case_table(fn.iloc[13:], "False negatives — predicted stay, actually left (2/2)", "graph_false_negatives_02.png")


def write_summary(cases: pd.DataFrame, cm: np.ndarray, threshold: float):
    verified = int(cases.explanation_status.str.startswith("local_shap").sum())
    text = f"""# GraphSAGE Error Audit

- Evaluation threshold: `{threshold:.2f}`
- Confusion matrix: `{cm.tolist()}`
- False positives: `{int((cases.error_type == 'FP').sum())}`
- False negatives: `{int((cases.error_type == 'FN').sum())}`
- Local SHAP explanations already persisted: `{verified}/37`

## Interpretation rule

`observed_profile` is descriptive context, not a model attribution and not a causal reason for attrition.
Only rows marked `local_shap_verified_inductive` have a persisted local SHAP explanation. Those SHAP
values come from the inductive wrapper and can differ slightly from the transductive evaluation probability.
Neighbor labels are joined only after prediction for audit; labels were not used to construct cosine edges.

## Outputs

- `graph_error_cases.csv`: all 37 errors
- `graph_false_positives.csv`: 12 false positives
- `graph_false_negatives.csv`: 25 false negatives
- `graph_error_neighbors.csv`: top-15 cosine neighbors for each error
- `graph_error_overview.png`: confusion matrix, probabilities and confidence margins
- `graph_error_feature_map.png`: observed feature map
- `graph_error_neighbors.png`: probability versus top-15 neighborhood labels
- `graph_false_positives.png`: false-positive casebook
- `graph_false_negatives_01.png`, `graph_false_negatives_02.png`: false-negative casebook
"""
    (REP / "GRAPH_ERROR_AUDIT.md").write_text(text, encoding="utf-8")


def main():
    REP.mkdir(parents=True, exist_ok=True)
    cases, neighbors, cm, threshold = build_error_table()
    save_csv(cases, neighbors)
    draw_overview(cases, cm, threshold)
    draw_feature_map(cases)
    draw_neighbor_audit(cases)
    draw_casebooks(cases)
    write_summary(cases, cm, threshold)
    print(f"[audit] threshold={threshold:.2f} CM={cm.tolist()} errors={len(cases)}")
    print(f"[audit] FP={(cases.error_type == 'FP').sum()} FN={(cases.error_type == 'FN').sum()}")
    print(f"[audit] local SHAP verified={cases.explanation_status.str.startswith('local_shap').sum()}/37")
    print("[saved] reports/ibm")


if __name__ == "__main__":
    main()
