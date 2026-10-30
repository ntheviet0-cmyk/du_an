"""graph_shap_lib.py - Dung chung cho explain_graph_shap.py va app_ibm.py.

Load GNN frozen (models/ibm/graph.joblib) + preprocessor/meta, mo SAGE o eval mode,
cung cap graph_predict_proba(): (N,51) -> (N,) theo kieu inductive (gan node moi vao
graph train co dinh qua top-15 cosine, forward 1 pass, xong bo - graph goc khong doi).
Day KHONG phai model du doan thu hai, chi la lop boc de SHAP/LIME goi duoc GNN.
"""
from pathlib import Path

import joblib
import numpy as np
import torch

from gnn_models import SAGE as SAGE_G

REPO = Path(__file__).resolve().parent
ART = REPO / "models" / "ibm"

GROUP_VN = {
    "OverTime": "Làm thêm giờ",
    "Department": "Phòng ban",
    "JobRole": "Chức danh",
    "EducationField": "Ngành học",
    "Gender": "Giới tính",
    "MaritalStatus": "Tình trạng hôn nhân",
    "BusinessTravel": "Đi công tác",
}

FIELD_VALUE_LABELS = {
    "JobSatisfaction": {1: "Rất thấp", 2: "Thấp", 3: "Cao", 4: "Rất cao"},
    "EnvironmentSatisfaction": {1: "Rất thấp", 2: "Thấp", 3: "Cao", 4: "Rất cao"},
    "RelationshipSatisfaction": {1: "Rất thấp", 2: "Thấp", 3: "Cao", 4: "Rất cao"},
    "JobInvolvement": {1: "Rất thấp", 2: "Thấp", 3: "Cao", 4: "Rất cao"},
    "WorkLifeBalance": {1: "Rất thấp", 2: "Thấp", 3: "Cao", 4: "Rất cao"},
    "Education": {1: "Dưới cao đẳng", 2: "Cao đẳng", 3: "Đại học", 4: "Thạc sĩ", 5: "Tiến sĩ"},
    "JobLevel": {1: "Nhân viên", 2: "Chuyên viên", 3: "Chuyên viên cao cấp", 4: "Quản lý", 5: "Lãnh đạo"},
    "PerformanceRating": {3: "Đạt yêu cầu", 4: "Vượt trội"},
    "StockOptionLevel": {0: "Chưa có", 1: "Ít", 2: "Trung bình", 3: "Nhiều"},
}

_MISSING = object()

_BUNDLE = None


def get_bundle():
    """Load 1 lan, tra ve dict {feat_names, thr, model, X_train, edges, ...}."""
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE
    meta = joblib.load(ART / "meta.joblib")
    g = joblib.load(ART / "graph.joblib")
    Xtr = np.asarray(g["X_train"], dtype=np.float32)
    ei = g["ei_train"]
    gm = SAGE_G(int(g["in_dim"]), h=int(g.get("h", 64)), drop=float(g.get("drop", 0.3)))
    gm.load_state_dict({k: v for k, v in g["sage_state"].items()})
    gm.eval()
    _BUNDLE = {
        "feat_names": [str(f) for f in meta["feat_names"]],
        "thr_graph": float(meta.get("thr_graph", 0.5)),
        "seed_run": g.get("seed_run"),
        "model": gm,
        "X_train": Xtr,
        "edge_rows": ei[0].tolist(),
        "edge_cols": ei[1].tolist(),
        "n_train": Xtr.shape[0],
        "train_norm": np.linalg.norm(Xtr, axis=1) + 1e-9,
        "k": 15,
    }
    return _BUNDLE


def graph_predict_proba(X, bundle=None):
    b = bundle or get_bundle()
    X = np.asarray(X, dtype=np.float32)
    Xtr, gm = b["X_train"], b["model"]
    ER, EC, NTR, NR = b["edge_rows"], b["edge_cols"], b["n_train"], b["train_norm"]
    out = np.zeros(len(X), dtype=np.float64)
    with torch.no_grad():
        for i, xn in enumerate(X):
            sims = (Xtr @ xn) / (NR * (np.linalg.norm(xn) + 1e-9))
            js = np.argsort(sims)[::-1][: b["k"]]
            Xa = np.vstack([Xtr, xn.reshape(1, -1)])
            er = ER + [NTR] * b["k"] + [int(j) for j in js]
            ec = EC + [int(j) for j in js] + [NTR] * b["k"]
            ei = torch.tensor([er, ec], dtype=torch.long)
            logit = gm(torch.tensor(Xa), ei)[NTR]
            out[i] = float(torch.sigmoid(logit).item())
    return out


def feature_group(name):
    """cat__OverTime_Yes -> OverTime; num__Age -> Age."""
    if name.startswith("cat__"):
        return name.split("__", 1)[1].split("_")[0]
    return name.split("__", 1)[1]


def group_order(feature_names):
    """Thu tu truong on dinh theo lan xuat hien trong feat_names."""
    groups, seen = [], set()
    for f in feature_names:
        g = feature_group(f)
        if g not in seen:
            seen.add(g)
            groups.append(g)
    return groups


def grouped_shap(values, row, feature_names, raw_row=None):
    """Gop SHAP theo truong HR (30 truong tu 51 cot one-hot).

    Tra ve (group_names, group_values, display_values). display_values giu
    gia tri goc neu raw_row duoc cung cap; neu khong thi fallback ve du lieu da
    bien doi. Tong group_values bang tong values goc (additivity duoc bao toan).
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    row = np.asarray(row).reshape(-1)
    groups = group_order(feature_names)
    idx_by_group = {g: [] for g in groups}
    for j, f in enumerate(feature_names):
        idx_by_group[feature_group(f)].append(j)
    gvals = np.zeros(len(groups), dtype=float)
    gdata = np.empty(len(groups), dtype=object)
    for i, g in enumerate(groups):
        js = idx_by_group[g]
        gvals[i] = values[js].sum()
        raw_value = raw_row.get(g, _MISSING) if hasattr(raw_row, "get") else _MISSING
        if raw_value is not _MISSING:
            gdata[i] = raw_value
        elif any(feature_names[j].startswith("cat__") for j in js):
            active = [j for j in js if row[j] > 0.5]
            if active:
                name = feature_names[active[0]].split("__", 1)[1]
                gdata[i] = name[len(g) + 1:]
            else:
                gdata[i] = "?"
        else:
            gdata[i] = float(row[js[0]])
    return groups, gvals, gdata


def format_group_value(name, value):
    """Dinh dang gia tri goc; giai nghia cac thang ma hoa HR."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if np.isfinite(number) and number.is_integer():
            integer = int(number)
            meaning = FIELD_VALUE_LABELS.get(name, {}).get(integer)
            return f"{integer} ({meaning})" if meaning else str(integer)
        return f"{number:.3f}".rstrip("0").rstrip(".")
    return str(value).replace("_", " ")


def contribution_order(values):
    """Chi so nhom HR co dong gop khac 0, sap xep theo |SHAP| giam dan."""
    values = np.asarray(values, dtype=float).reshape(-1)
    nonzero = np.flatnonzero(values != 0)
    return nonzero[np.argsort(np.abs(values[nonzero]))[::-1]]


def format_shap_points(value):
    """Dinh dang dong gop SHAP thanh diem phan tram, khong lam tron thanh 0.000."""
    points = float(value) * 100
    if points == 0:
        return "0 đ.%"
    precision = 1 if abs(points) >= 0.1 else 3
    return f"{points:+.{precision}f} đ.%"


def plot_grouped_shap(ax, values, names, display_values=None, hr=None,
                      base_value=None, prediction=None):
    """Ve dong gop SHAP cua tat ca nhom HR goc cho mot nhan vien."""
    values = np.asarray(values, dtype=float).reshape(-1)
    order = contribution_order(values)
    ax.figure.set_size_inches(10, max(5.0, len(order) * 0.42 + 2.0))
    if len(order) == 0:
        ax.text(0.5, 0.5, "Không có yếu tố nào làm thay đổi dự báo.",
                ha="center", va="center", transform=ax.transAxes, fontsize=13)
        ax.set_axis_off()
        return order

    vals = values[order]
    labels = []
    for rank, j in enumerate(order, 1):
        if display_values is None:
            labels.append(f"{rank}. {names[j]}")
            continue
        value = format_group_value(names[j], display_values[j])
        if names[j] in GROUP_VN:
            label = group_label(names[j], hr, display_values[j])
        else:
            label = f"{value} = {group_label(names[j], hr)}"
        labels.append(f"{rank}. {label}")

    y = np.arange(len(order))
    colors = np.where(vals >= 0, "#ff0051", "#008bfb")
    ax.barh(y, vals, color=colors, height=0.72)
    ax.set_yticks(y, labels=labels)
    ax.invert_yaxis()
    ax.axvline(0, color="#777777", linewidth=0.8)
    ax.grid(axis="x", color="#d8d8d8", linewidth=0.6, linestyle=(0, (1, 4)))
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=11)
    ax.set_xlabel("Mức đóng góp vào xác suất nghỉ việc")
    ax.set_title("Các yếu tố góp phần vào dự báo nghỉ việc",
                 loc="left", fontweight="bold")
    if base_value is not None and prediction is not None:
        ax.text(1, 1.015, f"base={base_value:.3f}   prediction={prediction:.3f}",
                ha="right", va="bottom", transform=ax.transAxes, color="#666666")

    lo, hi = min(float(vals.min()), 0.0), max(float(vals.max()), 0.0)
    span = max(hi - lo, 1e-6)
    ax.set_xlim(lo - span * 0.18, hi + span * 0.28)
    offset = span * 0.018
    for yi, value in enumerate(vals):
        ax.text(value + (offset if value >= 0 else -offset), yi, format_shap_points(value),
                ha="left" if value >= 0 else "right", va="center",
                color=colors[yi], fontweight="bold", fontsize=10)
    return order


def group_label(name, hr=None, value=None):
    """Nhan tieng Viet cho truong; kem category active neu co."""
    hr = hr or {}
    if name in GROUP_VN:
        if value is not None:
            full = hr.get(f"cat__{name}_{value}")
            if full:
                return full
            return f"{GROUP_VN[name]} ({value})"
        return GROUP_VN[name]
    return hr.get(f"num__{name}", name)
