"""Thu LLM (noi bo) - trang test sinh loi giai thich bang mo hinh ngon ngu.

Chi dung noi bo de thu prompt + API (mac dinh: Google AI Studio / Gemini Flash).
Key chi nam trong phien chay, khong ghi ra file. Chi gui ma NV + yeu to dang
chu, khong gui ho so tho.
"""
import pandas as pd
import streamlit as st

from ui.components import band_of
from ui.data import CAT, build_risk_list, get_test_frame, load_tabular
from ui.llm_probe import (DEFAULT_RULES, LEVEL_VN, audit_response, build_prompt,
                          call_llm, check_grounding, sensitive_in_factors,
                          tree_shap_for_row, vn)

st.title("Thu LLM (noi bo)")
st.caption("Trang thu nghiem cach sinh loi giai thich tu SHAP. Khong phai trang cho HR su dung hang ngay.")
st.warning("Trang noi bo: API key chi nam trong phien chay nay, khong luu ra file. "
           "He thong chi gui ma nhan vien + ten yeu to dang chu, khong gui ho so chi tiet.")

DEPT_VN = {"Sales": "Kinh doanh",
           "Research & Development": "Nghien cuu va Phat trien",
           "Human Resources": "Nhan su"}

PRESETS = {
    "sales": {"nhan": "Nhan vien kinh doanh hay lam them",
              "Age": 29, "Gender": "Male", "MaritalStatus": "Single",
              "DistanceFromHome": 12, "Education": 3, "Department": "Sales",
              "JobRole": "Sales Representative", "MonthlyIncome": 4000,
              "YearsAtCompany": 2, "OverTime": "Yes", "WorkLifeBalance": 2,
              "JobSatisfaction": 2, "EnvironmentSatisfaction": 2,
              "StockOptionLevel": 0, "JobInvolvement": 2,
              "YearsInCurrentRole": 2, "YearsSinceLastPromotion": 2},
    "lab": {"nhan": "Ky thuat vien moi vao",
            "Age": 26, "Gender": "Female", "MaritalStatus": "Single",
            "DistanceFromHome": 8, "Education": 3,
            "Department": "Research & Development",
            "JobRole": "Laboratory Technician", "MonthlyIncome": 3500,
            "YearsAtCompany": 1, "OverTime": "Yes", "WorkLifeBalance": 2,
            "JobSatisfaction": 3, "EnvironmentSatisfaction": 3,
            "StockOptionLevel": 0, "JobInvolvement": 3,
            "YearsInCurrentRole": 1, "YearsSinceLastPromotion": 1},
    "mgr": {"nhan": "Quan ly da on dinh",
            "Age": 42, "Gender": "Male", "MaritalStatus": "Married",
            "DistanceFromHome": 5, "Education": 4,
            "Department": "Research & Development", "JobRole": "Manager",
            "MonthlyIncome": 12000, "YearsAtCompany": 10, "OverTime": "No",
            "WorkLifeBalance": 3, "JobSatisfaction": 4,
            "EnvironmentSatisfaction": 4, "StockOptionLevel": 2,
            "JobInvolvement": 4, "YearsInCurrentRole": 5,
            "YearsSinceLastPromotion": 1},
}


def _level_of(proba):
    return LEVEL_VN[band_of(float(proba), 0.8)]


# ---------- Khoi 1: key + model ----------
st.subheader("1. Key va model")
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    api_key = st.text_input("Dan API key (Google AI Studio)", type="password",
                            help="Key chi nam trong phien chay nay.")
with c2:
    model = st.selectbox("Model", ["gemini-2.5-flash", "gemini-2.5-flash-lite",
                                  "gemini-2.0-flash"])
with c3:
    temperature = st.slider("Do phong tung", 0.0, 1.0, 0.3, 0.1,
                            help="Cang thap cang nhat quan. Thu nghiem nen de 0.0-0.3.")

# ---------- Khoi 2: chon ca ----------
st.subheader("2. Chon ca de thu")
df, pre, cat, num = load_tabular()
frame = get_test_frame()
source = st.radio("Nguon ca", ["14 ca mau co SHAP san",
                               "Ca muc cao (danh sach giu chan)",
                               "Nhap tay nhanh"],
                  horizontal=True)
ctx, factor_labels, src_note = None, [], ""
if source.startswith("14 ca"):
    cases = pd.read_csv("reports/ibm/graph_shap_cases.csv")
    labels = [f"Tinh huong {k + 1}: khoang {round(r['proba'] * 100)} tren 100 nguoi "
              f"giong vay muon nghi ({'da nghi that' if r['true_label'] == 1 else 'van o lai'})"
              for k, r in cases.iterrows()]
    pick = st.selectbox("Chon tinh huong (keo len xuong)", labels)
    k = labels.index(pick)
    r = cases.iloc[k]
    fr = frame.iloc[int(r["test_pos"])]
    ups = [vn(r[f"top_up_{i}"]) for i in (1, 2, 3)]
    dns = [vn(r[f"top_down_{i}"]) for i in (1, 2)]
    ctx = {"code": f"TH-{k + 1:02d}", "role": fr["JobRole"],
           "dept": DEPT_VN.get(fr["Department"], fr["Department"]),
           "level": _level_of(r["proba"]), "pct": round(r["proba"] * 100),
           "up": [(u, None) for u in ups], "down": [(d, None) for d in dns],
           "with_value": False}
    factor_labels = ups + dns
    src_note = "So SHAP co san cua GraphSAGE (14 ca mau)."
elif source.startswith("Ca muc"):
    risk = build_risk_list()
    reds = risk[risk["band"] == "red"].reset_index(drop=True)
    labels = [f"NV-{i + 1:04d} - {r['JobRole']} - Phong {DEPT_VN.get(r['Department'], r['Department'])}"
              for i, r in reds.iterrows()]
    pick = st.selectbox("Chon nguoi", labels)
    i = labels.index(pick)
    row = reds.iloc[i]
    with st.spinner("Dang tinh SHAP cay (duoi 1 giay)..."):
        t = tree_shap_for_row(row[CAT + num].to_frame().T)
    ctx = {"code": f"NV-{i + 1:04d}", "role": row["JobRole"],
           "dept": DEPT_VN.get(row["Department"], row["Department"]),
           "level": _level_of(t["proba"]), "pct": round(t["proba"] * 100),
           "up": t["up"], "down": t["down"], "with_value": True}
    factor_labels = [u for u, _ in t["up"]] + [d for d, _ in t["down"]]
    src_note = "SHAP cay tinh nhanh tu Random Forest (duoi 1 giay)."
else:
    pk = st.radio("Vi du", [f"{v['nhan']}" for v in PRESETS.values()], horizontal=True)
    key = [k for k, v in PRESETS.items() if v["nhan"] == pk][0]
    base = df.iloc[0].copy()
    for col, val in PRESETS[key].items():
        if col != "nhan":
            base[col] = val
    with st.spinner("Dang tinh SHAP cay (duoi 1 giay)..."):
        t = tree_shap_for_row(base[CAT + num].to_frame().T)
    ctx = {"code": "Nhap-tay", "role": base["JobRole"],
           "dept": DEPT_VN.get(base["Department"], base["Department"]),
           "level": _level_of(t["proba"]), "pct": round(t["proba"] * 100),
           "up": t["up"], "down": t["down"], "with_value": True}
    factor_labels = [u for u, _ in t["up"]] + [d for d, _ in t["down"]]
    src_note = "SHAP cay tinh nhanh tu Random Forest (duoi 1 giay)."

st.info(f"Ma {ctx['code']} - {ctx['role']} - Phong {ctx['dept']} - "
        f"Muc {ctx['level'].lower()} (khoang {ctx['pct']} tren 100 nguoi giong vay muon nghi). "
        f"Nguon so: {src_note}")
st.markdown("**Diem lam tang kha nang nghi:** " + "; ".join(u for u, _ in ctx["up"]))
st.markdown("**Diem dang giu nguoi o lai:** " + "; ".join(d for d, _ in ctx["down"]))

# ---------- Khoi 3: noi quy + xem prompt ----------
st.subheader("3. Noi quy va xem truoc prompt")
rules = st.text_area("Noi quy cong ty (sua truc tiep de thu prompt)",
                     value=DEFAULT_RULES, height=150)
system, user = build_prompt(ctx, rules)
with st.expander("Xem toan bo prompt se gui"):
    st.markdown("**Phan dan vai tro:**")
    st.code(system)
    st.markdown("**Phan context + yeu cau:**")
    st.code(user)

# ---------- Khoi 4: chay ----------
st.subheader("4. Chay va do gio")
if st.button("Sinh loi giai thich", type="primary"):
    with st.spinner("Dang goi mo hinh ngon ngu..."):
        res = call_llm(api_key, model, system, user, temperature=temperature)
    st.session_state["llm_res"] = res
    st.session_state["llm_factors"] = factor_labels
res = st.session_state.get("llm_res")
if res:
    if res["ok"]:
        st.success(f"Xong trong {res['elapsed']:.1f} giay. Ly do dung: {res.get('stop', 'hoan tat')}"
                   + (" (da tu goi lai 1 lan)." if res.get("retried") else "."))
        if res.get("truncated"):
            st.warning("Cau tra loi van chua day du sau khi goi lai. Hay rut gon noi quy hoac giam yeu cau do dai roi bam lai.")
        st.markdown(res["text"])
    else:
        st.error(res["error"])
        st.caption("Bam nut tren de thu lai.")

# ---------- Khoi 5: kiem tra theo checklist cua prompt ----------
st.subheader("5. Kiem tra theo checklist cua prompt")
factors = st.session_state.get("llm_factors", factor_labels)
if res and res.get("ok"):
    for _name, _ok, _detail in audit_response(res["text"], factors):
        if _ok:
            st.success(f"{_name}: {_detail}")
        else:
            st.error(f"{_name}: {_detail}")
    sens = sensitive_in_factors(factors)
    if sens:
        st.warning("Luu y cong bang: top yeu to co dinh yeu to nhan khau (" + "; ".join(sens) +
                   ") - chi dung de thiet ke chinh sach ho tro, khong dung de phan biet doi xu.")
else:
    st.caption("Chay xong khoi 4 thi ket qua kiem tra hien o day.")
