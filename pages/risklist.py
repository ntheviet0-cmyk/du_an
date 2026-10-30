"""2. Uu tien giu chan — danh sach can thiep hang tuan: loc, xem, xuat CSV."""
import pandas as pd
import streamlit as st

from ui.components import action_card, badge, chips, empty_state, footer_note, page_header, summary_bar
from ui.data import build_risk_list, dept_income_median, load_meta
from ui.vocab import suggest_actions

page_header("🚨 Ưu tiên giữ chân",
            "Ai cần gặp ngay tuần này — mỗi dòng đã ghi sẵn lý do, bấm vào để xem việc cần làm.",
            cta=("📊 Xem bức tranh chung →", "pages/executive.py"))

risk = build_risk_list()
meta = load_meta()
thr = float(meta.get("thr_graph", 0.8))
med = dept_income_median()

f1, f2, f3, f4 = st.columns([2, 2, 1.5, 2])
with f1:
    depts = st.multiselect("Phòng ban", sorted(risk["Department"].unique()), default=list(sorted(risk["Department"].unique())))
with f2:
    bands = st.multiselect("Mức rủi ro", ["red", "amber", "green"], default=["red", "amber"],
                           format_func={"red": "🔴 Ưu tiên giữ", "amber": "🟡 Theo dõi", "green": "🟢 Ổn định"}.get)
with f3:
    ot = st.selectbox("Làm thêm", ["Tất cả", "Có", "Không"])
with f4:
    q = st.text_input("🔎 Tìm theo chức danh", placeholder="vd: Sales, Lab...")

view = risk[risk["Department"].isin(depts) & risk["band"].isin(bands)]
if ot != "Tất cả":
    view = view[view["OverTime"] == ("Yes" if ot == "Có" else "No")]
if q.strip():
    view = view[view["JobRole"].str.contains(q.strip(), case=False)]
view = view.reset_index(drop=True)

if len(view) == 0:
    empty_state("Không ai khớp bộ lọc — thử nới phòng ban hoặc mức rủi ro.")
else:
    summary_bar([
        ("🔴 Ưu tiên giữ", int((view["band"] == "red").sum())),
        ("🟡 Theo dõi", int((view["band"] == "amber").sum())),
        ("🟢 Ổn định", int((view["band"] == "green").sum())),
    ])
    st.download_button("⬇️ Xuất CSV mang đi họp",
                       view.drop(columns=["_med_income_dept"]).to_csv(index=False).encode("utf-8-sig"),
                       file_name="uu_tien_giu_chan.csv", mime="text/csv")

    PAGE = 20
    pages = max(1, (len(view) + PAGE - 1) // PAGE)
    pg = st.number_input(f"Trang (1–{pages})", 1, pages, 1, key="rl_page") if pages > 1 else 1
    chunk = view.iloc[(pg - 1) * PAGE: pg * PAGE]
    st.caption(f"Hiển thị {len(chunk)}/{len(view)} người.")

    for i, r in chunk.iterrows():
        row = r.to_dict()
        row["_med_income_dept"] = med.get(r["Department"], 0)
        acts = suggest_actions(row)
        why = acts[0][0] if acts else "Đang ổn định"
        dot = "🔴" if r["band"] == "red" else ("🟡" if r["band"] == "amber" else "🟢")
        with st.expander(f"{dot} {r['JobRole']} · {r['Department']} · {r['proba'] * 100:.0f}% — {why}"):
            chips([f"{r['Age']} tuổi", r["Department"], r["JobRole"],
                   f"Thâm niên {r['YearsAtCompany']} năm",
                   f"Thu nhập {r['MonthlyIncome']:,.0f} USD".replace(",", "."),
                   "Làm thêm giờ" if r["OverTime"] == "Yes" else "Không làm thêm"])
            if acts:
                for t, reason in acts:
                    action_card(t, reason)
            else:
                st.markdown("Không có dấu hiệu bất thường nổi bật — duy trì trao đổi định kỳ.")
            if st.button("📋 Mở hồ sơ đầy đủ", key=f"open_{r['_test_pos']}"):
                st.session_state["prefill"] = {k: r[k] for k in
                    ["Age", "Gender", "MaritalStatus", "DistanceFromHome", "Education", "Department",
                     "JobRole", "JobLevel", "BusinessTravel", "MonthlyIncome", "YearsAtCompany",
                     "TotalWorkingYears", "NumCompaniesWorked", "OverTime", "WorkLifeBalance",
                     "JobSatisfaction", "EnvironmentSatisfaction", "StockOptionLevel", "JobInvolvement",
                     "RelationshipSatisfaction", "TrainingTimesLastYear", "EducationField", "DailyRate",
                     "HourlyRate", "MonthlyRate", "PercentSalaryHike", "PerformanceRating",
                     "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"]}
                st.switch_page("pages/profile.py")

footer_note()
