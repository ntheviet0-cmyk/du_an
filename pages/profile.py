"""3. Ho so nhan vien — chon mau co san hoac nhap 3 nhom, cham diem, mo phong."""
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ui.components import action_card, badge, band_of, chips, footer_note, insight, page_header, warn
from ui.data import CAT, dept_income_median, load_meta, load_tabular
from ui.vocab import (SCALE_4, SCALE_EDU, SCALE_LEVEL, SCALE_PERF, SCALE_STOCK,
                      hr_dict, risk_sentence, scale_options, scale_value, suggest_actions)

page_header("🔍 Hồ sơ nhân viên", "Chấm điểm 1 người và thử ngay cách giữ họ lại.")

df, pre, cat, num = load_tabular()
meta = load_meta()
thr = float(meta.get("thr_graph", 0.8))
from graph_shap_lib import (format_group_value, format_shap_points, get_bundle,
                            graph_predict_proba, group_label, grouped_shap,
                            plot_grouped_shap)
bundle = get_bundle()

# ---- dien nhanh ho so mau ----
st.markdown("**Điền nhanh hồ sơ mẫu:**")
p1, p2, p3 = st.columns(3)
PRESETS = {
    "sales_ot": dict(Age=29, Gender="Male", MaritalStatus="Single", DistanceFromHome=12, Education=3,
                     EducationField="Marketing", Department="Sales", JobRole="Sales Representative", JobLevel=1,
                     BusinessTravel="Travel_Frequently", MonthlyIncome=4000, YearsAtCompany=2, TotalWorkingYears=5,
                     NumCompaniesWorked=2, OverTime="Yes", WorkLifeBalance=2, JobSatisfaction=2,
                     EnvironmentSatisfaction=2, StockOptionLevel=0, JobInvolvement=2, RelationshipSatisfaction=2,
                     TrainingTimesLastYear=1, DailyRate=800, HourlyRate=65, MonthlyRate=12000, PercentSalaryHike=13,
                     PerformanceRating=3, YearsInCurrentRole=2, YearsSinceLastPromotion=2, YearsWithCurrManager=1),
    "lab_new": dict(Age=26, Gender="Female", MaritalStatus="Single", DistanceFromHome=8, Education=3,
                    EducationField="Life Sciences", Department="Research & Development", JobRole="Laboratory Technician",
                    JobLevel=1, BusinessTravel="Travel_Rarely", MonthlyIncome=3500, YearsAtCompany=1, TotalWorkingYears=2,
                    NumCompaniesWorked=1, OverTime="Yes", WorkLifeBalance=2, JobSatisfaction=3,
                    EnvironmentSatisfaction=3, StockOptionLevel=0, JobInvolvement=3, RelationshipSatisfaction=3,
                    TrainingTimesLastYear=2, DailyRate=800, HourlyRate=65, MonthlyRate=12000, PercentSalaryHike=14,
                    PerformanceRating=3, YearsInCurrentRole=1, YearsSinceLastPromotion=1, YearsWithCurrManager=1),
    "mgr_ok": dict(Age=42, Gender="Male", MaritalStatus="Married", DistanceFromHome=5, Education=4,
                   EducationField="Life Sciences", Department="Research & Development", JobRole="Manager", JobLevel=4,
                   BusinessTravel="Travel_Rarely", MonthlyIncome=12000, YearsAtCompany=10, TotalWorkingYears=18,
                   NumCompaniesWorked=1, OverTime="No", WorkLifeBalance=3, JobSatisfaction=4,
                   EnvironmentSatisfaction=4, StockOptionLevel=2, JobInvolvement=4, RelationshipSatisfaction=4,
                   TrainingTimesLastYear=3, DailyRate=800, HourlyRate=65, MonthlyRate=12000, PercentSalaryHike=15,
                   PerformanceRating=4, YearsInCurrentRole=5, YearsSinceLastPromotion=1, YearsWithCurrManager=6),
}
if p1.button("👔 Sales làm thêm nhiều"):
    st.session_state["prefill"] = PRESETS["sales_ot"]; st.rerun()
if p2.button("🧪 Kỹ thuật viên mới vào"):
    st.session_state["prefill"] = PRESETS["lab_new"]; st.rerun()
if p3.button("🧑‍💼 Quản lý ổn định"):
    st.session_state["prefill"] = PRESETS["mgr_ok"]; st.rerun()

P = st.session_state.pop("prefill", {}) if "prefill" in st.session_state else {}
def dv(key, default):
    return P.get(key, default)

def sbox(label, options, key, default, fmt=None, tip=None):
    try:
        idx = list(options).index(dv(key, default))
    except ValueError:
        idx = 0
    kw = dict(help=tip)
    if fmt is not None:
        kw["format_func"] = fmt
    return st.selectbox(label, options, index=idx, **kw)

def sc(label, scale, key, default, tip=None):
    opts = scale_options(scale)
    d = int(dv(key, default))
    idx = list(scale.keys()).index(d) if d in scale else 0
    return scale_value(st.selectbox(label, opts, index=idx, help=tip))

YESNO = {"Yes": "Có", "Không": "Không", "No": "Không"}
GENDER = {"Male": "Nam", "Female": "Nữ"}
MARITAL = {"Single": "Độc thân", "Married": "Đã kết hôn", "Divorced": "Ly hôn"}
TRAVEL = {"Non-Travel": "Không đi", "Travel_Rarely": "Thỉnh thoảng", "Travel_Frequently": "Thường xuyên"}

t1, t2, t3 = st.tabs(["👤 Cá nhân", "💼 Công việc", "🌱 Mức hài lòng"])
with t1:
    c1, c2, c3 = st.columns(3)
    Age = c1.slider("Tuổi", 18, 60, int(dv("Age", 35)), help="Tuổi của nhân viên")
    Gender = c2.selectbox("Giới tính", ["Male", "Female"],
                          index=["Male", "Female"].index(dv("Gender", "Male")) if dv("Gender", "Male") in ["Male", "Female"] else 0,
                          format_func=GENDER.get)
    MaritalStatus = c3.selectbox("Hôn nhân", ["Single", "Married", "Divorced"],
                                 index=["Single", "Married", "Divorced"].index(dv("MaritalStatus", "Single")) if dv("MaritalStatus", "Single") in ["Single", "Married", "Divorced"] else 0,
                                 format_func=MARITAL.get)
    c4, c5 = st.columns(2)
    DistanceFromHome = c4.slider("Khoảng cách tới công ty (km)", 1, 29, int(dv("DistanceFromHome", 5)))
    Education = sc("Trình độ học vấn", SCALE_EDU, "Education", 3)
    EducationField = sbox("Ngành học", ["Human Resources", "Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"],
                          "EducationField", "Life Sciences")
with t2:
    c1, c2, c3 = st.columns(3)
    Department = sbox("Phòng ban", ["Sales", "Research & Development", "Human Resources"], "Department", "Sales")
    JobRole = sbox("Chức danh", ["Health Care Representative", "Human Resources", "Laboratory Technician", "Manager",
                                 "Manufacturing Director", "Research Director", "Research Scientist",
                                 "Sales Executive", "Sales Representative"], "JobRole", "Sales Executive")
    JobLevel = sc("Cấp bậc", SCALE_LEVEL, "JobLevel", 2, "Cấp bậc càng thấp, nguy cơ nghỉ càng cao")
    c1, c2, c3 = st.columns(3)
    BusinessTravel = c1.selectbox("Đi công tác", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
                                  index=["Non-Travel", "Travel_Rarely", "Travel_Frequently"].index(dv("BusinessTravel", "Travel_Rarely")) if dv("BusinessTravel", "Travel_Rarely") in ["Non-Travel", "Travel_Rarely", "Travel_Frequently"] else 1,
                                  format_func=TRAVEL.get)
    MonthlyIncome = c2.slider("Thu nhập tháng (USD)", 1000, 20000,
                              int(dv("MonthlyIncome", 5000) if dv("MonthlyIncome", 5000) < 25000 else 5000),
                              help="Theo đơn vị dữ liệu gốc để mô hình chấm đúng")
    YearsAtCompany = c3.slider("Thâm niên ở công ty (năm)", 0, 40, int(dv("YearsAtCompany", 6)))
    c1, c2 = st.columns(2)
    TotalWorkingYears = c1.slider("Tổng kinh nghiệm (năm)", 0, 40, int(dv("TotalWorkingYears", 10)))
    NumCompaniesWorked = c2.slider("Số công ty đã từng làm", 0, 9, int(dv("NumCompaniesWorked", 1)))
    OverTime = c3.selectbox("Có làm thêm giờ không?", ["No", "Yes"],
                            index=["No", "Yes"].index(dv("OverTime", "No")) if dv("OverTime", "No") in ["No", "Yes"] else 0,
                            format_func=YESNO.get, help="Yếu tố ảnh hưởng mạnh nhất tới nguy cơ nghỉ")
    c1, c2 = st.columns(2)
    YearsInCurrentRole = c1.slider("Số năm ở vai trò hiện tại", 0, 18, int(dv("YearsInCurrentRole", 4)))
    YearsSinceLastPromotion = c2.slider("Số năm từ lần thăng tiến cuối", 0, 15, int(dv("YearsSinceLastPromotion", 2)))
    YearsWithCurrManager = st.slider("Số năm làm với quản lý hiện tại", 0, 17, int(dv("YearsWithCurrManager", 3)))
    PerformanceRating = sc("Đánh giá hiệu suất", SCALE_PERF, "PerformanceRating", 3)
    with st.expander("⚙️ Thông số nâng cao (đã điền sẵn, thường không cần sửa)"):
        a1, a2 = st.columns(2)
        DailyRate = a1.slider("DailyRate", 100, 1500, int(dv("DailyRate", 800)))
        HourlyRate = a2.slider("HourlyRate", 30, 100, int(dv("HourlyRate", 65)))
        b1, b2 = st.columns(2)
        MonthlyRate = b1.slider("MonthlyRate", 2000, 27000, int(dv("MonthlyRate", 12000)))
        PercentSalaryHike = b2.slider("% tăng lương gần nhất", 11, 25, int(dv("PercentSalaryHike", 15)))
with t3:
    st.caption("Chọn theo cảm nhận: Rất thấp · Thấp · Cao · Rất cao")
    c1, c2, c3 = st.columns(3)
    JobSatisfaction = sc("Hài lòng với công việc", SCALE_4, "JobSatisfaction", 3)
    EnvironmentSatisfaction = sc("Hài lòng với môi trường", SCALE_4, "EnvironmentSatisfaction", 3)
    WorkLifeBalance = sc("Cân bằng cuộc sống", SCALE_4, "WorkLifeBalance", 3)
    c1, c2, c3 = st.columns(3)
    JobInvolvement = sc("Mức gắn kết", SCALE_4, "JobInvolvement", 3)
    RelationshipSatisfaction = sc("Quan hệ đồng nghiệp", SCALE_4, "RelationshipSatisfaction", 3)
    StockOptionLevel = sc("Cổ phiếu thưởng", SCALE_STOCK, "StockOptionLevel", 1)
    TrainingTimesLastYear = st.slider("Số lượt đào tạo năm qua", 0, 6, int(dv("TrainingTimesLastYear", 2)))

if st.button("🚀 Chấm điểm rủi ro", type="primary", use_container_width=True):
    row = dict(Age=Age, Gender=Gender, MaritalStatus=MaritalStatus, DistanceFromHome=DistanceFromHome,
               Education=Education, Department=Department, JobRole=JobRole, JobLevel=JobLevel,
               BusinessTravel=BusinessTravel, MonthlyIncome=MonthlyIncome,
               YearsAtCompany=YearsAtCompany, TotalWorkingYears=TotalWorkingYears,
               NumCompaniesWorked=NumCompaniesWorked, OverTime=OverTime, WorkLifeBalance=WorkLifeBalance,
               JobSatisfaction=JobSatisfaction, EnvironmentSatisfaction=EnvironmentSatisfaction,
               StockOptionLevel=StockOptionLevel, JobInvolvement=JobInvolvement,
               RelationshipSatisfaction=RelationshipSatisfaction, TrainingTimesLastYear=TrainingTimesLastYear,
               EducationField=EducationField, DailyRate=DailyRate, HourlyRate=HourlyRate, MonthlyRate=MonthlyRate,
               PercentSalaryHike=PercentSalaryHike, PerformanceRating=PerformanceRating,
               YearsInCurrentRole=YearsInCurrentRole, YearsSinceLastPromotion=YearsSinceLastPromotion,
               YearsWithCurrManager=YearsWithCurrManager)
    st.session_state["last_row"] = row
    row_df = pd.DataFrame([row])[cat + num]
    Xt = pre.transform(row_df)
    Xt = Xt.toarray().astype("float32") if hasattr(Xt, "toarray") else np.asarray(Xt, "float32")
    st.session_state["last_proba"] = float(graph_predict_proba(Xt)[0])
    st.session_state["last_Xt"] = Xt

if "last_proba" in st.session_state:
    proba = st.session_state["last_proba"]
    row = st.session_state["last_row"]
    band = band_of(proba, thr)
    st.markdown(f"## Kết quả {badge(band)}")
    g1, g2 = st.columns([1, 2])
    with g1:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=proba * 100,
            number={"suffix": "%"}, gauge={"axis": {"range": [0, 100]},
            "bar": {"color": "#dc2626" if band == "red" else ("#d97706" if band == "amber" else "#16a34a")},
            "steps": [{"range": [0, thr * 60], "color": "#dcfce7"}, {"range": [thr * 60, thr * 100], "color": "#fef3c7"},
                      {"range": [thr * 100, 100], "color": "#fee2e2"}]}))
        fig.update_layout(height=260, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        chips([f"{row['Age']} tuổi", row["Department"], row["JobRole"], f"Thâm niên {row['YearsAtCompany']} năm"])
        row["_med_income_dept"] = dept_income_median().get(row["Department"], 0)
        for t, reason in suggest_actions(row):
            action_card(t, reason)

    st.markdown('<div class="hr-section">🔮 Thử ngay: đổi 1 chính sách thì sao?</div>', unsafe_allow_html=True)
    if row["OverTime"] == "Yes":
        row2 = dict(row); row2["OverTime"] = "No"
        Xt2 = pre.transform(pd.DataFrame([row2])[cat + num])
        Xt2 = Xt2.toarray().astype("float32") if hasattr(Xt2, "toarray") else np.asarray(Xt2, "float32")
        p2 = float(graph_predict_proba(Xt2)[0])
        st.success(f"Tắt làm thêm giờ cho người này: nguy cơ **{proba*100:.0f}% → {p2*100:.0f}%** "
                   f"(giảm {max(proba - p2, 0)*100:.0f} điểm %).")
    else:
        row2 = dict(row); row2["OverTime"] = "Yes"
        Xt2 = pre.transform(pd.DataFrame([row2])[cat + num])
        Xt2 = Xt2.toarray().astype("float32") if hasattr(Xt2, "toarray") else np.asarray(Xt2, "float32")
        p2 = float(graph_predict_proba(Xt2)[0])
        warn(f"Nếu bắt đầu làm thêm giờ: nguy cơ có thể tăng **{proba*100:.0f}% → {p2*100:.0f}%**. "
             f"Hiện tại người này không làm thêm — đó là điểm tốt, hãy giữ.")

    with st.expander("🧩 Vì sao mô hình chấm vậy? (phân tích chi tiết)"):
        st.caption("Mất khoảng 30 giây. Kết quả hiện bằng lời + biểu đồ.")
        if st.button("Chạy phân tích", key="pf_shap"):
            import shap
            import matplotlib.pyplot as plt
            with st.spinner("Đang phân tích (khoảng 30 giây)..."):
                Xt = st.session_state["last_Xt"]
                rng = np.random.RandomState(0)
                bg = bundle["X_train"][rng.choice(len(bundle["X_train"]), size=15, replace=False)]
                ex = shap.KernelExplainer(graph_predict_proba, bg)
                sv = np.asarray(ex.shap_values(Xt, nsamples=100)).reshape(-1)
                ev = float(np.atleast_1d(ex.expected_value)[0])
                names = bundle["feat_names"]
                hr = hr_dict()
                gn, gv, gd = grouped_shap(sv, Xt[0], names, raw_row=row)
                order = np.argsort(gv)
                up_idx = [j for j in order[::-1] if gv[j] > 0][:3]
                dn_idx = [j for j in order if gv[j] < 0][:2]
                ups = [group_label(gn[j], hr, gd[j]) for j in up_idx]
                dns = [group_label(gn[j], hr, gd[j]) for j in dn_idx]
                if not ups:
                    ups = ["không có yếu tố làm tăng dự báo"]
                st.markdown("**" + risk_sentence(proba, ups) + "**")
                st.markdown("Điểm **giảm** rủi ro: " +
                            (", ".join(dns) if dns else "không có yếu tố làm giảm dự báo") + ".")
                if any(k in " ".join(ups + dns).lower() for k in ["giới tính", "hôn nhân", "độc thân", "tuổi"]):
                    warn("Yếu tố nhân khẩu học lọt top — chỉ dùng để thiết kế chính sách hỗ trợ, không phân biệt đối xử.")
                fig2, ax2 = plt.subplots(figsize=(9, 5))
                shown_idx = plot_grouped_shap(
                    ax2, gv, gn, gd, hr=hr, base_value=ev, prediction=proba)
                plt.tight_layout()
                st.pyplot(fig2)
                rows = []
                for rank, j in enumerate(shown_idx, 1):
                    rows.append({
                        "#": rank,
                        "Yếu tố HR": group_label(gn[j], hr, gd[j]),
                        "Giá trị": format_group_value(gn[j], gd[j]),
                        "Chiều tác động": "Tăng" if gv[j] > 0 else "Giảm" if gv[j] < 0 else "Không đổi",
                        "Đóng góp": format_shap_points(gv[j]),
                    })
                st.markdown("**Toàn bộ yếu tố góp phần vào dự báo**")
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                st.caption("Đã gộp 51 cột sau mã hóa thành các nhóm HR gốc có đóng góp khác 0 và sắp xếp theo |SHAP|. "
                           "Đây là đóng góp vào dự báo của mô hình, không phải kết luận nhân quả.")

footer_note()
