"""
app_ibm.py - Dashboard HR: Dự đoán nguy cơ nghỉ việc (IBM HR Analytics)
Trang: Tong quan | Du doan ca nhan | Giai thich SHAP | Nhan qua & What-If | So sanh
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="HR Attrition AI", page_icon="🧑‍💼", layout="wide")

# ----------------------------- styles -----------------------------
st.markdown("""
<style>
.big-num {font-size:34px; font-weight:700; line-height:1.1;}
.kpi-card {background:#f5f7fa; border-radius:12px; padding:16px; text-align:center; box-shadow:0 1px 3px rgba(0,0,0,.08);}
.badge {padding:6px 14px; border-radius:20px; color:#fff; font-weight:700; display:inline-block;}
.badge-red{background:#e74c3c;} .badge-orange{background:#e67e22;} .badge-green{background:#27ae60;}
.section-title{font-size:20px; font-weight:700; margin:10px 0; color:#2c3e50;}
</style>""", unsafe_allow_html=True)

ART = Path("models/ibm")
REP = Path("reports/ibm")

@st.cache_resource
def load_artifacts():
    pre = joblib.load(ART / "preprocessor.joblib")
    rf = joblib.load(ART / "random_forest.joblib")
    xgbm = joblib.load(ART / "xgboost.joblib")
    meta = joblib.load(ART / "meta.joblib")
    return pre, rf, xgbm, meta

pre, rf, xgbm, meta = load_artifacts()
feat_names = meta["feat_names"]
CAT, NUM = meta["cat"], meta["num"]
thr_rf, thr_xgb = meta["thr_rf"], meta["thr_xgb"]
metrics = meta["metrics"]

causal = json.loads((REP / "ibm_results.json").read_text(encoding="utf-8"))["causal"]
ate = causal["ate"]

# ----------------------------- sidebar -----------------------------
st.sidebar.title("🧑‍💼 HR Attrition AI")
st.sidebar.caption("Hỗ trợ quyết định tuyển dụng & giữ chân")
page = st.sidebar.radio("Chọn trang", [
    "1. Tổng quan",
    "2. Dự đoán cá nhân",
    "3. Giải thích SHAP",
    "4. Nhân quả & What-If",
    "5. So sánh mô hình",
])
model_choice = st.sidebar.selectbox("Mô hình dùng dự đoán", ["Random Forest", "XGBoost"])
model = rf if model_choice == "Random Forest" else xgbm
thr = thr_rf if model_choice == "Random Forest" else thr_xgb

def predict_proba(row_df):
    Xt = pre.transform(row_df)
    return model.predict_proba(Xt)[:, 1][0]

# ===================================================================
# PAGE 1 : TONG QUAN
# ===================================================================
if page.startswith("1"):
    st.title("📊 Tổng quan hệ thống")
    col1, col2, col3, col4 = st.columns(4)
    n = 1470
    with col1:
        st.markdown('<div class="kpi-card"><div class="big-num">1,470</div>Nhân viên</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="kpi-card"><div class="big-num">16.1%</div>Tỷ lệ nghỉ việc</div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="kpi-card"><div class="big-num">{metrics["RandomForest"]["roc_auc"]:.3f}</div>ROC-AUC (RF)</div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="kpi-card"><div class="big-num">+{ate*100:.1f}%</div>ATE (OverTime)</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Hiệu năng mô hình</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.image(str(REP / "roc_curves.png"), use_container_width=True)
    with c2:
        st.image(str(REP / "metrics_bar.png"), use_container_width=True)

    st.markdown('<div class="section-title">Phát hiện chính</div>', unsafe_allow_html=True)
    st.info("""
    • **Làm thêm giờ** là yếu tố mạnh nhất: nhân viên làm thêm có tỷ lệ nghỉ **30.5%** so với **10.4%** người không làm thêm.\n
    • **Quyền chọn cổ phiếu, thu nhập, cấp bậc, độc thân** là các yếu tố quan trọng tiếp theo (SHAP).\n
    • Phân tích nhân quả: giảm làm thêm giờ làm **giảm 21.1 điểm %** xác suất nghỉ việc (đã qua kiểm định bác bỏ).\n
    • Nghỉ việc tập trung nhóm **0–2 năm** và **dưới 25 tuổi** → cần chương trình giữ chân sớm.
    """)

# ===================================================================
# PAGE 2 : DU DOAN CA NHAN
# ===================================================================
elif page.startswith("2"):
    st.title("🔍 Dự đoán nguy cơ nghỉ việc cá nhân")
    st.caption("Nhập thông tin nhân viên → hệ thống chấm điểm rủi ro và giải thích.")

    with st.form("emp_form"):
        st.markdown("**👤 Thông tin cá nhân**")
        c1, c2, c3 = st.columns(3)
        Age = c1.slider("Tuổi", 18, 60, 35)
        Gender = c2.selectbox("Giới tính", ["Male", "Female"])
        MaritalStatus = c3.selectbox("Tình trạng hôn nhân", ["Single", "Married", "Divorced"])
        c4, c5 = st.columns(2)
        DistanceFromHome = c4.slider("Khoảng cách tới công ty", 1, 29, 5)
        Education = c5.slider("Trình độ (1-5)", 1, 5, 3)

        st.markdown("**💼 Công việc & thu nhập**")
        c1, c2, c3 = st.columns(3)
        Department = c1.selectbox("Phòng ban", ["Sales", "Research & Development", "Human Resources"])
        JobRole = c2.selectbox("Chức danh", ["Health Care Representative", "Human Resources", "Laboratory Technician",
                                             "Manager", "Manufacturing Director", "Research Director",
                                             "Research Scientist", "Sales Executive", "Sales Representative"])
        JobLevel = c3.slider("Cấp bậc (1-5)", 1, 5, 2)
        c1, c2, c3 = st.columns(3)
        BusinessTravel = c1.selectbox("Đi công tác", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
        MonthlyIncome = c2.slider("Thu nhập tháng", 1000, 20000, 5000)
        YearsAtCompany = c3.slider("Thâm niên (năm)", 0, 40, 6)
        c1, c2 = st.columns(2)
        TotalWorkingYears = c1.slider("Tổng năm kinh nghiệm", 0, 40, 10)
        NumCompaniesWorked = c2.slider("Số công ty đã làm", 0, 9, 1)

        st.markdown("**🌱 Môi trường & gắn kết**")
        c1, c2, c3 = st.columns(3)
        OverTime = c1.selectbox("Làm thêm giờ", ["No", "Yes"])
        WorkLifeBalance = c2.slider("Cân bằng CV (1-4)", 1, 4, 3)
        JobSatisfaction = c3.slider("Hài lòng CV (1-4)", 1, 4, 3)
        c1, c2, c3 = st.columns(3)
        EnvironmentSatisfaction = c1.slider("Hài lòng môi trường (1-4)", 1, 4, 3)
        StockOptionLevel = c2.slider("Quyền chọn CP (0-3)", 0, 3, 1)
        JobInvolvement = c3.slider("Mức độ tham gia (1-4)", 1, 4, 3)
        c1, c2, c3 = st.columns(3)
        RelationshipSatisfaction = c1.slider("Hài lòng quan hệ (1-4)", 1, 4, 3)
        TrainingTimesLastYear = c2.slider("Lượt đào tạo/năm", 0, 6, 2)
        EducationField = c3.selectbox("Ngành học", ["Human Resources", "Life Sciences", "Marketing",
                                                   "Medical", "Other", "Technical Degree"])
        c1, c2, c3 = st.columns(3)
        DailyRate = c1.slider("DailyRate", 100, 1500, 800)
        HourlyRate = c2.slider("HourlyRate", 1, 100, 50)
        MonthlyRate = c3.slider("MonthlyRate", 2000, 27000, 12000)
        c1, c2, c3 = st.columns(3)
        PercentSalaryHike = c1.slider("Tăng lương %", 11, 25, 15)
        PerformanceRating = c2.slider("Đánh giá hiệu suất (3-4)", 3, 4, 3)
        YearsInCurrentRole = c3.slider("Năm ở vai trò hiện tại", 0, 18, 4)
        c1, c2 = st.columns(2)
        YearsSinceLastPromotion = c1.slider("Năm từ lần thăng tiến cuối", 0, 15, 2)
        YearsWithCurrManager = c2.slider("Năm với quản lý hiện tại", 0, 17, 3)

        submitted = st.form_submit_button("🚀 Chấm điểm rủi ro", use_container_width=True)

    if submitted:
        row = {k: v for k, v in [
            ("Age", Age), ("Gender", Gender), ("MaritalStatus", MaritalStatus),
            ("DistanceFromHome", DistanceFromHome), ("Education", Education),
            ("Department", Department), ("JobRole", JobRole), ("JobLevel", JobLevel),
            ("BusinessTravel", BusinessTravel), ("MonthlyIncome", MonthlyIncome),
            ("YearsAtCompany", YearsAtCompany), ("TotalWorkingYears", TotalWorkingYears),
            ("NumCompaniesWorked", NumCompaniesWorked), ("OverTime", OverTime),
            ("WorkLifeBalance", WorkLifeBalance), ("JobSatisfaction", JobSatisfaction),
            ("EnvironmentSatisfaction", EnvironmentSatisfaction), ("StockOptionLevel", StockOptionLevel),
            ("JobInvolvement", JobInvolvement), ("RelationshipSatisfaction", RelationshipSatisfaction),
            ("TrainingTimesLastYear", TrainingTimesLastYear), ("EducationField", EducationField),
            ("DailyRate", DailyRate), ("HourlyRate", HourlyRate), ("MonthlyRate", MonthlyRate),
            ("PercentSalaryHike", PercentSalaryHike), ("PerformanceRating", PerformanceRating),
            ("YearsInCurrentRole", YearsInCurrentRole), ("YearsSinceLastPromotion", YearsSinceLastPromotion),
            ("YearsWithCurrManager", YearsWithCurrManager),
        ]}
        row_df = pd.DataFrame([row])[CAT + NUM]
        proba = predict_proba(row_df)

        if proba >= thr:
            badge = f'<span class="badge badge-red">RỦI RO CAO</span>'
        elif proba >= thr * 0.6:
            badge = f'<span class="badge badge-orange">TRUNG BÌNH</span>'
        else:
            badge = f'<span class="badge badge-green">AN TOÀN</span>'

        st.markdown(f"### Kết quả: {badge}", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f'<div class="kpi-card"><div class="big-num">{proba*100:.1f}%</div>Xác suất nghỉ việc</div>', unsafe_allow_html=True)
            st.caption(f"Ngưỡng quyết định mô hình: {thr:.2f}")
        with col2:
            st.progress(float(proba))
            st.caption("Thanh xác suất (0 → 100%)")

        # SHAP waterfall
        st.markdown('<div class="section-title">🧩 Tại sao mô hình đưa ra kết quả này?</div>', unsafe_allow_html=True)
        clf = model.named_steps["clf"]
        explainer = shap.TreeExplainer(clf)
        Xt = pre.transform(row_df)
        sv = explainer.shap_values(Xt)
        if isinstance(sv, list):
            sv = sv[1]
        elif sv.ndim == 3:
            sv = sv[:, :, 1]
        else:
            sv = sv
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)) and len(np.atleast_1d(ev)) > 1:
            exp_val = ev[1]
        else:
            exp_val = ev
        expl = shap.Explanation(values=sv[0], base_values=exp_val,
                                data=Xt[0], feature_names=feat_names)
        fig, ax = plt.subplots(figsize=(9, 5))
        shap.plots.waterfall(expl, max_display=15, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Màu đỏ = đẩy tăng rủi ro, màu xanh = giảm rủi ro. Giá trị hiển thị là độ lệch so với trung bình.")

# ===================================================================
# PAGE 3 : SHAP
# ===================================================================
elif page.startswith("3"):
    st.title("🧩 Giải thích mô hình (SHAP)")
    st.caption("Tầm quan trọng đặc trưng toàn cục — giúp hiểu yếu tố nào thúc đẩy nghỉ việc.")
    m = st.radio("Chọn mô hình", ["Random Forest", "XGBoost"], horizontal=True)
    img = "shap_rf.png" if m == "Random Forest" else "shap_xgboost.png"
    st.image(str(REP / img), use_container_width=True)
    st.info("Mỗi điểm là một nhân viên; vị trí xa 0 = ảnh hưởng mạnh đến dự đoán. "
            "Màu đỏ = giá trị đặc trưng cao, xanh = thấp. OverTime, StockOptionLevel, Income, "
            "JobLevel, MaritalStatus consistently nằm top đầu.")

# ===================================================================
# PAGE 4 : NHAN QUA & WHAT-IF
# ===================================================================
elif page.startswith("4"):
    st.title("🔗 Phân tích nhân quả & Mô phỏng chính sách")
    st.markdown('<div class="section-title">Đồ thị nhân quả (Causal DAG)</div>', unsafe_allow_html=True)
    st.image(str(REP / "attrition_dag.png"), use_container_width=True)
    st.markdown(f"""
    **Treatment:** `OverTime` (làm thêm giờ) &nbsp; | &nbsp; **Outcome:** `Attrition` (nghỉ việc) &nbsp; | &nbsp; **Confounders:** {causal['n_confounders']} biến nền.
    """)
    col1, col2, col3 = st.columns(3)
    col1.metric("ATE (OverTime → Attrition)", f"+{ate*100:.1f} điểm %")
    col2.metric("Placebo ATE", f"{causal['placebo_ate']*100:.2f} điểm %", help="Kỳ vọng ≈ 0 → đạt")
    col3.metric("Random Common Cause ATE", f"{causal['random_common_cause_ate']*100:.1f} điểm %", help="Giữ nguyên → đạt")

    st.markdown('<div class="section-title">🧪 Mô phỏng What-If (cấp độ doanh nghiệp)</div>', unsafe_allow_html=True)
    st.caption("Ước lượng số nhân viên được 'cứu' nếu giảm tỷ lệ làm thêm giờ trong toàn công ty.")
    headcount = st.number_input("Tổng số nhân viên hiện tại", 100, 100000, 1470, step=50)
    ot_rate = st.slider("Tỷ lệ nhân viên đang làm thêm giờ (%)", 0, 100, 28)
    reduce = st.slider("Mức giảm làm thêm giờ nhờ chính sách (%)", 0, 100, 50)

    n_ot = headcount * ot_rate / 100
    n_shifted = n_ot * reduce / 100
    avoided = n_shifted * ate
    st.success(f"Ước tính giảm được **{avoided:.0f}** ca nghỉ việc "
               f"(trong {n_shifted:.0f} người được chuyển từ 'có làm thêm' sang 'không').")
    st.caption(f"Công thức: ATE({ate*100:.1f}%) × số người chuyển = {n_shifted:.0f} × {ate:.3f} ≈ {avoided:.0f}.")

    st.markdown('<div class="section-title">🧪 Mô phỏng What-If (cá nhân)</div>', unsafe_allow_html=True)
    ot_now = st.selectbox("Nhân viên hiện có làm thêm giờ?", ["Yes", "No"])
    if ot_now == "Yes":
        st.info(f"Nếu chuyển nhân viên này sang 'không làm thêm', xác suất nghỉ việc dự kiến giảm khoảng "
                f"**{ate*100:.1f} điểm %** (theo ATE đã hiệu chỉnh).")

# ===================================================================
# PAGE 5 : SO SANH
# ===================================================================
elif page.startswith("5"):
    st.title("⚖️ So sánh mô hình")
    dfm = pd.DataFrame([
        {"Mô hình": "Random Forest", **{k: round(metrics["RandomForest"][k], 3)
         for k in ["roc_auc", "pr_auc", "f1", "precision", "recall"]}},
        {"Mô hình": "XGBoost", **{k: round(metrics["XGBoost"][k], 3)
         for k in ["roc_auc", "pr_auc", "f1", "precision", "recall"]}},
    ]).set_index("Mô hình")
    st.dataframe(dfm, use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.image(str(REP / "roc_curves.png"), use_container_width=True)
    with c2:
        st.image(str(REP / "metrics_bar.png"), use_container_width=True)
    st.info("**Lựa chọn:** Random Forest làm mô hình chính (AUC & Recall cao, bắt được nhiều rủi ro). "
            "XGBoost dùng khi ưu tiên Precision (ít báo động giả).")
