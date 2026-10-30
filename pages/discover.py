"""4. Kham pha nguyen nhan — nhom nao nghi nhieu nhat + SHAP tieng Viet."""
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

from ui.components import footer_note, insight, page_header
from ui.data import REP, build_risk_list, load_tabular
from ui.vocab import hr_dict

page_header("📊 Khám phá nguyên nhân", "Nhóm nào nghỉ nhiều — và 3 câu trả lời có sẵn.")

df, _, _, _ = load_tabular()
risk = build_risk_list()

t1, t2, t3 = st.tabs(["Nhóm rủi ro", "SHAP toàn cục", "Ví dụ thực tế"])
with t1:
    c1, c2 = st.columns(2)
    with c1:
        by = df.groupby("Department")["Attrition"].mean().reset_index()
        fig = px.bar(by, x="Department", y="Attrition", color="Attrition", color_continuous_scale="Reds",
                     labels={"Attrition": "Tỷ lệ nghỉ", "Department": "Phòng ban"})
        fig.update_layout(showlegend=False, height=300, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        df2 = df.copy()
        df2["Thâm niên"] = pd.cut(df2["YearsAtCompany"], [0, 2, 5, 100], labels=["0–2", "3–5", "5+"])
        by2 = df2.groupby(["Thâm niên", "OverTime"], observed=True)["Attrition"].mean().reset_index()
        fig2 = px.bar(by2, x="Thâm niên", y="Attrition", color="OverTime", barmode="group",
                      labels={"Attrition": "Tỷ lệ nghỉ", "OverTime": "Làm thêm"},
                      color_discrete_map={"Yes": "#dc2626", "No": "#2563eb"})
        fig2.update_layout(height=300, margin=dict(t=10))
        st.plotly_chart(fig2, use_container_width=True)
    insight("Đọc nhanh: <b>Sales + làm thêm giờ + thâm niên 0–2 năm</b> là tổ hợp rủi ro cao nhất — trùng với kết luận SHAP. "
            "Câu 1: nhóm nào nghỉ nhiều nhất? → xem 2 biểu đồ. "
            "Câu 2: yếu tố nào quan trọng nhất? → xem tab SHAP. "
            "Câu 3: 1 người cụ thể vì sao bị gắn cờ? → xem tab Ví dụ.")
    try:
        st.dataframe(pd.read_csv(REP / "graph_shap_cohort.csv"), use_container_width=True)
    except Exception:
        pass
with t2:
    st.image(str(REP / "shap_graph_global.png"), use_container_width=True)
    st.image(str(REP / "shap_graph_beeswarm.png"), use_container_width=True)
    try:
        g = pd.read_csv(REP / "graph_shap_grouped.csv").head(12)
        hr = hr_dict()
        g["Tên tiếng Việt"] = g["feature_group"].map(lambda x: hr.get(f"num__{x}", hr.get(f"cat__{x}_Yes", x)))
        st.dataframe(g, use_container_width=True)
    except Exception as e:
        st.warning(f"Chưa có file SHAP: {e}")
with t3:
    try:
        cases = pd.read_csv(REP / "graph_shap_cases.csv")
        k = st.selectbox("Chọn ca mẫu", list(range(len(cases))),
                         format_func=lambda k: f"Ca {k}: {cases.loc[k, 'proba'] * 100:.0f}% — "
                                               f"{'nghỉ thật' if cases.loc[k, 'true_label'] == 1 else 'ở lại thật'}")
        st.image(str(REP / f"shap_graph_waterfall_{int(k):02d}.png"), use_container_width=True)
        st.caption("Hiển thị toàn bộ nhóm HR gốc có đóng góp khác 0, sắp xếp theo độ lớn |SHAP|. "
                   "Đỏ = tăng, xanh = giảm dự báo nghỉ việc. Base trung bình là 21,8%; "
                   "đây không phải kết luận nhân quả.")
    except Exception as e:
        st.warning(f"Chưa có biểu đồ SHAP local: {e}")

footer_note()
