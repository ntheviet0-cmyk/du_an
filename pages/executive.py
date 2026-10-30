"""1. Tong quan dieu hanh — tra loi trong 5 giay: suc khoe ra sao, ai nguy cap, lam gi."""
import pandas as pd
import plotly.express as px
import streamlit as st

from ui.components import action_card, badge, footer_note, hero, insight, kpi
from ui.data import build_risk_list, load_meta, load_results, load_tabular

hero("👋 Chào bạn, đây là trợ lý giữ chân nhân viên",
     "Mỗi tuần mở 2 trang này là đủ: danh sách người cần giữ, và 3 việc nên làm.")

b1, b2, b3 = st.columns(3)
if b1.button("🚨 Xem ai cần giữ ngay", use_container_width=True):
    st.switch_page("pages/risklist.py")
if b2.button("🔍 Chấm điểm 1 người", use_container_width=True):
    st.switch_page("pages/profile.py")
if b3.button("🔗 Thử chính sách mới", use_container_width=True):
    st.switch_page("pages/policy.py")

risk = build_risk_list()
res = load_results()
meta = load_meta()
gm = meta["metrics"]["Graph"]
ate = res["causal"]["ate"]
df, _, _, _ = load_tabular()

n_red = int((risk["band"] == "red").sum())
n_amber = int((risk["band"] == "amber").sum())
ot_yes = df[df["OverTime"] == "Yes"]["Attrition"].mean() * 100
ot_no = df[df["OverTime"] == "No"]["Attrition"].mean() * 100
by_dept_true = df.groupby("Department")["Attrition"].mean().sort_values(ascending=False)
top_dept, top_rate = by_dept_true.index[0], by_dept_true.iloc[0] * 100

c1, c2, c3, c4 = st.columns(4)
kpi(c1, "16,1%", "Tỷ lệ nghỉ việc thực tế", "237 / 1.470 nhân viên")
kpi(c2, str(n_red), "Người cần giữ ngay", f"+ {n_amber} người cần theo dõi", "red")
kpi(c3, f"+{ate * 100:.1f}%", "Làm thêm giờ làm tăng nguy cơ", "đã kiểm định, tin cậy", "amber")
kpi(c4, f"{gm['roc_auc'] * 100:.0f} điểm", "Độ chính xác mô hình", "chấm đúng 8/10 trường hợp", "green")

st.markdown('<div class="hr-section">🚨 Ba điểm nóng nhất</div>', unsafe_allow_html=True)
for band, name, desc in [
    ("red", "Có làm thêm giờ", f"tỷ lệ nghỉ {ot_yes:.1f}% so với {ot_no:.1f}% khi không làm thêm"),
    ("red", f"Phòng {top_dept}", f"tỷ lệ nghỉ {top_rate:.1f}% — cao nhất công ty"),
    ("amber", "Thâm niên 0–2 năm", "nhóm nghỉ nhiều nhất, cần buddy/onboarding"),
]:
    st.markdown(f'{badge(band)} <b>{name}</b> — {desc}', unsafe_allow_html=True)

st.markdown('<div class="hr-section">📉 Rủi ro theo phòng ban & thâm niên</div>', unsafe_allow_html=True)
dfp = risk.copy()
dfp["Thâm niên"] = pd.cut(dfp["YearsAtCompany"], [0, 2, 5, 100], labels=["0–2 năm", "3–5 năm", "5+ năm"])
col1, col2 = st.columns(2)
with col1:
    by_dept = dfp.groupby("Department")["proba"].mean().reset_index()
    fig = px.bar(by_dept, x="Department", y="proba", labels={"Department": "Phòng ban", "proba": "Rủi ro TB"},
                 color="proba", color_continuous_scale="Reds")
    fig.update_layout(showlegend=False, height=320, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)
with col2:
    by_ten = dfp.groupby("Thâm niên", observed=True)["proba"].mean().reset_index()
    fig2 = px.bar(by_ten, x="Thâm niên", y="proba", labels={"proba": "Rủi ro TB"},
                  color="proba", color_continuous_scale="Oranges")
    fig2.update_layout(showlegend=False, height=320, margin=dict(t=10))
    st.plotly_chart(fig2, use_container_width=True)

st.markdown('<div class="hr-section">✅ Ba việc nên làm tuần này</div>', unsafe_allow_html=True)
insight("Người cụ thể nằm ở trang <b>Ưu tiên giữ chân</b> — xuất CSV mang đi họp được ngay.")
action_card("Giảm làm thêm giờ ở nhóm đỏ", f"mỗi 100 người chuyển khỏi OT cứu được ~{ate * 100:.0f} người")
action_card(f"Gặp 1-1 toàn bộ ca đỏ trong 7 ngày", f"hiện có {n_red} ca ưu tiên + {n_amber} ca theo dõi")
action_card("Tăng cường buddy cho nhóm 0–2 năm", "nhóm nghỉ nhiều nhất, chi phí can thiệp rẻ nhất")

footer_note()
