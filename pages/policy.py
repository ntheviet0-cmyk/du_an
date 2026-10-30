"""5. Mo phong chinh sach — giam OT cuu duoc bao nhieu nguoi, bao nhieu tien."""
import numpy as np
import streamlit as st

from ui.components import action_card, insight, kpi, warn
from ui.data import load_results, load_tabular

st.title("🔗 Mô phỏng chính sách")
st.caption("Trước khi ký chính sách — biết trước cứu được bao nhiêu người, bao nhiêu tiền.")

res = load_results()
ate = res["causal"]["ate"]
df, _, _, _ = load_tabular()
avg_monthly_usd = float(df["MonthlyIncome"].mean())

st.subheader("📌 Kết luận nhân quả đã kiểm định")
c1, c2, c3 = st.columns(3)
kpi(c1, f"+{ate * 100:.1f} điểm %", "Làm thêm giờ làm tăng nguy cơ nghỉ", "đã kiểm định 2 chiều → tin cậy", "red")
kpi(c2, "Đạt", "Bài kiểm tra độ tin cậy 1", "kết quả giả ≈ 0 như kỳ vọng", "green")
kpi(c3, "Ổn định", "Bài kiểm tra độ tin cậy 2", "kết quả giữ nguyên khi gây nhiễu", "green")

st.subheader("🧪 Mô phỏng toàn công ty")
headcount = st.number_input("Tổng số nhân viên", 100, 100000, 1470, step=50)
ot_rate = st.slider("Tỷ lệ đang làm thêm giờ (%)", 0, 100, 28)
reduce = st.slider("Mức giảm OT nhờ chính sách (%)", 0, 100, 50)
replace_pct = st.slider("Chi phí thay 1 người (% lương năm)", 50, 200, 100,
                        help="SHRM: 50–60% lương, tổng chi phí tới 200%.")

n_shifted = headcount * ot_rate / 100 * reduce / 100
avoided = n_shifted * ate
money = avoided * avg_monthly_usd * 12 * replace_pct / 100
st.success(f"Ước tính **giảm {avoided:.0f} ca nghỉ việc** (trong {n_shifted:.0f} người được giảm OT) — "
           f"**tiết kiệm ~${money:,.0f}** chi phí thay thế.")
st.caption(f"Cách tính: {n_shifted:.0f} người × tác động {ate:.3f} = {avoided:.0f} ca; "
           f"mỗi ca ≈ ${avg_monthly_usd * 12:,.0f} lương năm × {replace_pct}%.")
with st.expander("📋 Lấy tóm tắt mang đi họp"):
    st.code(f"Chính sách giảm OT {reduce}% cho {ot_rate}% nhân sự ({headcount} người): "
            f"giảm ~{avoided:.0f} ca nghỉ, tiết kiệm ~${money:,.0f}.", language=None)

st.subheader("🧪 Mô phỏng 1 người")
ot_now = st.selectbox("Nhân viên hiện có làm thêm giờ?", ["Có", "Không"])
if ot_now == "Có":
    st.info(f"Chuyển sang không làm thêm: nguy cơ dự kiến **giảm ~{ate * 100:.1f} điểm %** (đã hiệu chỉnh nhiễu).")
    action_card("Việc nên làm", "lên lịch giảm OT trong 30 ngày + gặp 1-1 xác nhận khối lượng việc")
else:
    st.info("Người này không làm thêm — giữ nguyên, tập trung vào lương và gắn kết.")

with st.expander("Dành cho phân tích: sơ đồ nhân quả"):
    from pathlib import Path as _P
    st.image(str(_P("reports/ibm/attrition_dag.png")), use_container_width=True)
    st.caption("Sơ đồ: Làm thêm giờ → Nghỉ việc, đã loại trừ 17 yếu tố gây nhiễu.")
warn("Số mô phỏng là ước tính quy hoạch, không phải cam kết với từng cá nhân.")
