"""6. Mo hinh & niem tin — danh cho analyst: metric, duong cong, cong bang."""
import pandas as pd
import streamlit as st
from pathlib import Path

from ui.components import insight, warn
from ui.data import REP, load_meta, load_results

st.title("⚖️ Mô hình & niềm tin")
st.caption("Mọi con số kỹ thuật nằm gọn ở đây — các trang khác chỉ hiện kết luận.")

meta = load_meta()
res = load_results()
rows = []
for key, label in [("RandomForest", "Random Forest"), ("XGBoost", "XGBoost"), ("Graph", "GraphSAGE best-tune")]:
    if key in meta["metrics"]:
        r = {"Mô hình": label}
        r.update({k: round(meta["metrics"][key][k], 3) for k in ["roc_auc", "pr_auc", "f1", "precision", "recall"]})
        rows.append(r)
st.dataframe(pd.DataFrame(rows).set_index("Mô hình"), use_container_width=True)
insight("Đọc nhanh: GraphSAGE best-tune dẫn AUC (0.801); Random Forest bắt sót ít nhất (Recall 0.68); "
        "XGBoost cảnh báo chắc nhất (Precision 0.67). Không có model thắng mọi mặt.")

c1, c2 = st.columns(2)
with c1:
    st.image(str(REP / "roc_curves.png"), use_container_width=True)
with c2:
    st.image(str(REP / "confusion_matrices_3way.png"), use_container_width=True)

with st.expander("Vì sao GraphSAGE là mô hình chính?"):
    st.markdown("""
    - Mỗi nhân viên là 1 node trên đồ thị tương đồng (kNN-cosine k=15); model học từ 15 đồng nghiệp giống nhất.
    - Bản best-tune (`h64`, seed 42009): AUC 0.801 / F1 0.543 — ngang Random Forest.
    - SHAP giải thích trực tiếp GNN frozen (xem trang Khám phá) — không model thứ hai, không xây lại đồ thị.
    """)
with st.expander("Công bằng & giới hạn"):
    warn("Yếu tố nhân khẩu (tuổi, giới tính, hôn nhân) đôi khi lọt top SHAP — chỉ dùng để thiết kế chính sách hỗ trợ, "
         "tuyệt đối không dùng để phân biệt đối xử khi đánh giá cá nhân.")
    st.markdown("- Dữ liệu IBM 1.470 người (mẫu hư cấu chuẩn ngành), tỷ lệ nghỉ 16,1%.\n"
                "- Số mô phỏng chính sách là ước tính quy hoạch.\n"
                "- GNN có nhiễu ngẫu nhiên ±0.01 AUC giữa các lần train (đã đo stability 3 seeds).")
with st.expander("Từ điển thuật ngữ"):
    st.markdown("**AUC**: khả năng xếp đúng người (càng gần 1 càng tốt). **Precision**: gắn cờ 10 người thì mấy người nghỉ thật. "
                "**Recall**: 10 người nghỉ thật thì bắt được mấy người. **F1**: trung bình 2 chỉ số trên. "
                "**SHAP**: phần đóng góp của từng yếu tố vào điểm rủi ro. **ATE**: tác động nhân quả trung bình của chính sách.")
