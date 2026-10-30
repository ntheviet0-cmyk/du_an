"""HR Giữ chân — People Analytics theo workflow HR (model: GraphSAGE best-tune)."""
import streamlit as st

from ui.components import apply_theme

st.set_page_config(page_title="HR Giữ chân", page_icon="🧑‍💼", layout="wide")
apply_theme()

pg = st.navigation([
    st.Page("pages/executive.py", title="Tổng quan điều hành", icon="🏠"),
    st.Page("pages/risklist.py", title="Ưu tiên giữ chân", icon="🚨"),
    st.Page("pages/profile.py", title="Hồ sơ nhân viên", icon="🔍"),
    st.Page("pages/discover.py", title="Khám phá nguyên nhân", icon="📊"),
    st.Page("pages/policy.py", title="Mô phỏng chính sách", icon="🔗"),
    st.Page("pages/modeltrust.py", title="Mô hình & niềm tin", icon="⚖️"),
    st.Page("pages/llm_test.py", title="Thử LLM (nội bộ)"),
])
with st.sidebar:
    st.title("🧑‍💼 HR Giữ chân")
    st.caption("Trợ lý AI giúp phát hiện sớm\nnguy cơ nghỉ việc và gợi ý giữ người.")
pg.run()
