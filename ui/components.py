"""ui/components.py - Component dung chung: KPI card, huy hieu muc do, hop insight."""
import streamlit as st

BAND = {
    "red": ("ƯU TIÊN GIỮ", "hr-red"),
    "amber": ("THEO DÕI", "hr-amber"),
    "green": ("ỔN ĐỊNH", "hr-green"),
}


def band_of(proba, thr=0.8):
    if proba >= thr:
        return "red"
    if proba >= thr * 0.6:
        return "amber"
    return "green"


def badge(band):
    label, cls = BAND[band]
    return f'<span class="hr-badge {cls}">{label}</span>'


def kpi(col, value, label, delta="", tone=""):
    col.markdown(
        f'<div class="hr-kpi {tone}"><div class="v">{value}</div>'
        f'<div class="l">{label}</div>'
        + (f'<div class="d">{delta}</div>' if delta else '')
        + '</div>',
        unsafe_allow_html=True,
    )


def insight(text):
    st.markdown(f'<div class="hr-insight">{text}</div>', unsafe_allow_html=True)


def warn(text):
    st.markdown(f'<div class="hr-warn">⚠️ {text}</div>', unsafe_allow_html=True)


def action_card(title, reason):
    st.markdown(f'<div class="hr-action">✅ <b>{title}</b><br><span style="color:#64748b;font-size:13px;">Vì sao: {reason}</span></div>',
                unsafe_allow_html=True)


def chips(items):
    st.markdown("".join(f'<span class="hr-chip">{x}</span>' for x in items), unsafe_allow_html=True)


def page_header(title, subtitle, cta=None):
    """Tieu de trang thong nhat: ten + 1 cau 'trang nay de lam gi' + nut buoc tiep theo."""
    st.title(title)
    st.caption(subtitle)
    if cta:
        label, page = cta
        if st.button(label, key=f"cta_{page}"):
            st.switch_page(page)


def hero(title, subtitle):
    st.markdown(f'<div class="hr-hero"><h2>{title}</h2><p>{subtitle}</p></div>', unsafe_allow_html=True)


def summary_bar(items):
    """items: list ('nhan', gia tri). Hien thanh tóm tắt de luot nhanh."""
    st.markdown('<div class="hr-summary">' + "".join(
        f'<span class="hr-pill">{label}: <b>{val}</b></span>' for label, val in items
    ) + '</div>', unsafe_allow_html=True)


def empty_state(msg):
    st.info(f"📭 {msg}")


def footer_note():
    st.markdown('<div class="hr-footer">Nguồn: dữ liệu nội bộ 1.470 nhân viên · '
                'AI chỉ hỗ trợ quyết định, không thay thế đánh giá của HR.</div>', unsafe_allow_html=True)


def fmt_pct(x):
    return f"{x * 100:.0f}%"


def apply_theme():
    css = (__import__("pathlib").Path(__file__).resolve().parent / "theme.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
