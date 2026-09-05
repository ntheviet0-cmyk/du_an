"""build_attrition_report.py - Sinh bao cao DOCX tieng Viet tu REPORT.md + JSON + figures.

Nguon: reports/ibm/REPORT.md, ibm_results.json, analyze_stats.json, *.png
Dich:  reports/docx/HR_Attrition_Report.docx
Du lieu IBM HR Analytics Employee Attrition (nhan nghi viec that).
"""
import os
import re
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IBM_DIR = os.path.join(BASE, "reports", "ibm")
SRC = os.path.join(IBM_DIR, "REPORT.md")
OUT_DIR = os.path.join(BASE, "reports", "docx")
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, "HR_Attrition_Report.docx")

TITLE = "BÁO CÁO NGHIÊN CỨU KHOA HỌC"
SUBTITLE = "Xây dựng mô hình dự đoán khả năng nghỉ việc hỗ trợ quyết định tuyển dụng"
SUBTITLE_EN = ("Building an employee attrition and job-change prediction model "
               "to support recruitment decisions")

FIGURES = [
    ("roc_curves.png", "Hình 1. Đường cong ROC so sánh Random Forest và XGBoost trên tập kiểm thử."),
    ("metrics_bar.png", "Hình 2. Biểu đồ cột so sánh các chỉ số hiệu năng (ROC-AUC, PR-AUC, F1, Precision, Recall)."),
    ("graph_similarity.png", "Hình 3. Đồ thị tương đồng nhân viên (kNN, mẫu 150) minh họa mô hình Graph."),
    ("attrition_dag.png", "Hình 4. Đồ thị nhân quả (Causal DAG): OverTime → Attrition với 17 biến nhiễu."),
    ("shap_rf.png", "Hình 5. Biểu đồ SHAP summary của mô hình Random Forest."),
    ("shap_xgboost.png", "Hình 6. Biểu đồ SHAP summary của mô hình XGBoost."),
]


def strip_bold(s):
    return s.replace("**", "")


def add_rich(paragraph, text, size=11):
    parts = text.split("**")
    for i, part in enumerate(parts):
        if part == "":
            continue
        run = paragraph.add_run(part)
        run.font.size = Pt(size)
        run.font.name = "Times New Roman"
        if i % 2 == 1:
            run.bold = True


def add_toc(paragraph):
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = 'TOC \\o "1-3" \\h \\z \\u'
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run2 = paragraph.add_run()
    run2._r.append(instr)
    run3 = paragraph.add_run()
    run3._r.append(fld_sep)
    run4 = paragraph.add_run()
    run4._r.append(fld_end)


def style_document(doc):
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)
    for i in (1, 2, 3):
        hs = doc.styles[f"Heading {i}"]
        hs.font.name = "Times New Roman"
        hs.font.color.rgb = RGBColor(0x1F, 0x3B, 0x63)


def add_table(doc, data):
    ncol = len(data[0])
    t = doc.add_table(rows=1, cols=ncol)
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(data[0]):
        cell = t.rows[0].cells[j]
        cell.text = ""
        p = cell.paragraphs[0]
        run = p.add_run(strip_bold(h))
        run.bold = True
        run.font.size = Pt(10)
        run.font.name = "Times New Roman"
    for r in data[1:]:
        cells = t.add_row().cells
        for j in range(min(len(r), ncol)):
            cells[j].text = ""
            p = cells[j].paragraphs[0]
            run = p.add_run(strip_bold(r[j]))
            run.font.size = Pt(10)
            run.font.name = "Times New Roman"
    doc.add_paragraph()


def is_sep(row):
    return all(re.fullmatch(r"[-: ]+", c) for c in row)


def main():
    doc = Document()
    style_document(doc)

    # ---- Bia ----
    for _ in range(4):
        doc.add_paragraph()
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(TITLE); r.bold = True; r.font.size = Pt(18)
    r.font.name = "Times New Roman"
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(SUBTITLE); r.bold = True; r.font.size = Pt(14)
    r.font.name = "Times New Roman"
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(SUBTITLE_EN); r.italic = True; r.font.size = Pt(11)
    r.font.name = "Times New Roman"
    doc.add_page_break()

    # ---- Muc luc ----
    doc.add_heading("Mục lục", level=1)
    add_toc(doc.add_paragraph())
    doc.add_paragraph("Nhấn chuột phải vào mục lục và chọn Update Field để cập nhật số trang.")
    doc.add_page_break()

    lines = open(SRC, encoding="utf-8").read().split("\n")
    i = 0
    in_code = False
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            if in_code:
                code_buf = []
            else:
                p = doc.add_paragraph()
                run = p.add_run("\n".join(code_buf))
                run.font.name = "Consolas"
                run.font.size = Pt(9)
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue
        if stripped == "---":
            i += 1
            continue
        if stripped.startswith("|") and stripped.count("|") >= 2:
            tbl = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl.append(lines[i])
                i += 1
            rows = [[c.strip() for c in tl.strip().strip("|").split("|")] for tl in tbl]
            data = [r for r in rows if not is_sep(r)]
            if data:
                add_table(doc, data)
            continue
        if stripped.startswith("#### "):
            doc.add_heading(stripped[5:], level=3)
        elif stripped.startswith("### "):
            doc.add_heading(stripped[4:], level=2)
        elif stripped.startswith("## "):
            doc.add_heading(stripped[3:], level=1)
        elif stripped.startswith("# "):
            h = doc.add_heading(stripped[2:], level=1)
            for run in h.runs:
                run.font.size = Pt(16)
        elif re.match(r"^\d+\.\s", stripped):
            p = doc.add_paragraph(style="List Number")
            add_rich(p, re.sub(r"^\d+\.\s", "", stripped))
        elif stripped.startswith("- "):
            p = doc.add_paragraph(style="List Bullet")
            add_rich(p, stripped[2:])
        elif stripped.startswith("> "):
            p = doc.add_paragraph()
            run = p.add_run(stripped[2:])
            run.italic = True
            run.font.size = Pt(11)
            run.font.name = "Times New Roman"
        elif stripped == "":
            pass
        else:
            txt = stripped
            if txt.startswith("*") and not txt.startswith("**"):
                txt = txt.strip("*")
            p = doc.add_paragraph()
            add_rich(p, txt)
        i += 1

    # ---- Phu luc hinh anh ----
    doc.add_page_break()
    doc.add_heading("Phụ lục A – Hình ảnh minh họa", level=1)
    for idx, (im, caption) in enumerate(FIGURES, start=1):
        path = os.path.join(IBM_DIR, im)
        if os.path.exists(path):
            p = doc.add_paragraph()
            run = p.add_run(caption)
            run.bold = True
            run.font.size = Pt(11)
            run.font.name = "Times New Roman"
            doc.add_picture(path, width=Inches(6.0))
            doc.add_paragraph()

    doc.save(OUT)
    print("saved HR_Attrition_Report.docx", os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()
