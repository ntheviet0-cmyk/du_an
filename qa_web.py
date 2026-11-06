"""qa_web.py - Kiem tra web tinh:
- ID su dung trong app.js phai co trong index.html (hoac la ID sinh dong)
- File bat buoc phai du (ke ca du lieu model thuc: weights/xtrain/edges/preproc/parity)
- Moi data-policy trong HTML phai co key trong RATES/DESCS cua app.js
- Moi <script src> phai ton tai
- Font tu host: fonts.css ton tai + moi url(...) trong do ton tai
- Nav khop voi danh sach trang thuc te; data-goto tro dung trang
- Neu co node: chay test_sage_parity.js (JS <-> Python)
Chay: python qa_web.py
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
WEB = REPO / "web"
html = (WEB / "index.html").read_text(encoding="utf-8")
js = "\n".join((WEB / name).read_text(encoding="utf-8") for name in ("app.js", "policy.js"))

ids_html = set(re.findall(r'id="([^"]+)"', html))
duplicate_ids = sorted(i for i in ids_html if len(re.findall(r'id="' + re.escape(i) + r'"', html)) > 1)
print("duplicate-ids:", duplicate_ids if duplicate_ids else "NONE")
used = set(re.findall(r'\$\("([^"]+)"\)', js))
dynamic_prefix = ("pf-", "fg-")
dynamic_ids = {"drawer-track", "drawer-note", "drawer-note-hint"}
missing = sorted(u for u in used if u not in ids_html and u not in dynamic_ids and not u.startswith(dynamic_prefix))
print("used-not-in-html:", missing if missing else "NONE")

need = ["styles.css", "app.js", "policy.js", "sage.js", "test_sage_parity.js", "index.html",
        "data/kpi.js", "data/vocab.js", "data/shap.js", "data/risk.js", "data/meta.js",
        "data/preproc.js", "data/weights.js", "data/xtrain.js", "data/edges.js", "data/parity.js",
        "assets/fonts.css",
        "assets/shap_graph_global.png", "assets/shap_graph_beeswarm.png",
        "assets/confusion_matrices_3way.png", "assets/roc_curves.png",
        "assets/attrition_dag.png"]
bad = [f for f in need if not (WEB / f).exists()]
print("missing-files:", bad if bad else "NONE")

font_css = WEB / "assets" / "fonts.css"
font_urls = re.findall(r'url\(([^)]+)\)', font_css.read_text(encoding="utf-8")) if font_css.exists() else []
bad_fonts = [u for u in font_urls if not (WEB / "assets" / u).exists()]
print("missing-fonts:", bad_fonts if bad_fonts else "NONE (%d font)" % len(font_urls))

srcs = re.findall(r'<script src="([^"]+)"', html)
bad_srcs = [s for s in srcs if not (WEB / s).exists()]
print("missing-script-src:", bad_srcs if bad_srcs else "NONE")

nav_html = set(re.findall(r'data-page="([^"]+)"', html))
pages_expected = {"tong-quan", "uu-tien", "nhan-vien", "nguyen-nhan", "chinh-sach", "gioi-thieu"}
print("nav-mismatch:", (nav_html ^ pages_expected) if (nav_html ^ pages_expected) else "NONE")
goto_html = set(re.findall(r'data-goto="([^"]+)"', html))
bad_goto = sorted(goto_html - pages_expected)
print("goto-mismatch:", bad_goto if bad_goto else "NONE")
print("sections:", sorted(re.findall(r'<section class="page[^"]*" id="([^"]+)"', html)))

html_policies = set(re.findall(r'data-policy="([^"]+)"', html))
m = re.search(r'var RATES = \{([^}]+)\}', js)
rates_keys = set(re.findall(r'(\w+):', m.group(1))) if m else set()
bad_policy = sorted(html_policies - rates_keys)
print("html-policy-not-in-RATES:", bad_policy if bad_policy else "NONE")

wfs = sorted((WEB / "assets").glob("shap_graph_waterfall_*.png"))
print("waterfalls:", len(wfs))

node_ok = None
if shutil.which("node"):
    r = subprocess.run(["node", "test_sage_parity.js"], cwd=str(WEB), capture_output=True, text=True)
    print("node parity:\n" + r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr.strip())
    node_ok = r.returncode == 0
else:
    print("node parity: SKIP (khong co node)")

fail = (duplicate_ids or missing or bad or bad_fonts or bad_srcs or bad_goto or (nav_html ^ pages_expected)
        or bad_policy or len(wfs) != 14 or node_ok is False)
if fail:
    print("QA_STATIC_FAIL")
    sys.exit(1)
print("QA_STATIC_OK")
