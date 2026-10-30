"""ui/llm_probe.py - Helper cho trang thu LLM (noi bo).

Dung prompt 4 khoi (vai tro + context ca + noi quy cong ty + khuon dau ra),
goi API tuong thich OpenAI (mac dinh: Google AI Studio / Gemini) bang thu vien
chuan urllib (khong can cai them goi ngoai), va kiem tra cau tra loi co bam
sach top yeu to SHAP khong (chong ao giac).

Khong import streamlit: test duoc bang python thuan.
"""
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REP = REPO / "reports" / "ibm"
ART = REPO / "models" / "ibm"

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
# Khong gioi han thoi gian cho: doi den khi API tra loi xong.

CAT = ["BusinessTravel", "Department", "EducationField", "Gender",
       "JobRole", "MaritalStatus", "OverTime"]

LEVEL_VN = {"red": "Cao", "amber": "Trung binh", "green": "Thap"}

SENSITIVE_HINTS = ["tuoi", "gioi tinh", "nam", "nu", "doc than",
                   "ket hon", "ly hon", "hon nhan"]

DEFAULT_RULES = (
    "1. Hạn chế làm thêm giờ: tối đa 8 giờ làm thêm mỗi tuần, ưu tiên nghỉ bù.\n"
    "2. Khung xem lại thu nhập: 6 tháng một lần, dựa theo mặt bằng phòng ban.\n"
    "3. Nhân viên mới (dưới 2 năm) được bố trí người hướng dẫn (buddy) trong 3 tháng đầu.\n"
    "4. Linh hoạt giờ làm 1 đến 2 ngày mỗi tuần khi công việc cho phép.\n"
    "5. Mọi đề xuất thưởng giữ chân cần trưởng phòng phê duyệt."
)

SYSTEM_PROMPT = """Bạn là trợ lý nhân sự của công ty, viết tiếng Việt tự nhiên, giọng của một người làm nhân sự lâu năm: điềm tĩnh, tôn trọng, nói thẳng vào việc. Người đọc là chuyên viên hoặc trưởng phòng nhân sự, không có nền tảng kỹ thuật.

NHIỆM VỤ
Dựa vào thông tin của một nhân viên, viết ngắn gọn hai phần: vì sao người này có thể nghỉ việc, và gợi ý cách giữ chân phù hợp với nội quy công ty.

NGUYÊN TẮC BẮT BUỘC
1. Chỉ dùng đúng các yếu tố được cung cấp. Tuyệt đối không bịa thêm lý do, hoàn cảnh cá nhân, cảm xúc, câu nói hay sự kiện không có trong dữ liệu.
2. Không dùng từ kỹ thuật. Không nhắc tên mô hình, thuật toán, xác suất, điểm số hay bất kỳ con số nội bộ nào. Mức nguy cơ chỉ nói bằng chữ (Cao, Trung bình, Thấp).
3. Gợi ý giữ chân chỉ được lấy từ nội quy công ty. Không đề xuất biện pháp, mức lương hay phúc lợi nằm ngoài nội quy. Nếu không có điều khoản nào phù hợp với một yếu tố, bỏ qua yếu tố đó, không gượng ép.
4. Mỗi gợi ý phải nêu rõ áp dụng điều khoản nào của nội quy và nối với lý do nào ở phần 1.
5. Yếu tố "Tuổi" chỉ được nhắc ở mức mô tả trung tính. Không suy diễn về thế hệ, giai đoạn cuộc sống hay tính cách. Tuyệt đối không đưa ra gợi ý hành động dựa trên tuổi.
6. Yếu tố đang giữ người ở lại chỉ dùng để nói đâu là điểm tựa nên bảo vệ. Không biến chúng thành lý do nghỉ.
7. Nếu một đề xuất có liên quan đến thưởng giữ chân, phải ghi rõ cần trưởng phòng phê duyệt.
8. Không hứa hẹn kết quả. Dùng cách nói "có thể", "nên cân nhắc". Đây là gợi ý hỗ trợ, quyết định cuối cùng thuộc về người làm nhân sự.
9. Không dùng emoji, không dùng ký hiệu lạ, không viết hoa toàn bộ."""

OUTPUT_FORMAT = """ĐỊNH DẠNG ĐẦU RA
Trả lời đầy đủ cả hai phần, tổng khoảng 200 chữ, văn xuôi ngắn hoặc gạch đầu dòng đơn giản:

Phần 1. Vì sao người này có thể nghỉ
Một câu nêu mức cần giữ chân bằng chữ. Sau đó giải thích từng điểm làm tăng khả năng nghỉ bằng ngôn ngữ đời thường, mỗi điểm một ý ngắn. Cuối phần có một câu ghi nhận điểm đang giữ người ở lại.

Phần 2. Gợi ý giữ chân
Tối đa 3 gợi ý, sắp theo mức ưu tiên. Mỗi gợi ý theo dạng: việc nên làm, áp dụng điều khoản nào của nội quy, giải quyết lý do nào ở phần 1."""

SELF_CHECK = """TỰ KIỂM TRA TRƯỚC KHI TRẢ LỜI
- Có lý do hay chi tiết nào ngoài dữ liệu được cung cấp không? Nếu có, xóa.
- Có từ kỹ thuật hoặc con số nội bộ nào không? Nếu có, viết lại bằng lời thường.
- Mọi gợi ý đã dẫn đúng điều khoản trong nội quy chưa?
- Có gợi ý nào dựa trên tuổi không? Nếu có, xóa.
- Đã đủ cả hai phần và khoảng 200 chữ chưa?"""

_hr = None
_pre = _rf = None


def hr_dict():
    global _hr
    if _hr is None:
        try:
            _hr = json.loads((REP / "hr_dictionary.json").read_text(encoding="utf-8"))
        except Exception:
            _hr = {}
    return _hr


def vn(feat):
    """cat__OverTime_Yes -> 'Lam them gio'; giu nguyen neu khong co trong tu dien."""
    return hr_dict().get(feat, feat)


def _models():
    global _pre, _rf
    if _pre is None:
        import joblib
        _pre = joblib.load(ART / "preprocessor.joblib")
        _rf = joblib.load(ART / "random_forest.joblib")
    return _pre, _rf


def tree_shap_for_row(row_df):
    """SHAP cay (TreeExplainer, duoi 1 giay) cho 1 nhan vien.

    row_df: DataFrame 1 dong, dung cac cot CAT + so (giong app_ibm.py).
    Tra ve dict {proba, up: [(nhan, gia tri)] x3, down: [...] x2, base}.
    """
    import numpy as np
    import shap
    pre, rf = _models()
    Xt = pre.transform(row_df)
    Xt = Xt.toarray().astype("float32") if hasattr(Xt, "toarray") else np.asarray(Xt, "float32")
    clf = rf.named_steps["clf"]
    proba = float(clf.predict_proba(Xt)[:, 1][0])
    ex = shap.TreeExplainer(clf)
    sv = ex.shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[1]
    elif getattr(sv, "ndim", 2) == 3:
        sv = sv[:, :, 1]
    sv = np.asarray(sv)[0]
    ev = ex.expected_value
    ev = float(np.atleast_1d(ev)[1]) if len(np.atleast_1d(ev)) > 1 else float(np.atleast_1d(ev)[0])
    feats = list(pre.get_feature_names_out())
    order = np.argsort(sv)
    up = [(vn(feats[j]), float(sv[j])) for j in order[::-1][:3]]
    down = [(vn(feats[j]), float(sv[j])) for j in order[:2]]
    return {"proba": proba, "up": up, "down": down, "base": ev}


def build_prompt(ctx, rules):
    """Dung prompt 4 khoi tu context ca + noi quy. Tra ve (system, user).

    Khong dua con so SHAP vao prompt: chi gui ten yeu to dang chu de so
    noi bo khong lot ra cau tra loi.
    """
    up_lines = "\n".join(f"- {label}" for label, _ in ctx["up"])
    down_lines = "\n".join(f"- {label}" for label, _ in ctx["down"])
    meaning = (f"Khoảng {ctx['pct']} trên 100 người có hoàn cảnh "
               f"giống vậy muốn nghỉ.")
    user = (
        "NỘI QUY CÔNG TY (nguồn duy nhất cho phần gợi ý)\n" + rules.strip() + "\n\n"
        "THÔNG TIN NHÂN VIÊN\n"
        f"Mã nhân viên: {ctx['code']}\n"
        f"Mức cần giữ chân: {ctx['level']}\n"
        f"Ý nghĩa: {meaning}\n"
        f"Các điểm làm tăng khả năng nghỉ:\n{up_lines}\n"
        f"Các điểm đang giữ người ở lại:\n{down_lines}\n\n"
        + OUTPUT_FORMAT + "\n\n" + SELF_CHECK
    )
    return SYSTEM_PROMPT, user


STOP_VN = {"stop": "hoan tat", "length": "het gioi han chu giua chung",
            "content_filter": "bi chan noi dung", "tool_calls": "hoan tat"}


def call_llm(api_key, model, system, user, temperature=0.3,
             max_tokens=None, timeout=None, url=GEMINI_URL):
    """Goi API. Tra ve dict {ok, text|error, elapsed, stop, retried, truncated}.

    Neu API bao het gioi han chu giua chung (finish_reason=length) thi tu goi
    lai 1 lan voi tran gap doi de cau tra loi day du.
    """
    t0 = time.time()
    base = {"ok": False, "elapsed": 0.0, "stop": "loi",
            "retried": False, "truncated": False}
    if not (api_key or "").strip():
        return dict(base, error="Chua dan API key. Dan key vao o phia tren roi bam lai.")

    def _check_budget(budget):
        """None = khong gioi han (se bo truong max_tokens). So thi phai nguyen duong."""
        if budget is None:
            return None
        try:
            n = int(budget)
        except (TypeError, ValueError):
            raise ValueError("Tran chu phai la so nguyen duong, hoac de trong de khong gioi han.")
        if n <= 0:
            raise ValueError("Tran chu phai la so nguyen duong, hoac de trong de khong gioi han.")
        return n

    def _once(budget):
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": float(temperature),
        }
        if budget is not None:
            payload["max_tokens"] = budget
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=body,
            headers={"Authorization": "Bearer " + api_key.strip(),
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=None) as r:  # cho den khi xong, khong gioi han
            data = json.loads(r.read().decode("utf-8"))
        choice = data["choices"][0]
        return (choice["message"].get("content", "") or "").strip(), \
            choice.get("finish_reason", "stop")

    try:
        budget = _check_budget(max_tokens)
    except ValueError as e:
        return dict(base, error=str(e), elapsed=time.time() - t0)

    try:
        text, stop = _once(budget)
        retried = False
        if stop == "length" and budget is not None:
            retried = True
            text, stop = _once(budget * 2)
        if stop == "length":
            return dict(base, ok=True, text=text, elapsed=time.time() - t0,
                        stop="van het gioi han chu sau khi goi lai", retried=True,
                        truncated=True)
        return dict(base, ok=True, text=text, elapsed=time.time() - t0,
                    stop=STOP_VN.get(stop, stop), retried=retried)
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8")[:400]
        except Exception:
            detail = ""
        if e.code in (401, 403):
            msg = "Key chua dung hoac chua co quyen. Kiem tra lai key da dan."
        elif e.code == 429:
            msg = "Vuot han muc mien phi luc nay. Cho vai phut roi bam lai."
        elif e.code == 400:
            msg = f"Yeu cau chua hop le (co the sai ten model). Chi tiet: {detail}"
        else:
            msg = f"Loi tu phia dich vu (ma {e.code}). Chi tiet: {detail}"
        return dict(base, error=msg, elapsed=time.time() - t0)
    except (urllib.error.URLError, TimeoutError):
        return dict(base,
                    error="Khong goi duoc (loi mang). Kiem tra mang roi bam thu lai.",
                    elapsed=time.time() - t0)
    except Exception as e:
        return dict(base,
                    error=f"Loi ngoai du kien ({type(e).__name__}). Chup man hinh bao lai de kiem tra.",
                    elapsed=time.time() - t0)


def check_grounding(response, factor_labels):
    """Cau tra loi co nhac dung top yeu to khong. Tra ve (duoc_nhac, vang_mat)."""
    low = (response or "").lower()
    hit, miss = [], []
    for lb in factor_labels:
        (hit if lb.lower() in low else miss).append(lb)
    return hit, miss


def sensitive_in_factors(factor_labels):
    """Top yeu to co dinh yeu to nhan khau nhay cam khong."""
    out = []
    for lb in factor_labels:
        l = lb.lower()
        if any(k in l for k in SENSITIVE_HINTS):
            out.append(lb)
    return out


TECH_TERMS = ["xác suất", "mô hình", "thuật toán", "shap", "điểm số",
              "auc", "roc", "f1", "precision", "recall", "feature",
              "tham số", "api", "json", "kernel", "regression"]
INVENTION_HINTS = ["mâu thuẫn", "xung đột", "sếp", "quản lý trực tiếp",
                   "bị chèn ép", "đồng nghiệp cô lập", "sức khỏe",
                   "bệnh", "thai sản", "con nhỏ", "gia đình có việc"]
ACTION_VERBS = ["nên", "cần", "hãy", "đề xuất", "bố trí", "tăng",
                "giảm", "xem xét", "hẹn", "trao đổi", "điều chỉnh",
                "áp dụng", "triển khai"]


def _split_parts(response):
    low = (response or "").lower()
    idx = low.find("phần 2")
    if idx < 0:
        return response, ""
    return response[:idx], response[idx:]


def audit_response(response, factor_labels):
    """Kiem tra cau tra loi theo checklist cua prompt.

    Tra ve list (ten_muc, dat, chi_tiet). factor_labels: ten yeu to dang chu.
    """
    out = []
    text = response or ""
    part1, part2 = _split_parts(text)

    hit, miss = check_grounding(text, factor_labels)
    out.append(("Chi dung yeu to duoc cung cap",
                len(hit) > 0,
                ("duoc nhac: " + "; ".join(hit)) if hit else "khong thay nhac yeu to nao" +
                (". Vang mat (khong sao): " + "; ".join(miss) if miss else "")))
    invented = [h for h in INVENTION_HINTS if h in text.lower()]
    out.append(("Khong bia them chi tiet",
                len(invented) == 0,
                "nen doc lai bang mat cum tu: " + "; ".join(invented) if invented
                else "khong thay cum tu dang ngo"))

    bad = sorted({t for t in TECH_TERMS if t in text.lower()})
    import re
    nums = sorted(set(re.findall(r"\d+[.,]\d+|\d{2,}", text)))
    bad += [f"con so {n}" for n in nums]
    out.append(("Khong tu ky thuat, khong so noi bo",
                len(bad) == 0,
                "phat hien: " + "; ".join(bad) if bad else "sach"))

    has_p1 = "phần 1" in text.lower()
    has_p2 = "phần 2" in text.lower()
    out.append(("Du ca 2 phan",
                has_p1 and has_p2,
                "thieu " + " va ".join([p for p, h in (("Phan 1", has_p1), ("Phan 2", has_p2)) if not h])
                if not (has_p1 and has_p2) else "du Phan 1 va Phan 2"))

    age_in_p2 = "tuổ" in part2.lower()
    out.append(("Khong goi y dua tren tuoi",
                not age_in_p2,
                "thay nhac tuoi trong Phan 2, can xoa" if age_in_p2
                else "dat (duoc phep mo ta trung tinh o Phan 1)"))

    low_all = text.lower()
    bonus = "thưởng" in low_all and any(v in low_all for v in ACTION_VERBS)
    approved = "phê duyệt" in low_all or "truong phong" in low_all.replace("trưởng phòng", "truong phong")
    out.append(("Thuong giu chan ghi ro phe duyet",
                (not bonus) or approved,
                "de xuat thuong nhung thieu cau 'can truong phong phe duyet'" if bonus and not approved
                else ("da ghi phe duyet" if bonus else "khong de xuat thuong")))

    nwords = len(text.split())
    out.append((f"Do dai khoang 200 chu (dang co {nwords} chu)",
                100 <= nwords <= 300,
                "trong khoang hop ly" if 100 <= nwords <= 300
                else ("qua ngan" if nwords < 100 else "qua dai, nen rut gon")))
    return out
