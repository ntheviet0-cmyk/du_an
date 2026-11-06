"""Serve the browser demo and proxy retention suggestions to Gemini locally."""

import json
import os
import re
import socket
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB = Path(__file__).resolve().parent / "web"
RULES = (
    "1. Hạn chế làm thêm giờ: tối đa 8 giờ làm thêm mỗi tuần, ưu tiên nghỉ bù. "
    "2. Xem lại thu nhập 6 tháng một lần theo mặt bằng phòng ban. "
    "3. Nhân viên mới dưới 2 năm được bố trí người hướng dẫn trong 3 tháng đầu. "
    "4. Linh hoạt giờ làm 1 đến 2 ngày mỗi tuần khi công việc cho phép. "
    "5. Thưởng giữ chân phải được trưởng phòng phê duyệt."
)
SYSTEM = (
    "Bạn là chuyên viên tư vấn nhân sự (HRBP) giàu kinh nghiệm về giữ chân nhân tài. "
    "Viết tiếng Việt chuyên nghiệp, ngắn gọn, ưu tiên bảng và gạch đầu dòng trong các trường JSON. "
    "Chỉ dựa trên dữ kiện hồ sơ và nội quy được cung cấp; không suy diễn, không bịa số liệu, không hứa kết quả. "
    "Không suy luận từ tuổi, giới tính, tình trạng gia đình hoặc đời tư. "
    "Không dùng các cụm 'model cho thấy', 'đóng góp vào dự báo', hoặc 'SHAP'. "
    "Dùng cụm 'yếu tố rủi ro từ hồ sơ' khi cần diễn giải. "
    "Đánh giá rủi ro là định tính dựa trên dữ liệu HR đã gửi, không phải giải thích của mô hình dự báo. "
    "Điểm hài lòng 1-2 là rủi ro cao, 3 là trung tính, 4 là điểm mạnh. "
    "Nêu 1-2 yếu tố rủi ro nổi bật và yếu tố bù trừ nếu có. "
    "Đối chiếu mức cần giữ chân với rủi ro; đánh dấu ưu tiên khẩn cấp khi cả hai đều cao. "
    "Chỉ đề xuất hành động có căn cứ từ dữ liệu và nội quy, xếp theo ưu tiên, tối đa 5 hành động. "
    "Mỗi hành động phải ghi lý do gắn trực tiếp với dữ kiện hồ sơ, điều khoản nội quy, người chịu trách nhiệm, thời hạn, kết quả kỳ vọng và phê duyệt cần có. "
    "Với Điều 2 chỉ được ghi 'đưa vào xem xét thu nhập kỳ tới', không khẳng định tăng lương. "
    "Với Điều 5, thưởng giữ chân luôn cần trưởng phòng phê duyệt. "
    "Làm thêm giờ = Có chưa cho biết số giờ; yêu cầu xác minh có vượt 8 giờ/tuần trước khi kết luận. "
    "Thâm niên dưới 2 năm thì xét Điều 3; từ 2 năm trở lên phải nêu rõ Điều 3 không áp dụng. "
    "Điều 4 chỉ áp dụng khi công việc cho phép và cần nêu điều kiện cần xác nhận. "
    "Chỉ số theo dõi chỉ nêu chỉ số cần đo lại, không đặt mục tiêu số cụ thể. "
    "Chỉ trả JSON đúng schema, không kèm markdown hoặc văn bản ngoài JSON."
)
SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "executive_summary": {"type": "STRING"},
        "risk_assessment": {
            "type": "OBJECT",
            "properties": {"level": {"type": "STRING"}, "explanation": {"type": "STRING"}},
            "required": ["level", "explanation"],
        },
        "factor_analysis": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "factor": {"type": "STRING"}, "value": {"type": "STRING"},
                    "assessment": {"type": "STRING"}, "note": {"type": "STRING"},
                },
                "required": ["factor", "value", "assessment", "note"],
            },
        },
        "root_causes": {"type": "ARRAY", "items": {"type": "STRING"}},
        "action_plan": {
            "type": "ARRAY", "maxItems": 5,
            "items": {"type": "OBJECT", "properties": {
                "action": {"type": "STRING"}, "reason": {"type": "STRING"},
                "policy": {"type": "STRING"},
                "owner": {"type": "STRING"}, "deadline": {"type": "STRING"},
                "expected_result": {"type": "STRING"}, "approval": {"type": "STRING"},
            }, "required": ["action", "reason", "policy", "owner", "deadline", "expected_result", "approval"]},
        },
        "not_recommended": {"type": "ARRAY", "items": {"type": "STRING"}},
        "missing_data": {"type": "ARRAY", "items": {"type": "STRING"}},
        "monitoring": {"type": "OBJECT", "properties": {
            "one_month": {"type": "ARRAY", "items": {"type": "STRING"}},
            "three_months": {"type": "ARRAY", "items": {"type": "STRING"}},
        }, "required": ["one_month", "three_months"]},
    },
    "required": ["executive_summary", "risk_assessment", "factor_analysis", "root_causes", "action_plan",
                 "not_recommended", "missing_data", "monitoring"],
}
FIELDS = {
    "OverTime": ("Làm thêm giờ", {"Yes": "Có", "No": "Không"}),
    "JobSatisfaction": ("Hài lòng công việc (1 thấp, 4 cao)", (1, 4)),
    "EnvironmentSatisfaction": ("Hài lòng môi trường (1 thấp, 4 cao)", (1, 4)),
    "WorkLifeBalance": ("Cân bằng công việc/cuộc sống (1 thấp, 4 cao)", (1, 4)),
    "YearsAtCompany": ("Thâm niên (năm)", (0, 40)),
    "StockOptionLevel": ("Thưởng gắn bó (0 là chưa có)", (0, 3)),
    "incomeBelowDept": ("Thu nhập thấp hơn trung vị phòng ban", {True: "Có", False: "Không"}),
}


def profile_context(payload):
    if not isinstance(payload, dict) or payload.get("band") not in ("red", "amber", "green"):
        raise ValueError("Thông tin mức nguy cơ không hợp lệ.")
    profile = payload.get("profile")
    if not isinstance(profile, dict):
        raise ValueError("Thiếu thông tin công việc.")
    lines = []
    for field, (label, allowed) in FIELDS.items():
        value = profile.get(field)
        if isinstance(allowed, tuple):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not allowed[0] <= value <= allowed[1]:
                raise ValueError("Thông tin công việc không hợp lệ.")
            value = str(value)
        else:
            if type(value) not in (str, bool) or value not in allowed:
                raise ValueError("Thông tin công việc không hợp lệ.")
            value = allowed[value]
        lines.append(f"{label}: {value}")
    band = {"red": "Cao", "amber": "Trung bình", "green": "Thấp"}[payload["band"]]
    return f"Nội quy: {RULES}\nMức cần giữ chân: {band}\n" + "\n".join(lines)


def parse_report(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[A-Za-z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise
        result = json.JSONDecoder().raw_decode(text[start:])[0]
    def text_field(value, maximum=500):
        if not isinstance(value, str) or not 1 <= len(value.strip()) <= maximum:
            raise ValueError("invalid report field")
        return value.strip()

    def text_list(value, maximum_items=10):
        if not isinstance(value, list) or len(value) > maximum_items:
            raise ValueError("invalid report list")
        return [text_field(item) for item in value]

    report = {
        "executive_summary": text_field(result.get("executive_summary"), 1200),
        "risk_assessment": result.get("risk_assessment"),
        "factor_analysis": result.get("factor_analysis"),
        "root_causes": text_list(result.get("root_causes"), 3),
        "action_plan": result.get("action_plan"),
        "not_recommended": text_list(result.get("not_recommended"), 10),
        "missing_data": text_list(result.get("missing_data"), 10),
        "monitoring": result.get("monitoring"),
    }
    risk = report["risk_assessment"]
    if not isinstance(risk, dict):
        raise ValueError("invalid risk assessment")
    report["risk_assessment"] = {"level": text_field(risk.get("level"), 50),
                                 "explanation": text_field(risk.get("explanation"), 800)}
    if not isinstance(report["factor_analysis"], list) or len(report["factor_analysis"]) > 10:
        raise ValueError("invalid factor analysis")
    if not all(isinstance(item, dict) for item in report["factor_analysis"]):
        raise ValueError("invalid factor analysis")
    report["factor_analysis"] = [{field: text_field(item.get(field), 400)
                                  for field in ("factor", "value", "assessment", "note")}
                                 for item in report["factor_analysis"]]
    if not isinstance(report["action_plan"], list) or len(report["action_plan"]) > 5:
        raise ValueError("invalid action plan")
    action_fields = ("action", "reason", "policy", "owner", "deadline", "expected_result", "approval")
    if not all(isinstance(item, dict) for item in report["action_plan"]):
        raise ValueError("invalid action plan")
    report["action_plan"] = [{field: text_field(item.get(field), 500) for field in action_fields}
                             for item in report["action_plan"]]
    monitoring = report["monitoring"]
    if not isinstance(monitoring, dict):
        raise ValueError("invalid monitoring")
    report["monitoring"] = {"one_month": text_list(monitoring.get("one_month"), 10),
                            "three_months": text_list(monitoring.get("three_months"), 10)}
    return report


def generate_report(payload):
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Chưa đặt GEMINI_API_KEY cho máy chủ web.")
    model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", model):
        raise RuntimeError("MODEL không hợp lệ.")
    generation = {
        "temperature": 0.2,
        "maxOutputTokens": 3000,
        "responseMimeType": "application/json",
        "responseSchema": SCHEMA,
    }
    if "2.5" in model and "pro" not in model:
        generation["thinkingConfig"] = {"thinkingBudget": 0}
    request_body = {
        "systemInstruction": {"parts": [{"text": SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": profile_context(payload)}]}],
        "generationConfig": generation,
    }
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise RuntimeError("đã hết hạn mức lúc này. Vui lòng thử lại sau.") from exc
        if exc.code in (401, 403):
            raise RuntimeError("API key không hợp lệ hoặc chưa được cấp quyền.") from exc
        raise RuntimeError(f"trả lỗi HTTP {exc.code}. Kiểm tra tên model và hạn mức API.") from exc
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        raise RuntimeError("Không kết nối được hoặc yêu cầu đã quá 30 giây.") from exc
    finish = ""
    try:
        candidates = data.get("candidates") or []
        if not candidates:
            block = (data.get("promptFeedback") or {}).get("blockReason")
            raise RuntimeError(f"đã chặn yêu cầu (lý do: {block}). Hãy thử lại." if block else "không trả về nội dung. Hãy thử lại.")
        candidate = candidates[0]
        finish = candidate.get("finishReason", "")
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
        if not text.strip():
            if finish == "MAX_TOKENS":
                raise RuntimeError("bị cắt do giới hạn token trước khi trả JSON. Hãy thử lại.")
            raise RuntimeError("không trả về nội dung. Hãy thử lại.")
        return parse_report(text)
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        if finish == "MAX_TOKENS":
            raise RuntimeError("bị cắt do giới hạn token trước khi trả JSON. Hãy thử lại.") from exc
        raise RuntimeError("chưa trả về gợi ý hợp lệ. Hãy thử lại.") from exc


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def end_headers(self):
        # The UI and API schema evolve together; never reuse an old renderer.
        if self.path.split("?", 1)[0] in ("/", "/index.html", "/app.js", "/policy.js"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        if self.path != "/api/suggest":
            self.send_error(404)
            return
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{self.headers.get('Host')}":
            self.send_error(403)
            return
        try:
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Yêu cầu phải là JSON.")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 8192:
                raise ValueError("Dữ liệu gửi lên không hợp lệ.")
            payload = json.loads(self.rfile.read(length))
            context = profile_context(payload)
            if not context:
                raise ValueError("Thiếu dữ kiện.")
            report = generate_report(payload)
            # Keep already-open tabs on the former action-only contract working.
            legacy_actions = [
                {
                    "action": item["action"],
                    "reason": item["reason"],
                }
                for item in report["action_plan"]
            ]
            self.send_json(200, {"report": report, "actions": legacy_actions})
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json(400, {"error": str(exc)})
        except RuntimeError as exc:
            self.send_json(503, {"error": str(exc)})

    def send_json(self, status, value):
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    port = int(os.environ.get("HR_WEB_PORT", "5501"))
    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"Mở http://127.0.0.1:{port}/ (Ctrl+C để dừng)")
        server.serve_forever()
