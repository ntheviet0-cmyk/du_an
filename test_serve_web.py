import io
import json
import os
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest.mock import patch

import serve_web


PROFILE = {
    "band": "red",
    "profile": {
        "OverTime": "Yes",
        "JobSatisfaction": 2,
        "EnvironmentSatisfaction": 3,
        "WorkLifeBalance": 2,
        "YearsAtCompany": 1,
        "StockOptionLevel": 0,
        "incomeBelowDept": True,
        "Age": 29,
    },
}

REPORT = {
    "executive_summary": "Nguy cơ định tính cao do làm thêm giờ và hài lòng công việc thấp.",
    "risk_assessment": {"level": "Cao", "explanation": "Đây là đánh giá định tính từ hồ sơ HR."},
    "factor_analysis": [{"factor": "Làm thêm giờ", "value": "Có", "assessment": "Rủi ro", "note": "Cần xác minh số giờ."}],
    "root_causes": ["Khối lượng công việc cần được trao đổi."],
    "action_plan": [{"action": "Rà soát lịch làm thêm", "reason": "Nhân viên đang làm thêm giờ và cân bằng công việc thấp.",
                     "policy": "Điều 1", "owner": "Quản lý trực tiếp",
                     "deadline": "Trong tuần", "expected_result": "Có phương án nghỉ bù", "approval": "Không"}],
    "not_recommended": ["Không cam kết tăng lương."],
    "missing_data": ["Số giờ làm thêm thực tế."],
    "monitoring": {"one_month": ["Số giờ làm thêm"], "three_months": ["Điểm hài lòng công việc"]},
}


class GeminiSuggestionTests(unittest.TestCase):
    def test_prompt_only_contains_work_facts_and_key_is_not_in_url(self):
        answer = {"candidates": [{"content": {"parts": [{"text": json.dumps(REPORT, ensure_ascii=False)}]}}]}
        with patch.dict(os.environ, {"GEMINI_API_KEY": "dummy-test-key", "GEMINI_MODEL": "gemini-2.5-flash-lite"}):
            with patch.object(serve_web.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(answer).encode())) as call:
                report = serve_web.generate_report(PROFILE)
        self.assertEqual(report["action_plan"][0]["action"], "Rà soát lịch làm thêm")
        req = call.call_args.args[0]
        self.assertNotIn("dummy-test-key", req.full_url)
        self.assertEqual(req.get_header("X-goog-api-key"), "dummy-test-key")
        body = json.loads(req.data)
        sent = body["contents"][0]["parts"][0]["text"]
        self.assertNotIn("Age", sent)
        self.assertIn("Làm thêm giờ: Có", sent)
        generation = body["generationConfig"]
        self.assertEqual(generation["thinkingConfig"], {"thinkingBudget": 0})
        self.assertEqual(generation["responseMimeType"], "application/json")
        self.assertGreaterEqual(generation["maxOutputTokens"], 3000)
        self.assertEqual(generation["responseSchema"]["properties"]["action_plan"]["type"], "ARRAY")

    def call_with_text(self, text, finish="STOP"):
        answer = {"candidates": [{"finishReason": finish, "content": {"parts": [{"text": text}]}}]}
        with patch.dict(os.environ, {"GEMINI_API_KEY": "dummy-test-key"}):
            with patch.object(serve_web.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(answer).encode())):
                return serve_web.generate_report(PROFILE)

    def test_code_fenced_json_is_accepted(self):
        body = json.dumps(REPORT, ensure_ascii=False)
        report = self.call_with_text("```json\n" + body + "\n```")
        self.assertEqual(report["risk_assessment"]["level"], "Cao")

    def test_truncated_json_reports_token_limit(self):
        with self.assertRaises(RuntimeError) as captured:
            self.call_with_text('{\n  "executive_summary": "dang', finish="MAX_TOKENS")
        self.assertIn("giới hạn token", str(captured.exception))

    def test_http_endpoint_reports_missing_key_without_rule_fallback(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), serve_web.Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/api/suggest"
            request = urllib.request.Request(url, data=json.dumps(PROFILE).encode(), headers={"Content-Type": "application/json"})
            with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
                with self.assertRaises(urllib.error.HTTPError) as captured:
                    urllib.request.urlopen(request, timeout=5)
            self.assertEqual(captured.exception.code, 503)
            self.assertIn("GEMINI_API_KEY", json.load(captured.exception)["error"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_http_endpoint_returns_only_generated_report(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), serve_web.Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/api/suggest"
            request = urllib.request.Request(url, data=json.dumps(PROFILE).encode(), headers={"Content-Type": "application/json"})
            with patch.object(serve_web, "generate_report", return_value=REPORT) as call:
                with urllib.request.urlopen(request, timeout=5) as response:
                    payload = json.load(response)
                    self.assertEqual(payload["report"], REPORT)
                    self.assertEqual(payload["actions"][0]["action"], REPORT["action_plan"][0]["action"])
            call.assert_called_once_with(PROFILE)
            with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/", timeout=5) as response:
                self.assertIn(b"profile-result", response.read())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
