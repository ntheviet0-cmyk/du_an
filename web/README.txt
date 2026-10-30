HR GIỮ CHÂN — WEB DEMO TĨNH
==============================
Để nhận gợi ý "Việc nên làm" từ Gemini, chạy máy chủ cục bộ ở thư mục dự án:
  PowerShell: $env:GEMINI_API_KEY = Read-Host "Dán API key Gemini"
  PowerShell: python serve_web.py
  Mở http://127.0.0.1:5501/ (không mở cổng 5500 của Live Server).
Có thể chọn model bằng GEMINI_MODEL (mặc định gemini-2.5-flash).
API key chỉ nằm trong biến môi trường của phiên PowerShell; không đưa vào web/app.js.
Nếu mở file:// hoặc Live Server, phần chấm điểm vẫn chạy nhưng gợi ý Gemini
sẽ thông báo cần chạy serve_web.py. Cần kết nối mạng cho gợi ý Gemini.

6 mục ở thanh điều hướng dọc bên trái: Tổng quan | Cần giữ chân ngay | Nhân viên |
Nguyên nhân nghỉ việc | Chính sách, kèm nút "Độ tin cậy" và công tắc Giải thích.
Nút mũi tên ở đầu thanh thu gọn thanh lại thành dải icon (trạng thái được ghi nhớ);
trên màn hình nhỏ, thanh này ẩn vào nút menu ba gạch ở góc trên bên trái.

Tính năng mới:
- Điểm nguy cơ thật: chạy suy luận GraphSAGE ngay trong trình duyệt (sage.js,
  đọc trọng số từ data/weights.js, đồ thị từ data/edges.js). Trang "Nhân viên" và
  hồ sơ chi tiết đều dùng điểm này; nếu thiếu file dữ liệu thì tự lùi về quy tắc.
- Theo dõi xử lý: mỗi người có trạng thái Chưa liên hệ / Đã gặp / Đã xử lý + ghi chú,
  lưu cục bộ trên máy (localStorage). Có tiến độ ở Tổng quan, lọc theo trạng thái,
  cột trạng thái + ghi chú khi xuất CSV. Xóa sạch ở trang "Độ tin cậy".
- Danh sách: tìm theo mã NV-xxxx, lọc mức/phòng/thâm niên/trạng thái, bấm "Xem thêm",
  nút in tự mở rộng toàn bộ danh sách đang lọc.
- Chính sách: so sánh phương án A/B, nút "Lấy tóm tắt mang đi họp", số liệu lấy từ
  KPI.policy_rates (giảm làm thêm giờ là kết quả nhân quả đã kiểm định; hai chính sách
  còn lại là chênh lệch quan sát được).
- Font Be Vietnam Pro tự host trong assets/fonts (không gọi mạng).

Lưu ý trung thực khi demo:
- Điểm nguy cơ là điểm AI thật (GraphSAGE best-tune) cho 294 hồ sơ mẫu; form nhập tay
  cũng dùng đúng mô hình đó, đã đối chiếu khớp bản Python (sai số < 0,001).
- Mọi ảnh SHAP/ROC/ma trận đều từ pipeline Python đã kiểm chứng.

Kiểm tra nhanh (tùy chọn, khi sửa code):
- python qa_web.py            -> QA_STATIC_OK (ID, file, nav, font, policy, parity)
- node web/test_sage_parity.js -> PARITY_OK (JS <-> Python)
