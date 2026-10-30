"""ui/vocab.py - Tu vung tieng Viet cho HR + goi y hanh dong theo quy tac."""
import json
from pathlib import Path

REP = Path(__file__).resolve().parent.parent / "reports" / "ibm"

_DICT = None


def hr_dict():
    global _DICT
    if _DICT is None:
        try:
            _DICT = json.loads((REP / "hr_dictionary.json").read_text(encoding="utf-8"))
        except Exception:
            _DICT = {}
    return _DICT


def vn(feat):
    """cat__OverTime_Yes -> 'Làm thêm giờ'; giu nguyen neu khong co trong tu dien."""
    return hr_dict().get(feat, feat)


# --- goi y hanh dong theo quy tac (uu tien giam dan) ---
def suggest_actions(row):
    """row: dict thong tin nhan vien (tieng Viet khoa goc). Tra ve list (hanh dong, ly do)."""
    acts = []
    if row.get("OverTime") == "Yes":
        acts.append(("Giảm làm thêm giờ cho nhân viên này",
                     "làm thêm giờ làm tăng ~21 điểm % nguy cơ nghỉ (phân tích nhân quả)"))
    if row.get("JobSatisfaction", 3) <= 2:
        acts.append(("Trao đổi 1-1 trong tuần này, rà soát khối lượng công việc",
                     "mức hài lòng công việc đang thấp"))
    if row.get("MonthlyIncome", 0) < row.get("_med_income_dept", 0):
        acts.append(("Rà soát lương so với mặt bằng phòng ban",
                     "thu nhập đang dưới trung vị phòng"))
    if row.get("YearsAtCompany", 99) <= 2:
        acts.append(("Đưa vào chương trình buddy/onboarding tăng cường",
                     "nhóm 0–2 năm có tỷ lệ nghỉ cao nhất"))
    if row.get("StockOptionLevel", 1) == 0:
        acts.append(("Xem xét quyền chọn cổ phiếu / thưởng giữ chân",
                     "chưa có ESOP là yếu tố rủi ro"))
    if row.get("WorkLifeBalance", 3) <= 2:
        acts.append(("Linh hoạt giờ làm / hybrid 1–2 ngày/tuần",
                     "cân bằng công việc - cuộc sống thấp"))
    if row.get("DistanceFromHome", 0) > 15:
        acts.append(("Hỗ trợ đi lại hoặc hybrid",
                     "khoảng cách tới công ty lớn"))
    if row.get("NumCompaniesWorked", 0) >= 4:
        acts.append(("Trao đổi lộ trình thăng tiến rõ ràng trong 12 tháng",
                     "lịch sử nhảy việc dày"))
    if row.get("EnvironmentSatisfaction", 3) <= 2:
        acts.append(("Cải thiện môi trường/team (đổi nhóm, không gian làm việc)",
                     "hài lòng môi trường thấp"))
    return acts[:3]


# --- thang chu cho end user (thay slider so 1-4) ---
SCALE_4 = {1: "Rất thấp", 2: "Thấp", 3: "Cao", 4: "Rất cao"}
SCALE_EDU = {1: "Dưới cao đẳng", 2: "Cao đẳng", 3: "Đại học", 4: "Thạc sĩ", 5: "Tiến sĩ"}
SCALE_LEVEL = {1: "Nhân viên", 2: "Chuyên viên", 3: "Chuyên viên cao cấp", 4: "Quản lý", 5: "Lãnh đạo"}
SCALE_PERF = {3: "Đạt yêu cầu", 4: "Vượt trội"}
SCALE_STOCK = {0: "Chưa có", 1: "Ít", 2: "Trung bình", 3: "Nhiều"}


def scale_options(scale):
    return [f"{v} ({k})" for k, v in scale.items()]


def scale_value(option):
    return int(option.rsplit("(", 1)[1].rstrip(")"))


def risk_sentence(proba, top_factors_vn):
    if proba >= 0.8:
        head = "Nguy cơ RẤT CAO"
    elif proba >= 0.48:
        head = "Nguy cơ TRUNG BÌNH, cần theo dõi"
    else:
        head = "Đang ỔN ĐỊNH"
    return f"{head} ({proba * 100:.0f}%) — chủ yếu do: {', '.join(top_factors_vn[:3])}."
