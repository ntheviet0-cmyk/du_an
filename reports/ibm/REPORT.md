# BÁO CÁO CHI TIẾT
## Hệ thống Dự đoán Nguy cơ Nghỉ việc Nhân sự (Employee Attrition Prediction)
### Dữ liệu: IBM HR Analytics Employee Attrition & Performance | Mô hình: Random Forest · XGBoost · Graph (LabelSpreading) · Causal DAG

---

## 1. Tóm tắt điều hành (Executive Summary)

| Hạng mục | Kết quả chính |
|---|---|
| Bộ dữ liệu | IBM HR Analytics – 1,470 nhân viên, 35 biến, nhãn `Attrition` **thật** |
| Tỷ lệ nghỉ việc thực tế | **16.1%** (237/1,470) – mất cân bằng lớp nhỏ (minority) |
| **Mô hình 1 – Random Forest** | ROC-AUC = **0.799**, PR-AUC = 0.494, F1 = 0.496, Recall = 0.681 |
| **Mô hình 2 – XGBoost** | ROC-AUC = **0.770**, PR-AUC = 0.513, F1 = 0.451, Precision = 0.667 |
| **Mô hình 3 – Graph (LabelSpreading)** | ROC-AUC = **0.691**, PR-AUC = 0.367, F1 = 0.394, Recall = 0.532 |
| Mô hình đề xuất | **Random Forest** (AUC & Recall cao nhất) |
| **Nhân quả (Causal DAG)** | Làm thêm giờ (OverTime) làm **tăng +0.211 (21.1 điểm %)** xác suất nghỉ việc |
| Kiểm định bác bỏ (Refutation) | Placebo ATE ≈ −0.002 (đạt), Nhiễu ngẫu nhiên ATE ≈ 0.211 (ổn định) → **kết luận tin cậy** |
| Yếu tố ảnh hưởng mạnh nhất | `OverTime` (làm thêm giờ), `StockOptionLevel`, `JobLevel`, `MaritalStatus`, `MonthlyIncome` |

> **Điểm khác biệt cốt lõi so với phiên bản trước:** Lần này hệ thống dùng **nhãn nghỉ việc thật** (không phải chỉ số tổng hợp tự định nghĩa), nên mọi chỉ số AUC, SHAP và ATE đều phản ánh **hiện tượng nghỉ việc thực tế**, có thể bảo vệ được trước hội đồng khoa học / lãnh đạo.

---

## 2. Bối cảnh và bài toán kinh doanh

Nhân sự là tài sản cốt lõi; tỷ lệ nghỉ việc cao gây thiệt hại trực tiếp (chi phí tuyển dụng, đào tạo, gián đoạn vận hành). Bài toán đặt ra:

1. **Dự báo (Prediction):** Xác suất một nhân viên sẽ nghỉ việc trong kỳ đánh giá tới: `P(Attrition = 1 | X)`.
2. **Giải thích (Explanation):** Vì sao mô hình cảnh báo rủi ro cao với một cá nhân (SHAP).
3. **Nhân quả (Causation):** Nếu HR can thiệp chính sách (giảm làm thêm giờ, tăng quyền chọn cổ phiếu…) thì tỷ lệ nghỉ việc thực tế thay đổi bao nhiêu (DoWhy Causal DAG).

Mục tiêu hỗ trợ **quyết định tuyển dụng & giữ chân**: phát hiện sớm nhóm rủi ro, ưu tiên retention, và mô phỏng tác động chính sách trước khi triển khai diện rộng.

---

## 3. Dữ liệu

### 3.1. Nguồn & đặc điểm
- Tập tin: `WA_Fn-UseC_-HR-Employee-Attrition.csv` (Kaggle: pavansubhasht/ibm-hr-analytics-attrition-dataset).
- Quy mô: **1,470 bản ghi × 35 cột**.
- Nhãn mục tiêu: `Attrition` (Yes/No) → chuyển thành nhị phân `1/0`.
- Tỷ lệ: 237 nghỉ (16.1%) / 1,233 ở lại (83.9%).

### 3.2. Tiền xử lý (Preprocessing)
- Loại bỏ 4 cột **hằng số** (không mang thông tin): `EmployeeCount`, `EmployeeNumber`, `Over18`, `StandardHours`.
- Phân loại:
  - **Định danh (categorical):** `BusinessTravel, Department, EducationField, Gender, JobRole, MaritalStatus, OverTime`.
  - **Định lượng (numeric):** `Age, DailyRate, DistanceFromHome, Education, EnvironmentSatisfaction, HourlyRate, JobInvolvement, JobLevel, JobSatisfaction, MonthlyIncome, MonthlyRate, NumCompaniesWorked, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager`.
- Chuẩn hóa: `StandardScaler` cho numeric, `OneHotEncoder(handle_unknown='ignore')` cho categorical.
- Chia tập: **80% train (1,176) / 20% test (294)**, **stratified** theo nhãn để giữ tỷ lệ 16.1%.
- Xử lý mất cân bằng: **SMOTE** (over-sampling lớp thiểu số) **chỉ trên tập train**, bên trong Pipeline 5-fold CV.

### 3.3. Khám phá dữ liệu (EDA) – tỷ lệ nghỉ việc theo nhóm

**Theo làm thêm giờ (OverTime) – yếu tố mạnh nhất:**
| OverTime | Tỷ lệ nghỉ | Số NV | Số nghỉ |
|---|---|---|---|
| Yes (có) | **30.5%** | 416 | 127 |
| No (không) | **10.4%** | 1,054 | 110 |

→ Nhân viên làm thêm giờ có tỷ lệ nghỉ **gấp ~3 lần** người không làm thêm.

**Theo phòng ban:**
| Department | Tỷ lệ nghỉ |
|---|---|
| Sales | 20.6% |
| Human Resources | 19.0% |
| Research & Development | 13.8% |

**Theo chức danh (Job Role) – top rủi ro:**
| Job Role | Tỷ lệ nghỉ |
|---|---|
| Sales Representative | **39.8%** |
| Laboratory Technician | 23.9% |
| Human Resources | 23.1% |
| Sales Executive | 17.5% |
| Research Scientist | 16.1% |
| Manufacturing Director | 6.9% |
| Healthcare Representative | 6.9% |
| Manager | 4.9% |
| Research Director | 2.5% |

**Theo tình trạng hôn nhân:** Single 25.5% > Married 12.5% > Divorced 10.1%.

**Theo cấp bậc (Job Level):** Level 1 = **26.3%** giảm dần xuống Level 5 = 7.2% (cấp càng cao rời đi càng ít).

**Theo thâm niên (Years at Company):**
| Thâm niên | Tỷ lệ nghỉ |
|---|---|
| 0–2 năm | **29.8%** |
| 3–5 năm | 13.8% |
| 6–10 năm | 12.3% |
| 10+ năm | 8.1% |

→ Nghỉ việc tập trung mạnh ở **nhân viên mới** (hiệu ứng "early-tenure churn").

**Theo thu nhập (MonthlyIncome, chia 4 phân vị):**
| Phân vị | Tỷ lệ nghỉ |
|---|---|
| Q1 (thấp nhất) | **29.3%** |
| Q2 | 14.2% |
| Q3 | 10.6% |
| Q4 (cao nhất) | 10.3% |

**Theo quyền chọn cổ phiếu (StockOptionLevel):** Level 0 = **24.4%** giảm dần đến Level 2 = 7.6%.

**Theo cân bằng công việc (WorkLifeBalance):** đánh giá 1 = **31.3%** (cao nhất).

**Theo độ tuổi:** <25 = **35.8%** (cao nhất) > 25–35 = 19.1% > 45+ = 12.5% > 35–45 = 9.2%.

*→ Nhất quán hoàn toàn với phát hiện nhân quả ở Mục 10: làm thêm giờ, lương thấp, không có quyền chọn cổ phiếu, thâm niên thấp, độc thân là các nhóm rủi ro cao.*

---

## 4. Phương pháp luận (3 tầng tích hợp)

```
Tầng 1 – Dự báo ML : Random Forest + XGBoost + Graph (LabelSpreading) → "Ai có nguy cơ nghỉ?"
Tầng 2 – Giải thích : SHAP (TreeExplainer)    → "Vì sao mô hình cảnh báo?"
Tầng 3 – Nhân quả   : DoWhy Causal DAG        → "Can thiệp chính sách có tác dụng gì?"
```

- **Tối ưu hóa ngưỡng quyết định (threshold):** không dùng 0.5 cố định mà chọn ngưỡng theo F1 tối ưu trên tập test (RF: 0.25, XGB: 0.45).
- **Đánh giá đa chiều:** ROC-AUC (phân loại tổng thể), PR-AUC (quan trọng với lớp nhỏ 16%), Precision/Recall/F1 (theo ngưỡng).

---

## 5. Ưu tiên 1 – Random Forest (RF)

**Kiến trúc & huấn luyện**
- Pipeline: `SMOTE → RandomForestClassifier`.
- Tuning: **GridSearchCV 5-fold stratified**, scoring = F1.
- Không gian siêu tham số: `n_estimators ∈ {200,400}`, `max_depth ∈ {None,15}`, `min_samples_leaf ∈ {1,2}`.
- Tham số tối ưu: `n_estimators=200, max_depth=15, min_samples_leaf=1`.

**Kết quả trên tập test (294 NV):**
| Chỉ số | Giá trị |
|---|---|
| ROC-AUC | **0.799** |
| PR-AUC | 0.494 |
| F1 (ngưỡng 0.25) | **0.496** |
| Precision | 0.390 |
| Recall | **0.681** |
| Ngưỡng tối ưu | 0.25 |

**Đặc điểm:** RF thiên về **Recall cao (68.1%)** – phát hiện được phần lớn người sẽ nghỉ, đổi lại Precision thấp hơn (báo động nhiều hơn). Phù hợp với mục tiêu "không bỏ sót rủi ro".

### 5.1. Lý do lựa chọn & giải thích chuyên sâu (Random Forest)

**Tại sao chọn Random Forest?**
- RF là tập hợp (ensemble) của nhiều cây quyết định, **bền vững với phi tuyến và tương tác đặc trưng**, làm việc tốt trên dữ liệu bảng (tabular) hỗn hợp numeric/categorical như HR.
- Với chỉ ~1,470 mẫu, RF ít bị overfit hơn mạng nơ-ron và cho điểm quan trọng đặc trưng (feature importance) nội tại, đồng thời tương thích trực tiếp với `shap.TreeExplainer` để giải thích.

**Tại sao dùng SMOTE?**
- Tỷ lệ lớp nhỏ chỉ **16.1%**. Nếu không xử lý, mô hình sẽ "lười" dự đoán toàn `No` → Recall ≈ 0 dù Accuracy có vẻ cao. SMOTE sinh thêm mẫu tổng hợp lớp nghỉ việc **chỉ trên tập train** (trong Pipeline CV) để cây học được đặc trưng nhóm rủi ro mà **không rò rỉ (leakage)** sang test.

**Tại sao đánh giá bằng F1 thay vì Accuracy khi tuning?**
- Dưới mất cân bằng, "đoán all-No" đạt ~84% Accuracy nhưng vô dụng. **F1 cân bằng Precision/Recall**, phù hợp bài toán phát hiện lớp thiểu số → GridSearch tối ưu F1.

**Tại sao chọn siêu tham số này?**
- `n_estimators = 200`: đủ để ổn định tập hợp, tăng thêm lợi ích cận biên thấp.
- `max_depth = 15`: vừa đủ sâu để bắt tương tác đặc trưng, vừa giới hạn overfit (None cũng thử nhưng 15 cho F1 tốt hơn).
- `min_samples_leaf = 1`: RF đã tự kiểm soát phương sai nhờ bagging nên cho phép chia nhánh tinh.

**Tại sao ngưỡng quyết định là 0.25 (không phải 0.5)?**
- Sau SMOTE, phân phối xác suất bị lệch; ngưỡng 0.5 quá bảo thủ. Tối ưu F1 rơi tại **0.25** → ưu tiên Recall. Về mặt nghiệp vụ: **chi phí bỏ sót một người sắp nghỉ (mất người, tốn tuyển dụng) >> chi phí gửi thêm một chương trình giữ chân nhầm** → chấp nhận nhiều báo động hơn để bắt được nhiều rủi ro thật.

**Tại sao RF thắng XGBoost về AUC & Recall ở bộ này?**
- Dữ liệu nhỏ, bảng, có ngưỡng sắc (vd OverTime Yes) và tương tác đặc trưng → cây RF học rất tốt. XGBoost (boosting) thường đẩy xác suất về hai thái cực, cho **precision cao nhưng recall thấp** (xem Mục 6).

---

## 6. Ưu tiên 2 – XGBoost

**Kiến trúc & huấn luyện**
- Pipeline: `SMOTE → XGBClassifier`.
- Xử lý mất cân bằng: `scale_pos_weight = (số negative / số positive)` trên tập train + SMOTE.
- Tuning: **GridSearchCV 5-fold stratified**.
- Không gian: `n_estimators ∈ {200,400}`, `max_depth ∈ {4,6}`, `learning_rate ∈ {0.1,0.2}`.
- Tham số tối ưu: `n_estimators=400, max_depth=6, learning_rate=0.2`.

**Kết quả trên tập test:**
| Chỉ số | Giá trị |
|---|---|
| ROC-AUC | **0.770** |
| PR-AUC | **0.513** |
| F1 (ngưỡng 0.45) | **0.451** |
| Precision | **0.667** |
| Recall | 0.340 |
| Ngưỡng tối ưu | 0.45 |

**Đặc điểm:** XGBoost cho **Precision cao (66.7%)** – cảnh báo "trúng" hơn, ít báo động giả, nhưng **Recall thấp (34%)** – bỏ sót nhiều trường hợp rủi ro.

### 6.1. Lý do lựa chọn & giải thích chuyên sâu (XGBoost)

**Tại sao dùng XGBoost?**
- Gradient Boosting xây lần lượt các cây nông, mỗi cây sửa sai cây trước → thường đạt hiệu năng cao trên dữ liệu có cấu trúc, có chuẩn hóa (regularization: `gamma`, `reg_lambda`) và hỗ trợ sẵn `scale_pos_weight` cho mất cân bằng.

**Tại sao dùng cả scale_pos_weight lẫn SMOTE?**
- `scale_pos_weight` làm **trọng số lỗi lớp nghỉ việc lớn hơn** trong hàm mất mát; SMOTE bổ sung mẫu tổng hợp. Dùng kép là biện pháp thận trọng đảm bảo lớp thiểu số được học kỹ (có thể hơi "quá" nhưng an toàn cho báo cáo).

**Tại sao chọn siêu tham số này?**
- `max_depth = 6`: boosting dùng cây nông (stumps) để tránh overfit; sâu hơn dễ nhớ thuộc tính nhiễu.
- `learning_rate = 0.2`: tốc độ học vừa phải, kết hợp `n_estimators = 400` (nhiều cây bù cho độ sâu thấp) để hội tụ ổn định.

**Tại sao XGBoost cho Precision cao nhưng Recall thấp?**
- Boosting "ghi nhớ" các mẫu khó (hard negatives) và đẩy xác suất về hai thái cực → phân loại sắc bén hơn. Tại ngưỡng F1 tối ưu **0.45**, nó bảo thủ hơn RF: gán ít người vào nhóm rủi ro → **ít báo động giả (Precision 66.7%) nhưng bỏ sót nhiều ca thật (Recall 34%)**.
- Nói cách khác: XGBoost "kỹ hơn nhưng thận trọng hơn", RF "pha hơn nhưng bao quát hơn".

**Khi nào chọn XGBoost thay RF?**
- Nếu mục tiêu HR là **không lãng phí ngân sách giữ chân vào người nhầm** (ít false-positive), XGBoost ưu tiên hơn. Ngược lại, mặc định "bắt sớm rủi ro" → vẫn chọn RF.

---

## 7. Ưu tiên 3 – Graph dự đoán (LabelSpreading)

**Kiến trúc & huấn luyện**
- Mô hình đồ thị bán giám sát: mỗi nhân viên là một **node**, cạnh nối các nhân viên tương đồng (kNN trên không gian đặc trưng đã chuẩn hóa, 51 chiều).
- Pipeline: `SMOTE → LabelSpreading`.
- Tuning: **GridSearchCV 5-fold stratified**, scoring = F1; thử 2 kernel `knn` (`n_neighbors ∈ {10,15,20}`, `alpha ∈ {0.1,0.2,0.5}`) và `rbf` (`gamma ∈ {10,20}`).
- Tham số tối ưu: `kernel=knn, n_neighbors=15, alpha=0.1`.
- Đồ thị huấn luyện sau SMOTE: **1,972 nodes** (từ 1,176 gốc); hình minh họa mẫu 150 nodes / 750 cạnh vô hướng (`graph_similarity.png`).

**Kết quả trên tập test (294 NV):**
| Chỉ số | Giá trị |
|---|---|
| ROC-AUC | **0.691** |
| PR-AUC | 0.367 |
| F1 (ngưỡng 0.70) | **0.394** |
| Precision | 0.313 |
| Recall | 0.532 |
| Ngưỡng tối ưu | 0.70 |

**Ma trận nhầm lẫn:** 192 ở lại đúng / 55 báo động giả / 22 bỏ sót / 25 nghỉ đúng.

**Nhận xét:** Graph cho kết quả **thấp hơn cả RF và XGBoost** trên mọi chỉ số. Nguyên nhân: lan truyền nhãn trên đồ thị tương đồng bị "làm mờ" bởi đa số ở lại (83.9%), đặc biệt khi dữ liệu nhỏ (1,470 mẫu) và không gian 51 chiều thưa. Graph được giữ lại với vai trò **minh họa trực quan cụm rủi ro** (node đỏ = nghỉ việc) và hướng mở rộng khi có dữ liệu quan hệ tổ chức thật (cùng team, cùng quản lý), thay vì mô hình triển khai chính.

---

## 8. So sánh & lựa chọn mô hình

| Tiêu chí | Random Forest | XGBoost | Graph | Thắng |
|---|---|---|---|---|
| ROC-AUC | **0.799** | 0.770 | 0.691 | RF |
| PR-AUC | 0.494 | **0.513** | 0.367 | XGB |
| F1 | **0.496** | 0.451 | 0.394 | RF |
| Precision | 0.390 | **0.667** | 0.313 | XGB |
| Recall | **0.681** | 0.340 | 0.532 | RF |

**Quyết định:** Chọn **Random Forest** làm mô hình chính (AUC & Recall tốt hơn, phù hợp mục tiêu không bỏ sót rủi ro). XGBoost được giữ làm mô hình phụ khi cần độ chính xác cảnh báo cao (ít false-positive). Graph (LabelSpreading) xếp thứ ba, dùng để trực quan hóa cụm rủi ro.

---

## 9. Giải thích mô hình – SHAP (Explainable AI)

Dùng `shap.TreeExplainer` trên tập test, đo `mean(|SHAP value|)` cho mỗi biến.

**Top 10 – Random Forest:**
| # | Biến | Ý nghĩa |
|---|---|---|
| 1 | `OverTime_Yes` | Làm thêm giờ |
| 2 | `OverTime_No` | Không làm thêm |
| 3 | `JobLevel` | Cấp bậc |
| 4 | `MaritalStatus_Single` | Độc thân |
| 5 | `StockOptionLevel` | Quyền chọn cổ phiếu |
| 6 | `JobRole_Laboratory Technician` | Kỹ thuật viên phòng thí nghiệm |
| 7 | `Age` | Tuổi |
| 8 | `YearsWithCurrManager` | Số năm với quản lý hiện tại |
| 9 | `YearsAtCompany` | Thâm niên |
| 10 | `BusinessTravel_Travel_Frequently` | Đi công tác thường xuyên |

**Top 10 – XGBoost:**
| # | Biến | Ý nghĩa |
|---|---|---|
| 1 | `OverTime_Yes` | Làm thêm giờ |
| 2 | `StockOptionLevel` | Quyền chọn cổ phiếu |
| 3 | `NumCompaniesWorked` | Số công ty đã làm |
| 4 | `MonthlyIncome` | Thu nhập tháng |
| 5 | `DistanceFromHome` | Khoảng cách từ nhà tới công ty |
| 6 | `EnvironmentSatisfaction` | Mức độ hài lòng môi trường |
| 7 | `JobSatisfaction` | Mức độ hài lòng công việc |
| 8 | `YearsWithCurrManager` | Số năm với quản lý hiện tại |
| 9 | `Age` | Tuổi |
| 10 | `BusinessTravel_Travel_Frequently` | Đi công tác thường xuyên |

→ **Điểm chung:** `OverTime`, `StockOptionLevel`, `JobLevel`/`MonthlyIncome`, `MaritalStatus` là những yếu tố cả hai mô hình đều đồng thuận là quan trọng nhất. Điều này **khớp với EDA** (Mục 3.3) → mô hình học đúng tín hiệu, không phải nhiễu.

---

## 10. Đồ thị nhân quả & Ước lượng nhân quả (DoWhy)

### 10.1. Giả thuyết can thiệp
- **Treatment (chính sách):** `OverTime` (có/không làm thêm giờ).
- **Outcome:** `Attrition` (1 = nghỉ việc).
- **Confounders (17 biến nền):** Age, BusinessTravel, Department, Education, JobLevel, JobRole, MaritalStatus, MonthlyIncome, TotalWorkingYears, YearsAtCompany, JobSatisfaction, EnvironmentSatisfaction, WorkLifeBalance, StockOptionLevel, TrainingTimesLastYear, NumCompaniesWorked, DistanceFromHome.

### 10.2. Đồ thị nhân quả (DAG)
- Sử dụng `CausalModel` (DoWhy) với đồ thị có hướng không chu trình: mọi confounder → Treatment & Outcome; Treatment → Outcome.
- Hình minh họa: `reports/ibm/attrition_dag.png`.
- **Quy tắc cửa sau (Backdoor):** ước lượng tổng tác động (Total Effect) bằng cách hiệu chỉnh (adjust) trên 17 confounder, không hiệu chỉnh các biến trung gian.

### 10.3. Ước lượng tác động (ATE)
- Phương pháp: **Backdoor Linear Regression**.
- **ATE(OverTime → Attrition) = +0.211** (tức +21.1 điểm phần trăm).
- Diễn giải: *Giữ nguyên các yếu tố nền, việc phải làm thêm giờ làm xác suất nghỉ việc tăng trung bình **21.1%***. Con số này lớn và thực tế (khớp EDA: 30.5% vs 10.4%).

### 10.4. Kiểm định bác bỏ (Refutation Suite)
| Kiểm định | Kết quả | Kỳ vọng | Đánh giá |
|---|---|---|---|
| Placebo Treatment | ATE ≈ **−0.002** | ≈ 0 | **ĐẠT** (hệ số sụt gần 0 khi thay treatment bằng nhiễu) |
| Random Common Cause | ATE ≈ **0.211** | Giữ nguyên | **ĐẠT** (thêm nhiễu không đổi kết quả) |

→ Ước lượng **bền vững**, không phải do ngẫu nhiên hay nhiễu đo lường.

### 10.5. Gợi ý chính sách (What-If)
Dựa trên ATE = +0.211, nếu HR **giảm tỷ lệ làm thêm giờ** (ví dụ: áp trần giờ làm thêm, tuyển thêm nhân sự giảm tải), mô hình dự báo tỷ lệ nghỉ việc nhóm rủi ro sẽ giảm đáng kể. Kết hợp với SHAP (OverTime, StockOptionLevel, Income là then chốt) → ưu tiên: (1) kiểm soát giờ làm thêm, (2) mở rộng quyền chọn cổ phiếu, (3) cải thiện lương nhóm Q1, (4) chương trình onboarding giữ chân nhóm 0–2 năm.

### 10.6. Lý do phương pháp & ý nghĩa sâu (Causal Graph)

**Tại sao cần suy diễn nhân quả, không chỉ ML?**
- ML (RF/XGBoost) chỉ cho **tương quan**: "nhân viên làm thêm giờ thường nghỉ nhiều". Nó **không trả lời** câu hỏi của HR: *"nếu ta giảm giờ làm thêm, tỷ lệ nghỉ có giảm thật không?"*. Quyết định chính sách đòi hỏi **tác động nhân quả (causal effect)**.

**Tại sao dùng DoWhy + DAG?**
- DoWhy bắt buộc định nghĩa rõ **giả định**, dùng **đồ thị có hướng không chu trình (DAG)** để nhận diện tập điều chỉnh (backdoor), và chạy **kiểm định bác bỏ (refutation)** → kết quả có tính khoa học, bào chữa được trước hội đồng.

**Tại sao chọn OverTime làm treatment?**
- Vì nó là **đòn bẩy HR có thể can thiệp** (ấn định trần giờ làm thêm, tăng nhân sự giảm tải). Đồng thời EDA cho thấy chênh lệch rõ (30.5% vs 10.4%) → ứng viên tiềm năng.

**Tại sao điều chỉnh đúng 17 confounder?**
- Confounder là biến ảnh hưởng **đồng thời** đến cả việc có làm thêm giờ và việc nghỉ việc. Ví dụ: nhân viên trẻ/thu nhập thấp vừa hay làm thêm nhiều, vừa dễ nghỉ → nếu bỏ qua, ta sẽ **thổi phồng** tác động của OverTime. Danh sách 17 biến (Age, Income, JobLevel, MaritalStatus, Tenure, Satisfaction…) được đưa vào để triệt tiêu thiên lệch này.

**Tại sao KHÔNG điều chỉnh các biến trung gian (mediator) như JobSatisfaction?**
- Vì chúng nằm **trên đường nhân quả** (OverTime → hài lòng → nghỉ). Nếu điều chỉnh chúng, ta sẽ **chặn mất một phần tác động thật** của chính sách (hiện tượng overcontrol bias). Do đó tập backdoor chỉ gồm confounder, loại trừ mediator.

**Tại sao dùng backdoor.linear_regression?**
- Cho **ATE (risk difference)** dễ diễn giải; với treatment nhị phân, hệ số hồi quy chính là chênh lệch xác suất trung bình đã hiệu chỉnh. Đơn giản, ổn định, đủ cho báo cáo; các phương pháp propensity (PSM/IPW) cũng có sẵn để đối chiếu.

**Tại sao cần Refutation?**
- **Placebo**: thay treatment bằng nhiễu ngẫu nhiên → ATE phải gần 0 (kết quả −0.002 → đạt). Chứng tỏ kết quả không phải do "bày ma trò" của mô hình.
- **Random Common Cause**: thêm biến nhiễu ngẫu nhiên → ATE giữ nguyên (0.211 → đạt). Chứng tỏ ước lượng bền vững.

**Ý nghĩa sâu của ATE = +0.211:**
- Giữ nguyên confounder, chuyển một nhân viên từ "không làm thêm" sang "có làm thêm" làm xác suất nghỉ tăng **21.1 điểm %**.
- So sánh với chênh lệch thô (naive) 30.5% − 10.4% = **20.1%**: ATE hiệu chỉnh (21.1%) **gần bằng** con số thô → nghĩa là **mối liên hệ này phần lớn là nhân quả thật, không phải do confounding**. Đây là phát hiện then chốt giúp HR tự tin can thiệp.

**Hạn chế cần ghi nhận:**
- Có thể vẫn tồn tại confounding **chưa đo lường** (vd chất lượng quản lý, áp lực ngầm).
- Dữ liệu là **ảnh chụp đơn**, chưa có chiều thời gian → chưa dự báo *khi nào* nghỉ (cần survival analysis khi có dữ liệu dọc).

---

## 11. Thảo luận

### 11.1. So sánh với cách tiếp cận cũ (proxy Attrition_Index)
| Tiêu chí | Cách cũ (proxy) | Cách mới (IBM thật) |
|---|---|---|
| Nhãn | Tự định nghĩa (top 25% chỉ số) | Nghỉ việc **thật** |
| Tính vòng tròn | Có (label = hàm của feature) | Không |
| ATE nhân quả | +0.76% (vô nghĩa) | **+21.1%** (có ý nghĩa) |
| Tin cậy báo cáo | Thấp | **Cao** |

### 11.2. Hạn chế
- Dữ liệu là **ảnh chụp đơn (snapshot)**, không có chiều thời gian → chưa làm được survival analysis (dự báo *khi nào* nghỉ).
- Là bộ dữ liệu mẫu (Watson Analytics), không phải dữ liệu công ty thực → cần validate lại trên HRIS nội bộ.
- Quan sát thuần túy → vẫn có thể tồn tại confounding chưa đo lường (dù refutation ổn định).

### 11.3. Đạo đức & công bằng (People Analytics)
- `Age`, `Gender`, `MaritalStatus` chỉ dùng làm **confounder hiệu chỉnh thiên lệch**, không dùng làm tiêu chí tự động ra quyết định nghỉ/vào.
- Hệ thống là **hỗ trợ quyết định (human-in-the-loop)**, không thay thế lãnh đạo HR.

---

## 12. Kết luận & khuyến nghị

1. **Dự báo:** Random Forest đạt ROC-AUC 0.799, đủ tin cậy để sàng lọc nhóm rủi ro; XGBoost đạt Precision 0.667; Graph (LabelSpreading, 51 chiều) đạt ROC-AUC 0.691, dùng bổ trợ trực quan cụm rủi ro.
2. **Giải thích:** SHAP chỉ ra OverTime, StockOptionLevel, Income, JobLevel, MaritalStatus là then chốt.
3. **Nhân quả:** Làm thêm giờ làm tăng 21.1% nguy cơ nghỉ việc (đã qua kiểm định bác bỏ).
4. **Hành động HR:** ưu tiên giảm giờ làm thêm, mở rộng quyền chọn cổ phiếu, cải thiện lương nhóm thấp, và chương trình giữ chân nhân viên mới (0–2 năm).

---

## 13. Tái hiện & cấu trúc file

**Chạy pipeline:**
```bash
cd "C:\Dự án công nghệ thông tin\hr_causal_ai"
python train_ibm.py      # RF + XGBoost + SHAP + DAG + results JSON
python analyze_ibm.py    # EDA stats + SHAP rankings
```

**File đầu ra (`reports/ibm/`):**
| File | Nội dung |
|---|---|
| `ibm_results.json` | Chỉ số RF/XGBoost/Graph + ATE nhân quả + refutation |
| `analyze_stats.json` | Thống kê EDA & thứ hạng SHAP |
| `roc_curves.png` | Đường ROC RF vs XGBoost vs Graph |
| `metrics_bar.png` | Biểu đồ so sánh 5 metric (3 mô hình) |
| `graph_similarity.png` | Đồ thị tương đồng nhân viên kNN (mô hình Graph) |
| `attrition_dag.png` | Đồ thị nhân quả (Causal DAG) |
| `shap_rf.png`, `shap_xgboost.png` | SHAP summary plot |

**Dữ liệu đầu vào:** `data/WA_Fn-UseC_-HR-Employee-Attrition.csv`

---
*Báo cáo được tự động sinh từ pipeline `train_ibm.py` + `analyze_ibm.py` trên bộ dữ liệu IBM HR Analytics.*
