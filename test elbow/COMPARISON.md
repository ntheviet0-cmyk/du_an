# So sánh phương pháp chọn siêu tham số

## 1. Đối chiếu cấu hình hiện tại với cấu hình đã chọn

Sau khi `run_experiments.py --stage all` hoàn tất, chạy từ thư mục gốc:

```powershell
py -3.13 "test elbow/compare_methods.py" --mode existing
```

Môi trường phải có torch, torch-geometric, sklearn, pandas, numpy, matplotlib, joblib. Có thể dùng `py -3.12` nếu môi trường đó đã cài đủ. Snapshot existing chỉ đọc CSV nên không phụ thuộc fingerprint của lần training.

Kết quả ở `outputs/existing_comparison/`:

- `model_comparison.csv`: baseline và candidate đã chọn, cùng fold/seed của finalists.
- `paired_ap.csv`: chênh lệch AP theo fold sau khi trung bình seed.
- `model_comparison.png`, `COMPARISON_REPORT.md`.

Nếu mới có baseline, lệnh báo chưa đủ dữ liệu thay vì tự chọn cấu hình từ smoke. Đây là so sánh mô tả trên development validation đã dùng để chọn cấu hình; không phải chứng minh độc lập phương pháp nào tốt hơn.

## 2. Nested CV so sánh hai bộ chọn

```powershell
# Chạy kiểm tra nhỏ trước: 2 outer × 2 inner folds, 2 candidates, 2 epochs
py -3.13 "test elbow/compare_methods.py" --smoke

# Chạy thí nghiệm đầy đủ, không cần chờ bộ elbow cũ hoàn tất
py -3.13 "test elbow/compare_methods.py" --mode nested

# Giới hạn tối đa 5 job train mới; chạy lại cùng lệnh để resume
py -3.13 "test elbow/compare_methods.py" --mode nested --limit 5

# Chỉ dựng lại bảng/hình từ outer results đã lưu
py -3.13 "test elbow/compare_methods.py" --report-only

# Các tests, gồm kiểm tra leakage và batch inference
py -3.13 -m unittest discover -s "test elbow" -p "test_*.py" -v
```

`comparison_config.json` dùng 5 outer folds, 3 inner folds, 1 search seed, 3 evaluation seeds. Pool gồm 72 cấu hình (`3 k × 3 hidden × 2 dropout × 2 LR × 2 WD`) và baseline nếu chưa nằm trong pool. Tối đa 1.080 inner training jobs và 45 outer refits. Thực tế ít refit hơn nếu các phương pháp chọn cùng cấu hình. Cần chốt config trước và dự trù thời gian; smoke không phải kết quả tuning.

### Ba nhánh so sánh

1. **baseline:** cấu hình hiện tại `k15, 64→32, d.3, lr.01, wd5e-5`; train lại theo protocol chung.
2. **grid_ap:** chọn mean inner-CV AP cao nhất.
3. **elbow_one_se:** xác định vùng one-SE của cấu hình AP cao nhất; kiểm tra knee trên max-AP profiles theo k và số tham số; ưu tiên cấu hình ít cạnh, ít tham số trong vùng đó.

Nếu đường profile phẳng/không đơn điệu, không ép có elbow. Nếu knee caps loại hết ứng viên one-SE, bỏ caps và dùng one-SE toàn cục. Tie-break theo candidate ID để lựa chọn không phụ thuộc nhiễu thời gian chạy.

**Định nghĩa so sánh:** hai bộ chọn dùng chính xác cùng pool, cùng inner runs và budget. Đây là grid search **chuẩn hóa theo AP** so với **knee-assisted one-SE**, không phải tái hiện workflow legacy chọn accuracy/top-3 rồi xem test. Nó cũng không đánh giá tốc độ tìm kiếm của quy trình sequential one-axis screening cũ. Nhờ dùng chung pool, chất lượng khác nhau được quy cho quy tắc lựa chọn, không bị lẫn với không gian tìm kiếm khác nhau.

### Chống leakage

```text
Development 1.176 hồ sơ (holdout lịch sử 294 hồ sơ bị loại)
  └── Outer fold
        ├── Outer train
        │     └── Inner CV: fit preprocessor/graph chỉ trên inner train
        │           → chọn cấu hình, median best epoch, F1 threshold từ inner OOF
        └── Outer validation
              → chỉ chấm điểm sau khi train lại outer train đúng số epoch đã chốt
```

- Không early-stop trên outer validation.
- Không chọn threshold bằng outer labels. Threshold scan `.05–.95`, bước `.01`, tối đa F1; nếu bằng nhau chọn threshold thấp hơn.
- Threshold dùng inner OOF probability sau khi trung bình search seeds; cùng một threshold cho evaluation seeds của cấu hình đó.
- Outer train chỉ nhận nhãn của outer train. Scaler/encoder fit trên outer train; graph không có validation–validation edges.
- Chỉ báo cáo fold khi đủ cả 3 phương pháp và tất cả evaluation seeds. Không tuyên bố kết quả cuối từ partial folds.
- Đã xem xét dataset trong các phân tích trước: nested CV kiểm soát vòng chọn tự động mới, không thay cho external validation.

### Đầu ra

```text
outputs/comparison/                  # smoke dùng comparison_smoke riêng
├── experiment_config.json           # fingerprint code/config/dataset/environment
├── split_manifest.json              # source indices của cả outer và inner splits
├── jobs/
│   └── <job_id>/
│       ├── model.pt
│       ├── preprocessor.joblib
│       ├── predictions.csv
│       └── result.json               # completion marker, ghi cuối cùng
├── selections/outer_<n>.json         # cấu hình/epoch/threshold, ghi trước outer evaluation
├── tables/
│   ├── inner_runs_o<n>.csv
│   ├── inner_summary_o<n>.csv
│   ├── selected_configs.csv
│   ├── outer_fold_results.csv        # AP/AUC/F1/P/R/accuracy, FP/FN/TN/TP, thời gian
│   ├── fold_seed_averaged.csv
│   ├── method_comparison.csv         # mean/SD sau khi trung bình seed trong fold
│   └── paired_ap_differences.csv
├── figures/
│   ├── paired_ap_comparison.png
│   └── performance_vs_complexity.png
└── COMPARISON_REPORT.md
```

Resume bỏ qua job có đủ artifact và completion marker; ngắt giữa job chỉ cần chạy lại job đó. Không chạy hai process ghi chung một output. Không sửa code/config/environment giữa một thí nghiệm; fingerprint sẽ yêu cầu lưu trữ output cũ nếu thay đổi.

### Đọc kết quả

- Metric chính: AP. AP không phải accuracy.
- AP/ROC-AUC theo cặp outer fold là cơ sở so sánh chính; FP/FN phụ thuộc threshold được chọn trong inner CV.
- Error bar là SD giữa outer folds sau khi trung bình seed, không phải các seed được coi như người quan sát độc lập.
- Search time: grid và elbow đều tính cùng ngân sách toàn pool, dù các job thực tế được cache/chia sẻ. Baseline tính các inner jobs riêng của baseline để chọn epoch/threshold. Refit train/inference time ghi riêng.
- AP gần nhau mà model nhẹ hơn là đánh đổi có thể hữu ích, không tự động chứng minh tương đương thống kê. Không kết luận superiority chỉ vì một con số lớn hơn.

Không thay production model, không ghi đè `outputs/full`, `outputs/smoke` hoặc báo cáo cũ.
