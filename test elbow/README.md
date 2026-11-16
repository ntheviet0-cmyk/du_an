# GraphSAGE — elbow và validation experiments

**So sánh baseline / grid search / elbow:** xem [COMPARISON.md](COMPARISON.md) và `compare_methods.py`.

Thư mục thử nghiệm độc lập. Đọc dataset IBM từ `../data/`; không thay model, preprocessing, report hay giao diện production.

## Kiến trúc

```text
config.json → run_experiments.py
                    ├── data_pipeline.py: train-only preprocessing + cosine-kNN
                    ├── model.py: mean GraphSAGE 51 → h1 → h2 → logit
                    ├── select_model.py: fold summaries, one-SE, knee heuristic
                    └── outputs/full/ hoặc outputs/smoke/
                                      ↓
                              draw_results.py
```

Số chiều input được xác định từ encoder của mỗi fold, không hard-code 51. Cấu hình baseline giữ `k=15`, `64→32`, dropout `.3`, Adam LR `.01`, WD `5e-5`.

## Phụ thuộc

Dùng môi trường project có `torch`, `torch-geometric`, `scikit-learn` (hỗ trợ `sparse_output=False`), `numpy`, `pandas`, `matplotlib`, `joblib`. Chạy CPU, số thread do config quyết định. Không tải dataset hoặc gọi API.

## Chạy từ thư mục gốc dự án

```powershell
# Kiểm tra protocol và parity của inductive batching
python -m unittest discover -s "test elbow" -p "test_pipeline.py" -v

# Smoke test: 2 folds, 3 epochs; KHÔNG phải chứng cứ chọn hyperparameter
python "test elbow/run_experiments.py" --stage baseline --smoke
python "test elbow/draw_results.py" --smoke

# Các giai đoạn chính; mặc định tự resume
python "test elbow/run_experiments.py" --stage baseline
python "test elbow/run_experiments.py" --stage k
python "test elbow/run_experiments.py" --stage hidden
python "test elbow/run_experiments.py" --stage dropout
python "test elbow/run_experiments.py" --stage regularization
python "test elbow/run_experiments.py" --stage finalists
python "test elbow/draw_results.py"

# Hoặc toàn bộ chuỗi
python "test elbow/run_experiments.py" --stage all

# Giới hạn số job mới rồi dừng; chạy lại cùng lệnh để tiếp tục
python "test elbow/run_experiments.py" --stage k --limit 5
```

`--limit` đếm fold/seed jobs, không đếm cấu hình. Đơn vị resume là một job hoàn tất; nếu ngắt giữa job, chỉ job đó cần chạy lại. Không chạy hai runner đồng thời vào cùng output directory.

## Thiết kế đánh giá

- Giữ 294 mẫu holdout lịch sử ngoài tuning; development gồm 1.176 mẫu còn lại.
- Stratified 5-fold trên development, cùng fold/seed cho mọi cấu hình.
- Fit scaler và encoder **chỉ trên training fold**.
- Graph train: cosine kNN, connectivity không trọng số, đối xứng hóa, không self-edge.
- Mỗi validation node gắn độc lập vào graph train qua top-k cạnh hai chiều. Không có validation–validation edges; nhãn validation không dùng để tạo graph.
- Disjoint batching gom nhiều bản graph độc lập để tăng tốc; đã có test đối chiếu cách attach một node của app.
- Weighted BCE: `pos_weight` tính trên nhãn train fold. ReLU sau mỗi SAGEConv, dropout chỉ sau layer đầu.
- Early stopping theo validation AP: tối đa 200 epochs, patience 30 epochs, đánh giá mỗi 5 epochs. Khôi phục trọng số epoch tốt nhất.
- **Validation dùng cho cả early stopping và lựa chọn cấu hình.** CV score dùng để lựa chọn, có thiên lệch lạc quan; không phải ước lượng hiệu năng độc lập. Muốn claim hiệu năng cuối cần nested CV hoặc dữ liệu ngoài chưa từng xem.
- Holdout cũ đã được khảo sát nhiều lần; không gọi nó là test mới, cũng không chọn threshold trên 37 lỗi đã biết.

## Giai đoạn tuning

| Stage | Không gian | Jobs mặc định |
|---|---|---:|
| baseline | Cấu hình đang dùng | 5 |
| k | 3, 5, 8, 10, 15, 20, 25, 30, 40 | 45 |
| hidden | 16→8, 32→16, 64→32, 128→64, 256→128 | 25 |
| dropout | 0, .1, .2, .3, .4, .5, .6 | 35 |
| regularization | 4 LR × 5 WD | 100 |
| finalists | Top-2 của mỗi axis; thêm baseline; 3 seeds | Tối đa 255 |

Screening thay từng axis quanh baseline. Finalists kết hợp top-2 k × top-2 hidden × top-2 dropout × top-2 LR/WD, tối đa 16 cấu hình cộng baseline. Phải hoàn tất cả 4 stage screening trước finalists. Đây là tìm kiếm có giới hạn, không chứng minh tối ưu toàn cục.

### Chọn và diễn giải elbow

- AP (`average_precision_score`) là metric chính; ROC-AUC là metric phụ. Không gọi AP là trapezoidal PR-AUC.
- Elbow cho `k` và số tham số hidden: khoảng cách lớn nhất phía trên đường nối hai đầu sau chuẩn hóa.
- Đường phẳng, tuyến tính, không đơn điệu hoặc không có knee nội bộ rõ ràng → báo **không có elbow rõ**, không ép chọn.
- Dropout dùng validation curve, LR/WD dùng heatmap; không áp elbow máy móc.
- One-standard-error: lấy mean AP tốt nhất trừ SE của cấu hình đó, chọn ứng viên trong vùng này với ít cạnh nhất → ít tham số nhất → thời gian thấp nhất.
- Trung bình seed **trong từng fold trước**, sau đó tính SD/SE giữa folds. CV training sets chồng lấp, nên SE là mô tả chứ không phải chứng minh ý nghĩa thống kê.
- Chưa tự tune threshold, calibration hoặc thay mô hình production. Checkpoint của từng fold không phải model final refit toàn development.

## Output

```text
outputs/full/                 # smoke dùng outputs/smoke riêng
├── experiment_config.json    # config, code/data hash, phiên bản thư viện
├── candidates_<stage>.json
├── splits/split_manifest.json
├── checkpoints/              # model state + train-fitted preprocessing mỗi job
├── predictions/              # source_index, EmployeeNumber, nhãn thật, probability
├── history/                  # train loss / validation AP từng lần đánh giá
├── tables/
│   ├── runs.csv
│   ├── summary.csv
│   └── finalist_comparison.csv
├── figures/
│   ├── baseline_learning_curve.png
│   ├── elbow_k.png
│   ├── elbow_hidden.png
│   ├── validation_dropout.png
│   └── lr_weight_decay_heatmap.png
├── selected_candidate.json   # chỉ khi finalists hoàn tất
└── selection_report.md
```

Chỉ vẽ/chọn từ candidate hoàn tất đủ fold/seed; stage chưa chạy sẽ không có hình tương ứng. Output bị gitignore để tránh commit checkpoint dung lượng lớn. Fingerprint ngăn resume với code/config/dataset hoặc môi trường đã đổi; muốn thí nghiệm mới cần lưu trữ/chuyển tên output cũ trước.
