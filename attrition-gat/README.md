# Attrition-GAT — pipeline tai lap duoc (Giai doan 1)

## Quy tac
- Test chia truoc, fit chi tren train. Khong SMOTE. Metric chinh PR-AUC.
- Baseline duy nhat: RF (`class_weight`) va XGBoost (`scale_pos_weight`).

## Chay lai
```bash
pip install -r attrition-gat/requirements.txt
python attrition-gat/tests/test_no_leakage.py
python attrition-gat/run_smoke.py   # 1 fold, 3 trials/model
```
Full-run (sau khi xac nhan): 25 folds x 50 trials — xem `results/logs/smoke_time.json` de uoc luong.

## Cau truc
`src/{preprocess,cv,metrics,tune_baseline}.py`, `configs/{base,rf,xgb,gat}.yaml`,
`results/{folds,logs,tables}/`. GAT + Gower (G2) tu Giai doan 3.
