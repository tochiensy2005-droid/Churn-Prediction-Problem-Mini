# Churn Prediction Final Model (Version v2)

## Problem (Bài toán)
Dự đoán customer churn trong 30 ngày tiếp theo ($S \rightarrow S + 30$ ngày) sử dụng lịch sử hành vi của khách hàng theo thời gian (historical temporal behavior). Khách hàng được định nghĩa là rời bỏ dịch vụ (`churn_next_30d = 1`) nếu thỏa mãn ít nhất một trong ba điều kiện (phiên bản v2):
1. **Rule 1 (Closed)**: Khách đóng tài khoản trong 30 ngày sau snapshot.
2. **Rule 2 (Downgrade to Free + Inactive)**: Khách hàng ở paid tier thực hiện hạ cấp xuống Free tier trong 30 ngày tiếp theo VÀ hoàn toàn inactive (không sử dụng app, không có đơn hàng completed, không có thanh toán success).
3. **Rule 3 (Already Free + Inactive)**: Khách hàng đã ở Free tier tại snapshot date VÀ không có downgrade xuống Free trong 30 ngày tiếp theo VÀ hoàn toàn inactive trong 30 ngày tiếp theo.

## Dataset (Tập dữ liệu)
- **Cấu trúc**: `customer_id` × `snapshot_date` (ảnh chụp nhanh trượt hàng tháng vào ngày 1 hàng tháng).
- **Phạm vi thời gian**: 01/08/2023 đến 01/08/2026 (tổng cộng 37 snapshot).
- **Tổng số dòng**: 185,160 dòng.
- **Tỷ lệ Churn toàn tập**: `23.516418%` (43,543 churn events).
- **Tỷ lệ Churn tập Test**: `9.865293%` (3,486 churn events trên tổng số 35,336 customer-snapshots).

## Temporal Features (Đặc trưng thời gian)
- **Lag Features**: dịch chuyển (shift) 1, 2, và 3 tháng cho tất cả các biến hành vi cơ sở.
- **Rolling Features**: các thống kê tổng (sum), trung bình (mean), độ lệch chuẩn (std), tối thiểu (min), và tối đa (max) trên các cửa sổ trượt 1, 3, và 6 tháng.
- **Trend Features**: chênh lệch tuyệt đối 1 tháng (1-month difference), phần trăm thay đổi (percentage change), và độ dốc hồi quy tuyến tính 3 tháng (3-month linear regression slope).
- **Recency Features**: số ngày trôi qua kể từ lần tương tác gần nhất của các sự kiện sử dụng dịch vụ (usage), đơn hàng (order), thanh toán (payment), hỗ trợ (support ticket), và hạ cấp (downgrade).

## Final Model (Mô hình cuối cùng)
- **Thuật toán (Algorithm)**: `LightGBM Classifier`
- **Số lượng đặc trưng (Feature Count)**: `All 379 features` (bao gồm toàn bộ đặc trưng thời gian lags, rolling, trend và recency).
- **Cửa sổ huấn luyện trượt (Rolling Training Window)**: `12 Months` huấn luyện + `3 Months` hiệu chuẩn + `3 Months` Test.
- **Hiệu chuẩn (Calibration)**: `Platt Scaling` (sử dụng Logistic Regression khớp trên xác suất của tập Validation).
- **Ngưỡng (Threshold)**: `0.24` (được hiệu chuẩn trên tập Validation để tối ưu hóa F1 score).

## Final Performance (Hiệu năng cuối cùng)

### A. Clean Test Split (Entire Cohort: 35,336 samples, 3,486 churns)
- **PR-AUC**: `0.538246`
- **ROC-AUC**: `0.951322`
- **Precision**: `54.3224%`
- **Recall**: `93.3735%`
- **F1**: `68.6854%`
- **Confusion Matrix**:
  ```
  [[29113  2737]
   [  231  3255]]
  ```

### B. Validation Split (45,586 samples, 10,844 churns)
- **PR-AUC**: `0.542051`
- **ROC-AUC**: `0.857541`
- **Precision**: `50.0308%`
- **Recall**: `89.8469%`
- **F1**: `64.2720%`
- **Brier Score**: `0.125086`

## Saved Artifact (Tập tin mô hình đã lưu)
`artifacts/temporal_churn_model_v2.joblib`

## How To Load (Cách tải mô hình)
```python
import joblib
import pandas as pd

# 1. Tải gói mô hình (model bundle)
bundle = joblib.load("artifacts/temporal_churn_model_v2.joblib")
model = bundle["model"]                  # LightGBM Classifier
calibrator = bundle["calibrator"]        # Platt Calibrator
selected_features = bundle["selected_features"]  # Danh sách All 379 tên đặc trưng
imputer = bundle["imputer"]              # SimpleImputer
threshold = bundle["threshold"]          # Ngưỡng F1 hiệu chuẩn (0.24)

# 2. Chuẩn bị đặc trưng (ví dụ với df chứa dữ liệu thô)
df = pd.read_parquet("output/churn_temporal_dataset_v2.parquet")
X = df[selected_features]

# 3. Impute, Predict, Calibrate, và Threshold
X_imp = imputer.transform(X)
raw_probs = model.predict_proba(X_imp)[:, 1]
calibrated_probs = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
predictions = (calibrated_probs >= threshold).astype(int)
```
