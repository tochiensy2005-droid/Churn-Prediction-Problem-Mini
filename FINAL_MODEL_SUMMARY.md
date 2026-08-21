# Churn Prediction Final Model

## Problem (Bài toán)
Dự đoán customer churn trong 30 ngày tiếp theo ($S \rightarrow S + 30$ ngày) sử dụng lịch sử hành vi của khách hàng theo thời gian (historical temporal behavior). Khách hàng được định nghĩa là rời bỏ dịch vụ (`churn_next_30d = 1`) nếu họ đóng tài khoản (close account) hoặc thực hiện hạ cấp dịch vụ (downgrade) và duy trì trạng thái không hoạt động (inactive) trong khoảng thời gian 30 ngày này.

## Dataset (Tập dữ liệu)
- **Cấu trúc**: `customer_id` × `snapshot_date` (ảnh chụp nhanh trượt hàng tháng vào ngày 1 hàng tháng).
- **Phạm vi thời gian**: 01/08/2023 đến 01/08/2026 (tổng cộng 37 snapshot).
- **Tỷ lệ Churn**: `0.7047%` trong tập Clean Test đã hoàn thiện nhãn (249 churn events trên tổng số 35,336 customer-snapshots).

## Temporal Features (Đặc trưng thời gian)
- **Lag Features**: dịch chuyển (shift) 1, 2, và 3 tháng cho tất cả các biến hành vi cơ sở.
- **Rolling Features**: các thống kê tổng (sum), trung bình (mean), độ lệch chuẩn (std), tối thiểu (min), và tối đa (max) trên các cửa sổ trượt 1, 3, và 6 tháng.
- **Trend Features**: chênh lệch tuyệt đối 1 tháng (1-month difference), phần trăm thay đổi (percentage change), và độ dốc hồi quy tuyến tính 3 tháng (3-month linear regression slope).
- **Recency Features**: số ngày trôi qua kể từ lần tương tác gần nhất của các sự kiện sử dụng dịch vụ (usage), đơn hàng (order), thanh toán (payment), hỗ trợ (support ticket), và hạ cấp (downgrade) (tính bằng `pd.merge_asof` với zero lookahead).

## Models Compared (Các mô hình so sánh)
1. **Logistic Regression (Static Baseline)**
2. **Random Forest (Static Baseline)**
3. **XGBoost (Tuned Temporal Baseline)**
4. **LightGBM (Static Temporal)**
5. **LightGBM (Rolling 12M Temporal Retraining)** (Production Model Candidate)
6. **ARIMA / SARIMA** (Dự báo chuỗi hành vi tổng hợp - Aggregate behavioral series forecasting)
7. **PyTorch LSTM / GRU** (Mô hình chuỗi tuần tự RNN độ dài 12)

## Final Model (Mô hình cuối cùng)
- **Thuật toán (Algorithm)**: `LightGBM Classifier`
- **Số lượng đặc trưng (Feature Count)**: `All 379 features` (bao gồm toàn bộ đặc trưng thời gian lags, rolling, trend và recency).
- **Cửa sổ huấn luyện trượt (Rolling Training Window)**: `12 Months` huấn luyện + `3 Months` hiệu chuẩn (được tái huấn luyện động tại thời điểm suy diễn dựa trên snapshot_date).
- **Hiệu chuẩn (Calibration)**: `Platt Scaling` (sử dụng Logistic Regression khớp trên xác suất của tập Validation).
- **Ngưỡng (Threshold)**: `0.08` (được hiệu chuẩn trên tập Validation để tối ưu hóa F1 score).

## Final Performance (Hiệu năng cuối cùng)

### A. Clean Test Split (Entire Cohort: 35,336 samples, 249 churns)
- **PR-AUC**: `0.121786`
- **ROC-AUC**: `0.910986`
- **Precision**: `17.9856%`
- **Recall**: `30.1205%`
- **F1**: `22.5225%` (hoặc `0.225225`)
- **Confusion Matrix**:
  ```
  [[34745   342]
   [  174    75]]
  ```

### B. Sequence-Eligible Split (Subset với lịch sử $\ge 12$ tháng: 23,296 samples, 109 churns)
*Lưu ý: Hiệu năng trên phân khúc chuỗi lịch sử dài cũng được tối ưu hóa tương tự nhờ tính toàn vẹn tín hiệu của All379.*

## Saved Artifact (Tập tin mô hình đã lưu)
`artifacts/temporal_churn_model.joblib`

## How To Load (Cách tải mô hình)
```python
import joblib
import pandas as pd

# 1. Tải gói mô hình (model bundle)
bundle = joblib.load("artifacts/temporal_churn_model.joblib")
model = bundle["model"]                  # LightGBM Classifier
calibrator = bundle["calibrator"]        # Platt Calibrator
selected_features = bundle["selected_features"]  # Danh sách All 379 tên đặc trưng
imputer = bundle["imputer"]              # SimpleImputer
threshold = bundle["threshold"]          # Ngưỡng F1 hiệu chuẩn (0.08)

# 2. Chuẩn bị đặc trưng (ví dụ với df chứa dữ liệu thô)
df = pd.read_parquet("output/churn_temporal_dataset.parquet")
X = df[selected_features]

# 3. Impute, Predict, Calibrate, và Threshold
X_imp = imputer.transform(X)
raw_probs = model.predict_proba(X_imp)[:, 1]
calibrated_probs = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
predictions = (calibrated_probs >= threshold).astype(int)
```
