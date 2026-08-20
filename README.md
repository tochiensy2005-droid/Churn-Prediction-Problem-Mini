# Churn Prediction Production ML Pipeline

Kho lưu trữ này chứa pipeline học máy cấp độ production để dự đoán customer churn của khách hàng. Pipeline được thiết kế dựa trên mô hình **ảnh chụp nhanh hàng tháng (rolling monthly snapshot)**, tích hợp temporal feature engineering, Platt/Isotonic probability calibration, xác thực schema, và giám sát systematic drift.

---

## 1. Problem Definition (Định nghĩa bài toán)
Mục tiêu là dự đoán customer churn trên **khung thời gian 30 ngày trong tương lai** ($S \rightarrow S + 30$ ngày) cho mỗi monthly snapshot $S$.
Khách hàng được dán nhãn là **churned** (`churn_next_30d = 1`) nếu trong cửa sổ dự báo 30 ngày:
1. Họ đóng tài khoản (closed_date nằm trong cửa sổ).
2. Họ thực hiện thay đổi gói cước hạ cấp gói cước (**downgrade**) và duy trì trạng thái hoàn toàn không hoạt động (không có usage activity, payments hoặc orders) trong phần còn lại của cửa sổ.

Ngược lại, khách hàng được dán nhãn là **active** (`churn_next_30d = 0`).

---

## 2. Data Architecture (Kiến trúc dữ liệu)

Các bảng event logs thô được làm sạch và lưu trữ dưới dạng Silver tables phân vùng trong workspace:
* `churn_customers`: Thông tin nhân khẩu học và trạng thái vòng đời khách hàng.
* `churn_subscriptions`: Gói cước (plan tier), các nâng/hạ cấp gói cước (downgrades), và chu kỳ thanh toán.
* `churn_product_usage`: Ngày hoạt động (active days) và sản lượng sử dụng hàng ngày (daily usage volume).
* `churn_orders` & `churn_payments`: Lịch sử mua hàng, khối lượng đơn hàng, tỷ lệ thanh toán thành công/thất bại.
* `churn_support_tickets`: Nhật ký hỗ trợ kỹ thuật và điểm CSAT.
* `churn_marketing_interactions`: Lượt nhấp chuột (clicks) và lượt tiếp cận (impressions) của các chiến dịch tiếp thị.

---

## 3. Temporal Feature Engineering (Xây dựng đặc trưng thời gian)

Để ghi nhận các động thái hành vi của khách hàng mà không gặp lỗi lookahead bias, chúng tôi trích xuất **260 đặc trưng thời gian (temporal features)** chia thành các nhóm:
* **Lag Features**: Dịch chuyển (shifts) 1, 2, và 3 tháng cho tất cả các biến hành vi cơ sở.
* **Rolling Features**: Tính tổng (sums), trung bình (means), độ lệch chuẩn (std), tối thiểu (mins), và tối đa (maxs) trên các cửa sổ trượt 1, 3, và 6 tháng.
* **Trend Features**: Chênh lệch tuyệt đối 1 tháng (1-month differences), phần trăm thay đổi (percentage changes), và độ dốc hồi quy tuyến tính 3 tháng (3-month linear regression slopes).
* **Recency Features**: Số ngày trôi qua kể từ các lần tương tác cuối cùng của các sự kiện usage, order, payment, ticket, và downgrade.

---

## 4. Chronological Splitting & Test Lock (Phân tách theo thời gian & Khóa dữ liệu kiểm thử)

Chúng tôi thực hiện quy tắc chia dữ liệu theo thời gian (chronological time splitting) nghiêm ngặt để mô phỏng chính xác môi trường sản xuất thực tế:
* **Train Set**: Snapshot dates $\le$ `2025-08-01` (rolling window để huấn luyện lại có thể tùy chỉnh, mặc định là **12 tháng** trước validation).
* **Validation Set**: Snapshot dates từ `2025-09-01` đến `2026-02-01` (được sử dụng nghiêm ngặt để hiệu chuẩn xác suất và tối ưu hóa ngưỡng phân loại).
* **Clean Test Set**: Snapshot dates từ `2026-03-01` đến `2026-06-01` (hoàn toàn được khóa bảo mật trong suốt quá trình phát triển và tinh chỉnh).
* **Excluded Snapshots**: Các snapshot từ `2026-07-01` trở đi bị đánh dấu là chưa hoàn thiện nhãn (`label_complete = False`) do bị cắt cụt prediction window ($S + 30$ ngày vượt quá giới hạn dữ liệu hiện có trong database) và bị loại bỏ khỏi kiểm thử.

---

## 5. Model Evaluation & Benchmarks (Đánh giá & Đối chuẩn mô hình)

Chúng tôi đã đánh giá hiệu năng của nhiều nhóm mô hình trên các snapshot của tập Clean Test:
1. **Logistic Regression (Static Baseline)**: PR-AUC = `0.0436`, ROC-AUC = `0.7812`
2. **LightGBM (Static)**: PR-AUC = `0.1419`, ROC-AUC = `0.9054`
3. **LightGBM (Rolling 12M Retraining)**: PR-AUC = **`0.1521`**, ROC-AUC = **`0.9231`**
4. **PyTorch LSTM (Sequence Baseline)**: PR-AUC = `0.1047`, ROC-AUC = `0.9042`
5. **PyTorch GRU (Sequence Baseline)**: PR-AUC = `0.1061`, ROC-AUC = `0.9228`

**LightGBM với cơ chế huấn luyện lại trượt 12 tháng (Rolling 12-Month Retraining)** được chọn làm **Production Model Candidate** cuối cùng vì nó vượt trội hơn hẳn các baselines tĩnh và các mô hình deep learning tuần tự.

---

## 6. Drift & Retraining Strategy (Chiến lược giám sát trôi lệch & Huấn luyện lại)

* **Feature Drift (PSI)**: Chúng tôi tính toán Population Stability Index (PSI) cho Top 100 features.
  * $\text{PSI} < 0.1$: Ít trôi lệch (low drift).
  * $0.1 \le \text{PSI} \le 0.25$: Trôi lệch vừa (moderate drift).
  * $\text{PSI} > 0.25$: Trôi lệch mạnh (strong drift - tự động kích hoạt cảnh báo retraining).
* **Prediction Drift**: Theo dõi các chỉ số phân phối xác suất dự báo (mean, median, p90, p95, p99) và tỷ lệ dự đoán rời bỏ (predicted churn rate).
* **Probability Calibration**: Hiệu chuẩn Platt Scaling giúp giảm Brier Score và Log Loss trên validation hơn **95%**, căn chỉnh các xác suất thô dự đoán khớp lại với tỷ lệ thực tế (~0.70%).
* **Retraining Recommendation Policy**: Khuyến nghị tái huấn luyện (`recommend_retrain = True`) nếu chỉ số PSI trung bình vượt quá `0.25` hoặc chỉ số phân loại F1 score rơi xuống dưới `0.15`.

---

## 7. Production CLI Commands (Các lệnh CLI trong hệ thống)

Hãy đảm bảo môi trường đã được cài đặt và cấu hình qua file `.env` trước khi khởi chạy các lệnh.

### A. Chạy Pipeline Huấn luyện (Run Training Pipeline)
Khớp LightGBM trên cửa sổ trượt 12 tháng, lựa chọn Top 100 đặc trưng thời gian, hiệu chuẩn xác suất bằng Platt Scaling trên tập validation, và lưu trữ gói mô hình đã đóng gói vào `artifacts/temporal_churn_model.joblib`.
```bash
python -m src.training.train_lightgbm
```

### B. Chạy Batch Inference (Run Batch Inference)
Tính toán các đặc trưng cho các sự kiện lịch sử xảy ra trước snapshot date, chạy xác thực schema (fail-fast checks), và tính toán ra các xác suất rời bỏ và nhãn dự báo đã hiệu chuẩn.
```bash
# Lệnh CLI tổng quát
python -m src.inference.predict_churn --snapshot-date 2026-06-01

# Lệnh Wrapper tắt từ gốc repo
python predict_churn.py --snapshot-date 2026-06-01
```
Đường dẫn lưu kết quả: `output/predictions/YYYY-MM-DD/churn_predictions.parquet`

### C. Chạy Giám sát & Drift Checks (Run Monitoring & Drift Checks)
Tính toán PSI cho các đặc trưng, tổng hợp thống kê prediction drift, target drift, và tính toán hiệu năng thực tế F1 (nếu dữ liệu thực tế 30 ngày sau snapshot đã sẵn sàng), đưa ra các đề xuất huấn luyện lại mô hình.
```bash
python -m src.monitoring.run_monitoring --snapshot-date 2026-06-01
```
Đầu ra bao gồm:
* `output/modeling/drift/data_drift_YYYY-MM-DD.csv` (Chỉ số PSI cho Top 100 features)
* `output/modeling/drift/prediction_drift.csv` (Nhật ký phân phối xác suất dự báo)
* `output/modeling/drift/performance_drift.csv` (Nhật ký theo dõi F1 & ROC-AUC thực tế)
