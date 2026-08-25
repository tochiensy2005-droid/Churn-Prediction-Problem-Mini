# Churn Prediction Production ML Pipeline (Version v2)

Kho lưu trữ này chứa pipeline học máy cấp độ production để dự đoán customer churn của khách hàng. Pipeline được thiết kế dựa trên mô hình **ảnh chụp nhanh hàng tháng (rolling monthly snapshot)**, tích hợp temporal feature engineering, Platt Scaling probability calibration, xác thực schema, và giám sát systematic drift.

---

## 1. Problem Definition (Định nghĩa bài toán)
Mục tiêu là dự đoán customer churn trên **khung thời gian 30 ngày trong tương lai** ($S \rightarrow S + 30$ ngày) cho mỗi monthly snapshot $S$.
Khách hàng được dán nhãn là **churned** (`churn_next_30d = 1`) nếu thỏa mãn ít nhất một trong ba điều kiện (phiên bản v2):
1. **Rule 1 (Closed)**: Khách đóng tài khoản trong 30 ngày sau snapshot.
2. **Rule 2 (Downgrade to Free + Inactive)**: Khách hàng ở paid tier thực hiện hạ cấp xuống Free tier trong 30 ngày tiếp theo VÀ hoàn toàn inactive (không sử dụng app, không có đơn hàng completed, không có thanh toán success).
3. **Rule 3 (Already Free + Inactive)**: Khách hàng đã ở Free tier tại snapshot date VÀ không có downgrade xuống Free trong 30 ngày tiếp theo VÀ hoàn toàn inactive trong 30 ngày tiếp theo.

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

Chúng tôi trích xuất **379 đặc trưng thời gian (temporal features)** chia thành các nhóm hành vi:
* **Lag Features**: Dịch chuyển (shifts) 1, 2, và 3 tháng cho tất cả các biến hành vi cơ sở.
* **Rolling Features**: Tính tổng (sums), trung bình (means), độ lệch chuẩn (std), tối thiểu (mins), và tối đa (maxs) trên các cửa sổ trượt 1, 3, và 6 tháng.
* **Trend Features**: Chênh lệch tuyệt đối 1 tháng (1-month differences), phần trăm thay đổi (percentage changes), và độ dốc hồi quy tuyến tính 3 tháng (3-month linear regression slopes).
* **Recency Features**: Số ngày trôi qua kể từ các lần tương tác cuối cùng của các sự kiện usage, order, payment, ticket, và downgrade.

---

## 4. Chronological Splitting & Test Lock (Phân tách theo thời gian & Khóa dữ liệu kiểm thử)

Chúng tôi thực hiện quy tắc chia dữ liệu theo thời gian (chronological time splitting) nghiêm ngặt để mô phỏng chính xác môi trường sản xuất thực tế:
* **Train Set**: Snapshot dates từ `2024-09-01` đến `2025-08-01` (rolling window để huấn luyện lại là **12 tháng**).
* **Validation Set**: Snapshot dates từ `2025-09-01` đến `2026-02-01` (được sử dụng nghiêm ngặt để hiệu chuẩn xác suất và tối ưu hóa ngưỡng phân loại).
* **Clean Test Set**: Snapshot dates từ `2026-03-01` đến `2026-06-01` (hoàn toàn được khóa bảo mật trong suốt quá trình phát triển và tinh chỉnh).
* **Excluded Snapshots**: Các snapshot từ `2026-07-01` trở đi bị đánh dấu là chưa hoàn thiện nhãn do bị cắt cụt prediction window và bị loại bỏ khỏi kiểm thử.

---

## 5. Model Evaluation & Benchmarks (Đánh giá & Đối chuẩn mô hình v2)

Hiệu năng mô hình LightGBM v2 được huấn luyện và đánh giá trên tập Clean Test sạch:
- **PR-AUC**: `0.538246`
- **ROC-AUC**: `0.951322`
- **Precision**: `54.3224%`
- **Recall**: `93.3735%`
- **F1-Score**: `68.6854%` (tại ngưỡng tối ưu `0.24` được hiệu chuẩn bằng Platt Scaling).

---

## 6. CLI Commands & Notebook Orchestration (Thứ tự chạy pipeline)

Pipeline có thể khởi chạy và điều phối qua 3 script Python hoặc trực tiếp trong Jupyter Notebook [`Modeling.ipynb`](Modeling.ipynb):

### A. Chạy trích xuất đặc trưng và dán nhãn Churn v2
Trích xuất lags, rolling, trend, recency từ các bảng Silver gốc và gán nhãn Churn v2:
```bash
python generate_features_v2.py
```
Kết quả ghi nhận tại `output/churn_temporal_dataset_v2.parquet` và báo cáo phân bổ `output/churn_rule_v2_audit.csv`.

### B. Huấn luyện và hiệu chuẩn mô hình
Chia dữ liệu theo thời gian, huấn luyện LightGBM và Platt Scaling calibrator, tìm ngưỡng tối ưu:
```bash
python train_lightgbm_v2.py
```
Model bundle được đóng gói và lưu tại `artifacts/temporal_churn_model_v2.joblib`.

### C. Đánh giá và xác thực mô hình
Đánh giá chi tiết hiệu năng mô hình v2 và chạy test kiểm định suy diễn:
```bash
python compare_models.py
```
Metrics chi tiết lưu tại `output/churn_rule_v2_metrics.csv`.
