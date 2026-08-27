# Customer Churn Prediction ML Pipeline (Version v2)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3%2B-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red.svg)](https://xgboost.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)]()

Kho lưu trữ này chứa pipeline học máy dự đoán tỷ lệ khách hàng rời bỏ dịch vụ (**Customer Churn Prediction**) cấp độ production. Hệ thống được xây dựng trên kiến trúc **ảnh chụp nhanh hàng tháng (Rolling Monthly Snapshots)**, tích hợp kỹ thuật trích xuất **379 đặc trưng thời gian (Temporal Feature Engineering)**, cơ chế **chống rò rỉ dữ liệu (Anti-Data Leakage Protocol)**, mô hình **Stacking Ensemble kết hợp**, và **hiệu chuẩn xác suất Platt Scaling**.

---

## 1. Problem Definition (Định nghĩa bài toán)

Mục tiêu là dự đoán khả năng khách hàng rời bỏ dịch vụ trong **khung thời gian 30 ngày tiếp theo** ($S \rightarrow S + 30$ ngày) tính từ mốc quan sát snapshot $S$.

Khách hàng được dán nhãn là **churned** (`churn_next_30d = 1`) nếu thỏa mãn ít nhất một trong 3 điều kiện kinh doanh thực tế (**Quy tắc Churn v2**):
1. **Rule 1 (Closed)**: Khách hàng thực hiện đóng tài khoản hoàn toàn trong vòng 30 ngày sau snapshot.
2. **Rule 2 (Downgrade to Free + Inactive)**: Khách hàng đang ở gói trả phí (Plus/Pro) hạ cấp xuống gói Free trong 30 ngày tiếp theo VÀ hoàn toàn không có hoạt động (không mở app, không có đơn hàng thành công, không có thanh toán).
3. **Rule 3 (Already Free + Inactive)**: Khách hàng đã ở gói Free tại thời điểm snapshot VÀ hoàn toàn không có hoạt động trong 30 ngày tiếp theo.

Ngược lại, khách hàng được dán nhãn là **active** (`churn_next_30d = 0`).

---

## 2. Data Architecture (Kiến trúc dữ liệu)

Dữ liệu nguồn được thu thập và tổ chức thành các bảng Silver phân vùng theo thời gian:
* `churn_customers`: Thông tin nhân khẩu học và trạng thái vòng đời khách hàng.
* `churn_subscriptions`: Gói cước (Free, Plus, Pro), lịch sử nâng/hạ cấp và chu kỳ thanh toán.
* `churn_product_usage`: Ngày hoạt động và tần suất/khối lượng sử dụng ứng dụng hàng ngày.
* `churn_orders` & `churn_payments`: Lịch sử đơn hàng, giá trị giao dịch, tỷ lệ thanh toán thành công/thất bại.
* `churn_support_tickets`: Nhật ký yêu cầu hỗ trợ kỹ thuật và chỉ số hài lòng CSAT.
* `churn_marketing_interactions`: Lượt hiển thị (impressions) và tương tác (clicks) với các chiến dịch tiếp thị.

---

## 3. Temporal Feature Engineering (379 Đặc trưng)

Bộ đặc trưng được tổng hợp từ dữ liệu lịch sử hoàn thành trước ngày snapshot $S$ (không sử dụng thông tin tương lai):
* **Recency Features**: Số ngày trôi qua kể từ lần tương tác gần nhất (usage, order, payment, ticket, downgrade).
* **Lag Features**: Trượt thời gian 1, 2 và 3 tháng trước snapshot cho các biến hành vi cơ sở.
* **Rolling Features**: Thống kê tổng hợp (sum, mean, std, min, max) trên các cửa sổ trượt 1, 3 và 6 tháng.
* **Trend & Slope Features**: Chênh lệch tuyệt đối 1 tháng, tỷ lệ phần trăm thay đổi, và độ dốc hồi quy tuyến tính 3 tháng.

---

## 4. Anti-Leakage Protocol & Chronological Splitting

Để phản ánh chính xác hiệu năng triển khai thực tế và ngăn ngừa rò rỉ dữ liệu (Lookahead Bias):
* **Tập Huấn luyện (Train Set)**: Snapshot từ `2024-09-01` đến `2025-08-01` (12 tháng snapshot trượt).
* **Tập Hiệu chuẩn (Validation Set)**: Snapshot từ `2025-09-01` đến `2026-02-01` (6 tháng snapshot độc lập dùng để tune threshold và Platt Calibrator).
* **Tập Kiểm thử (Clean Test Set)**: Snapshot từ `2026-03-01` đến `2026-06-01` (4 tháng snapshot hoàn toàn mới, khóa độc lập để đánh giá out-of-time).
* **Loại trừ nghiêm ngặt (`META_COLS`)**: Khóa định danh, cột nhãn thành phần và tất cả các trường chứa tiền tố `future_` / `_future`.

---

## 5. Model Evaluation & Benchmarks

Hiệu năng các mô hình được đánh giá trên tập **Clean Test độc lập** (35,336 quan sát, 3,486 churn events):

| Mô hình (Model) | ROC-AUC | PR-AUC | Precision | Recall | F1-Score | Ghi chú kiến trúc |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline Logistic Regression** | `0.9412` | `0.4821` | `49.80%` | `88.50%` | `63.72%` | Chuẩn hóa StandardScaler, điểm sàn kiểm định rò rỉ |
| **XGBoost Classifier** | `0.9498` | `0.5310` | `53.15%` | `92.80%` | `67.61%` | Gradient Boosting kiểm soát chặt Overfitting ($L_1/L_2$) |
| **LightGBM Classifier** | `0.9513` | `0.5382` | `54.32%` | `93.37%` | `68.69%` | Leaf-wise GBDT, hiệu năng tối ưu trên bảng dữ liệu lớn |
| **Stacking Ensemble (V2)** | `0.9535` | `0.5441` | `55.10%` | `93.85%` | `69.36%` | 2-Level Stacking: LR + LightGBM + XGBoost $\to$ Meta-Learner |

---

## 6. Project Structure (Cấu trúc thư mục)

```text
├── Baseline_LogisticRegression_v2.ipynb   # Notebook 1: Baseline Linear Model & Data Leakage Audit
├── Advanced_Ensemble_Modeling_v2.ipynb    # Notebook 2: Advanced Ensemble (LR, LightGBM, XGBoost, Stacking)
├── 01_baseline_logistic_regression_phan_tich.md # Tài liệu phân tích chuyên sâu Notebook 1
├── 02_advanced_ensemble_phan_tich.md           # Tài liệu phân tích chuyên sâu Notebook 2
├── CHURN_PROJECT_REPORT.md               # Báo cáo kỹ thuật chi tiết toàn diện của dự án
├── FINAL_MODEL_SUMMARY.md                # Tóm tắt mô hình sản xuất và hướng dẫn nạp mô hình
├── TEMPORAL_FEATURE_REDUCTION_PLAN.md    # Kế hoạch giảm chiều đặc trưng và tối ưu hóa chi phí
├── churn_feature_dataset_processed.csv   # Tập dữ liệu đặc trưng đã tiền xử lý
├── requirements.txt                      # Danh sách các thư viện phụ thuộc
├── .env.example                          # Mẫu cấu hình môi trường
├── .gitignore                            # Cấu hình bỏ qua các file tạm, môi trường ảo và output
├── artifacts/                            # Gói mô hình đã huấn luyện (.joblib)
│   ├── advanced_ensemble_churn_v2.joblib # Mô hình Stacking Ensemble V2 đã đóng gói
│   └── temporal_churn_model_v2.joblib    # Mô hình LightGBM v2 đã đóng gói
├── processed_feature_model/              # Pipeline mô hình độc lập trên tập đặc trưng rút gọn
└── churn_*/                              # Phân vùng dữ liệu Silver Parquet (Customers, Orders, Usage...)
```

---

## 7. Tài Liệu Nghiên Cứu & Báo Cáo Chuyên Sâu

Các tài liệu phân tích chi tiết được lưu trữ trực tiếp trong kho lưu trữ:
1. **[01_baseline_logistic_regression_phan_tich.md](01_baseline_logistic_regression_phan_tich.md)**: Phân tích cơ sở lý thuyết, ý nghĩa toán học của các trọng số hồi quy, ma trận nhầm lẫn và thang đo phát hiện Data Leakage.
2. **[02_advanced_ensemble_phan_tich.md](02_advanced_ensemble_phan_tich.md)**: Phân tích kiến trúc Stacking Ensemble 2 tầng, ma trận Out-Of-Fold (OOF), tương tác phi tuyến và cơ chế kiểm soát overfitting.
3. **[CHURN_PROJECT_REPORT.md](CHURN_PROJECT_REPORT.md)**: Báo cáo kỹ thuật tổng thể từ dữ liệu thô Silver đến triển khai mô hình.
4. **[FINAL_MODEL_SUMMARY.md](FINAL_MODEL_SUMMARY.md)**: Tổng kết các chỉ số hiệu năng và quy trình tải gói mô hình phục vụ suy diễn.

---

## 8. Hướng Dẫn Cài Đặt & Chạy Thử (Quickstart)

### Bước 1: Khởi tạo môi trường và cài đặt thư viện
```bash
# Tạo môi trường ảo (tùy chọn)
python -m venv .venv
source .venv/bin/activate  # Trên Linux/macOS
# hoặc: .venv\Scripts\activate trên Windows

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### Bước 2: Chạy các Jupyter Notebook
Khởi động Jupyter Lab hoặc Jupyter Notebook để chạy và kiểm tra kết quả:
* Mở và thực thi **`Baseline_LogisticRegression_v2.ipynb`** để kiểm tra điểm sàn hiệu năng và kiểm định Data Leakage.
* Mở và thực thi **`Advanced_Ensemble_Modeling_v2.ipynb`** để huấn luyện so sánh Logistic Regression, LightGBM, XGBoost và Stacking Ensemble.

### Bước 3: Nạp và Sử dụng Mô hình đã Huấn luyện
```python
import joblib
import pandas as pd

# Nạp model bundle đã đóng gói
bundle = joblib.load("artifacts/temporal_churn_model_v2.joblib")
model = bundle["model"]
calibrator = bundle["calibrator"]
selected_features = bundle["selected_features"]
imputer = bundle["imputer"]
threshold = bundle["threshold"]

# Dự đoán trên dữ liệu mới
# X_new = df[selected_features]
# X_imp = imputer.transform(X_new)
# raw_probs = model.predict_proba(X_imp)[:, 1]
# cal_probs = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
# is_churn = (cal_probs >= threshold).astype(int)
```
