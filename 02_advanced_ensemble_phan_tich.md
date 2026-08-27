# Phân Tích Chi Tiết: Advanced_Ensemble_Modeling_v2.ipynb

Tài liệu này phân tích cấu trúc kiến trúc, cơ chế toán học, kỹ thuật học máy nâng cao và logic xử lý của từng khối mã trong file notebook **`Advanced_Ensemble_Modeling_v2.ipynb`**.

---

## 1. Mục Đích & Kiến Trúc Tổng Thể

### 1.1. Mục tiêu bài toán
Xây dựng một hệ thống mô hình dự đoán rời bỏ (Customer Churn Prediction) đạt hiệu năng tối ưu trên dữ liệu dạng bảng 379 đặc trưng thời gian (Rule v2) bằng cách kết hợp ưu điểm của 3 thuật toán nền tảng:
- **Logistic Regression:** Mô hình tuyến tính ổn định, cung cấp phân phối xác suất trơn.
- **LightGBM:** Mô hình cây quyết định tăng cường (GBDT) với cơ chế phân nhánh theo lá (Leaf-wise), xử lý tương tác phi tuyến phức tạp cực nhanh.
- **XGBoost:** Mô hình GBDT với hàm mục tiêu bổ sung thành phần phạt chuẩn $L_1/L_2$, kiểm soát hiện tượng quá khớp (overfitting) rất chặt chẽ.

### 1.2. Kiến trúc Stacking Ensemble 2 Tầng (2-Level Stacking)

```
                            ┌────────────────────────────────────────┐
                            │      TẬP HUẤN LUYỆN (TRAIN SET)        │
                            └───────────────────┬────────────────────┘
                                                │
                                                ▼
                            ┌────────────────────────────────────────┐
                            │    5-FOLD STRATIFIED CROSS-VALIDATION  │
                            └───────┬───────────┬───────────┬────────┘
                                    │           │           │
                     ┌──────────────┘           │           └──────────────┐
                     ▼                          ▼                          ▼
            [Logistic Regression]          [LightGBM]                  [XGBoost]
                     │                          │                          │
                     ▼                          ▼                          ▼
               oof_preds_lr               oof_preds_lgb              oof_preds_xgb
                     │                          │                          │
                     └──────────────┬───────────┴───────────┬──────────────┘
                                    │                       │
                                    ▼                       ▼
                   ┌─────────────────────────────────┐ ┌─────────────────────────────────┐
                   │   MA TRẬN META-FEATURES (OOF)   │ │ RETRAIN 3 BASE MODELS TRÊN FULL │
                   │  X_meta = [oof_lr, lgb, xgb]    │ │       TRAIN ĐỂ SỬ DỤNG TEST     │
                   └────────────────┬────────────────┘ └────────────────┬────────────────┘
                                    │                                   │
                                    ▼                                   │
                   ┌─────────────────────────────────┐                  │
                   │    LEVEL-1 META-LEARNER (FIT)   │                  │
                   │      (Logistic Regression)      │                  │
                   └────────────────┬────────────────┘                  │
                                    │                                   │
                                    ▼                                   ▼
                               ┌─────────────────────────────────────────────┐
                               │  DỰ ĐOÁN TRÊN TẬP KIỂM THỬ ĐỘC LẬP (TEST)   │
                               │   P_ensemble = MetaLearner(P_lr, P_lgb, P_xgb)│
                               └─────────────────────────────────────────────┘
```

---

## 2. Phân Tích Chi Tiết Từng Phần (Sections / Cells)

---

### Section 1: Cấu Hình Môi Trường & Xác Thực Phiên Bản Thư Viện
- **Mục tiêu phân tích:** Đảm bảo toàn bộ các thư viện cốt lõi (`lightgbm`, `xgboost`, `scikit-learn`, `pandas`, `numpy`, `joblib`) hoạt động đồng bộ, tránh lỗi phân mảnh API hoặc không tương thích phiên bản khi đóng gói mô hình.
- Cố định hạt giống ngẫu nhiên `SEED = 42` trên toàn bộ các bộ phân chia dữ liệu và khởi tạo trọng số mô hình.

---

### Section 2: Đọc Dữ Liệu v2 & Lọc Đặc Trưng Độc Lập
- Đọc tập dữ liệu chuẩn hóa `output/churn_temporal_dataset_v2.parquet` (185,160 dòng).
- Loại bỏ toàn bộ danh sách `META_COLS` (nhãn `churn_next_30d`, các cờ quy tắc `rule1/rule2/rule3`, thông tin đóng tài khoản `closed_date`, loại gói `tier_at_snapshot` và lý do `churn_reason`).
- Đảm bảo ma trận đặc trưng $X$ chỉ chứa đúng **379 biến số thực tế** đã được đo lường trước hoặc tại thời điểm snapshot.

---

### Section 3: Phân Chia Dòng Thời Gian (Chronological Splitting)
- **Tập Train (`2024-09` → `2025-08`):** Chứa dữ liệu 12 tháng liên tiếp. Dùng để chạy 5-Fold Cross-Validation sinh OOF và retrain các mô hình cơ sở.
- **Tập Validation (`2025-09` → `2026-02`):** Chứa 6 tháng tiếp theo. Dùng để đánh giá trong quá trình Grid Search và tìm ngưỡng phân loại tối ưu (Optimal Threshold).
- **Tập Test (`2026-03` → `2026-06`):** Chứa 4 tháng độc lập trong tương lai. Dùng làm bài kiểm tra cuối cùng để so sánh 4 mô hình một cách khách quan nhất.

---

### Section 4: Tiền Xử Lý Dữ Liệu Hai Luồng (Two-track Preprocessing)
- **Luồng 1 — Xử lý giá trị thiếu (Median Imputation):**
  - Sử dụng `SimpleImputer(strategy='median')` fit trên Train để tạo ra ma trận `X_imp` phục vụ trực tiếp cho LightGBM và XGBoost (giúp đồng bộ hóa dữ liệu đầu vào cho các thuật toán cây).
- **Luồng 2 — Chuẩn hóa thang đo (Z-score Standardization):**
  - Sử dụng `StandardScaler()` biến đổi ma trận `X_imp` thành `X_scaled` dành riêng cho Logistic Regression. Cây quyết định không cần chuẩn hóa thang đo, nhưng mô hình tuyến tính bắt buộc phải có bước này để tránh việc các đặc trưng có biên độ lớn (như tổng chi tiêu `spend`) lấn át các đặc trưng có biên độ nhỏ (như số lần đăng nhập `login_count`).

---

### Section 5: Grid Search Tối Ưu Siêu Tham Số LightGBM Theo AUCPR
- **Cơ sở lý luận chọn chỉ số mục tiêu:**
  - Trong bài toán phân loại nhị phân mất cân bằng lớp (Imbalanced Binary Classification), **ROC-AUC có thể bị sai lệch lạc quan** do kích thước lớn của lớp đa số (True Negatives).
  - **PR-AUC (Precision-Recall Area Under Curve / AUCPR)** chỉ tập trung vào lớp thiểu số (Khách hàng Churn), phản ánh trực tiếp sự đánh đổi giữa độ chuẩn xác (Precision) và độ bao phủ (Recall).
- **Không gian tham số tìm kiếm:**
  - `num_leaves`: Số lượng lá tối đa trong một cây (kiểm soát độ phức tạp của phân vùng phi tuyến).
  - `max_depth`: Độ sâu tối đa để giới hạn chiều sâu cây, tránh overfit.
  - `learning_rate`: Tốc độ học (co ngót trọng số từng cây).
  - `colsample_bytree` & `subsample`: Tỷ lệ lấy mẫu đặc trưng và dòng dữ liệu ngẫu nhiên tại mỗi vòng lặp.
  - `reg_alpha` (L1) & `reg_lambda` (L2): Các hệ số phạt điều chuẩn.
- **Tiêu chí dừng:** Thuật toán tự động lưu cấu hình có điểm **Validation AUCPR** cao nhất.

---

### Section 6: Khởi Tạo 3 Mô Hình Cơ Sở (Level-0 Base Models)
- Khởi tạo 3 kiến trúc đại diện cho 3 trường phái mô hình hóa:
  1. `model_lr`: Logistic Regression chuẩn hóa L2 với bộ giải tối ưu `lbfgs`.
  2. `model_lgb`: LightGBM Classifier sử dụng cấu hình siêu tham số tối ưu từ Section 5 kèm tham số cân bằng trọng số lớp `scale_pos_weight`.
  3. `model_xgb`: XGBoost Classifier cấu hình thuật toán xây dựng cây histogram (`tree_method='hist'`) với regularization mạnh mẽ.

---

### Section 7: 5-Fold Stratified CV Sinh Ma Trận Out-Of-Fold (OOF)
- **Cơ chế Out-Of-Fold (OOF) giải quyết vấn đề gì?**
  - Nếu ta huấn luyện mô hình Level-0 trên tập Train rồi dự đoán lại chính tập Train để làm dữ liệu huấn luyện cho Meta-Learner, Meta-Learner sẽ bị hiện tượng **Overfitting trầm trọng** (vì dự đoán trên tập đã học luôn có độ tin cậy bị thổi phồng).
  - Bằng cách chia tập Train thành $K=5$ Folds: Mô hình được huấn luyện trên 4 Folds và dự đoán trên 1 Fold còn lại. Quá trình lặp lại 5 lần giúp ta thu được trọn vẹn xác suất dự đoán cho toàn bộ tập Train, nhưng tại mỗi điểm dữ liệu, xác suất đó hoàn toàn là dự đoán "chưa từng thấy" (Out-Of-Fold).
- **Kết quả thu được:** Ba vector xác suất `oof_lr`, `oof_lgb`, `oof_xgb` phản ánh trung thực khả năng tổng quát hóa của từng mô hình cơ sở trên dữ liệu huấn luyện.

---

### Section 8: Huấn Luyện Meta-Learner & Retrain Trên Full Train
- **Huấn luyện Meta-Learner (Level-1):**
  - Xây dựng ma trận đặc trưng cấp cao: $X_{\text{meta}} = [\mathbf{p}_{\text{LR}}, \mathbf{p}_{\text{LGBM}}, \mathbf{p}_{\text{XGB}}]_{N \times 3}$.
  - Sử dụng Logistic Regression làm Meta-Learner:
    $$P_{\text{ensemble}} = \sigma\left(\beta_0 + \beta_1 p_{\text{LR}} + \beta_2 p_{\text{LGBM}} + \beta_3 p_{\text{XGB}}\right)$$
  - Các hệ số $\beta_1, \beta_2, \beta_3$ phản ánh trọng số tin cậy tối ưu mà Meta-Learner gán cho từng thuật toán con.
- **Retrain trên Full Train:**
  - Sau khi Meta-Learner đã được fit trên ma trận OOF, cả 3 mô hình cơ sở được huấn luyện lại trên 100% dữ liệu của tập Train để đảm bảo tận dụng tối đa lượng mẫu khi bước vào giai đoạn dự đoán tập Test.
- **Chuyển đổi dự đoán tập Test:**
  - Ba mô hình cơ sở sinh 3 vector xác suất trên tập Test: $P_{\text{test\_lr}}, P_{\text{test\_lgb}}, P_{\text{test\_xgb}}$.
  - Ma trận này được đưa qua Meta-Learner để tạo ra xác suất kết hợp cuối cùng $P_{\text{test\_ensemble}}$.

---

### Section 9: Đánh Giá So Sánh 4 Thuật Toán Trên Tập Test Độc Lập
- **Quy trình đánh giá chuẩn:**
  1. Dò tìm ngưỡng phân loại tối ưu $T^*$ trên tập Validation để tối đa hóa điểm $F_1$.
  2. Áp dụng ngưỡng $T^*$ vào tập Test để tính ma trận nhầm lẫn và các chỉ số vận hành thực tế.
  3. Tính toán toàn bộ 6 chỉ số: `AUCPR`, `ROC-AUC`, `F1-Score`, `Precision`, `Recall`, `Brier Score`.
- **Tiêu chuẩn quyết định mô hình chiến thắng (Decision Hierarchy):**
  $$\text{Ưu tiên 1: } \max(\text{AUCPR}) \quad \longrightarrow \quad \text{Ưu tiên 2 (nếu hòa): } \max(\text{ROC-AUC})$$
- Tự động đánh dấu mô hình đạt hiệu năng tốt nhất làm `Best Model`.

---

### Section 10: Trực Quan Hóa So Sánh Đa Chiều (ROC & PR Curves)
- **Đồ thị 1 (ROC Curves):** Đánh giá năng lực phân loại tổng thể của 4 mô hình trên toàn dải ngưỡng phân loại.
- **Đồ thị 2 (Precision-Recall Curves):** Phân tích trực quan xem mô hình nào duy trì được độ chính xác (Precision) cao nhất khi ta mở rộng độ bao phủ (Recall) để thu hồi tối đa lượng khách hàng có nguy cơ rời bỏ.

---

### Section 11: Phân Tích Tầm Quan Trọng Đặc Trưng (Variable Importance — GBM)
- Trích xuất điểm **Split Importance Gain** từ mô hình LightGBM (tổng mức giảm mất mát Log-loss thu được khi phân chia tại các nút sử dụng đặc trưng đó).
- Xếp hạng Top 25 đặc trưng quan trọng nhất (như `days_since_last_activity`, `payment_success_rolling_mean`, `spend_trend`) và xuất dữ liệu ra file `output/feature_importance_ensemble_v2.csv`.

---

### Section 12: Đóng Gói Toàn Diện Model Bundle (`.joblib`)
- **Cấu trúc gói lưu trữ (`artifacts/advanced_ensemble_churn_v2.joblib`):**
  - Bộ tiền xử lý: `imputer` (Median), `scaler` (StandardScaler).
  - Ba mô hình cơ sở đã retrain: `base_model_lr`, `base_model_lgb`, `base_model_xgb`.
  - Mô hình meta: `meta_learner`.
  - Danh sách đặc trưng: `feature_names` (379 tên biến).
  - Ngưỡng phân loại tối ưu: `selected_threshold`.
  - Bảng chỉ số kiểm định: `comparison_metrics`.
- **Kiểm thử tính toàn vẹn (Sanity Check):** Tải lại bundle từ đĩa và thực hiện dự đoán thử nghiệm trên 5 mẫu kiểm thử để đảm bảo pipeline suy luận (Inference Pipeline) hoạt động trơn tru không lỗi.

---

### Section 13: Tổng Kết Vận Hành & Hướng Triển Khai
- Phân tích ưu thế của giải pháp Ensemble trong môi trường sản xuất (Production):
  - Khả năng chống chịu lỗi (Fault tolerance) tốt hơn mô hình đơn lẻ.
  - Phân phối xác suất đầu ra mượt mà, giúp bộ phận nghiệp vụ phân nhóm khách hàng (Customer Risk Segmentation) chính xác theo các dải xác suất nguy cơ cao, trung bình, thấp.
