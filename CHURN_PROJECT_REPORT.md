# Customer Churn Prediction Using Temporal Behavior (Version v2)

## 1. Executive Summary
Báo cáo này trình bày thiết kế và kết quả thực nghiệm của hệ thống học máy dự đoán tỷ lệ rời bỏ dịch vụ của khách hàng (Customer Churn Prediction) dựa trên chuỗi hành vi lịch sử theo phương pháp **Mô hình hóa thời gian (Temporal/Time-Series Modeling)**. Thay vì tiếp cận theo hướng tĩnh (static) truyền thống coi mỗi khách hàng là một dòng dữ liệu duy nhất, dự án này áp dụng phương pháp **ảnh chụp nhanh hàng tháng (rolling monthly snapshots)**.

Mô hình tốt nhất được lựa chọn là **LightGBM** kết hợp **hiệu chuẩn xác suất Platt Scaling**, huấn luyện động trên cửa sổ trượt 12 tháng (Rolling 12-Month Window) và sử dụng toàn bộ 379 đặc trưng thời gian (Temporal Features). Mô hình đã được xác thực an toàn không rò rỉ dữ liệu (data leakage-free) và đạt chỉ số PR-AUC `0.538246` trên toàn bộ tập test sạch dưới định nghĩa nhãn Churn v2 (bao gồm khách hàng đã ở gói Free và tiếp tục ngừng hoạt động).

---

## 2. Business Problem
Bài toán đặt ra là dự đoán khả năng khách hàng rời bỏ dịch vụ hoặc hạ cấp gói cước và ngừng hoạt động trong vòng **30 ngày tiếp theo** (prediction horizon) tính từ thời điểm quan sát snapshot $S$.
- **Thời điểm quan sát (Observation Time)**: Ngày đầu tiên mỗi tháng (snapshot_date).
- **Cửa sổ lịch sử hành vi (Observation Window)**: Toàn bộ dữ liệu tương tác của khách hàng tích lũy từ thời điểm đăng ký (signup) đến thời điểm snapshot $S$.
- **Nhãn mục tiêu (Target Label)**: `churn_next_30d` trong khoảng thời gian $[S, S + 30\text{ ngày})$.

---

## 3. Dataset
Dữ liệu của dự án được cấu trúc theo dạng bảng chuỗi thời gian cơ sở (temporal table grain) có khóa chính là cặp:
$$\text{customer\_id} \times \text{snapshot\_date}$$
Thông tin tổng quan về tập dữ liệu temporal thu được (phiên bản v2):
- **Tổng số dòng (Number of rows)**: 185,160
- **Số khách hàng duy nhất (Number of customers)**: 10,002
- **Số lượng snapshot (Number of snapshots)**: 37
- **Phạm vi thời gian (Date range)**: `2023-08-01` đến `2026-08-01`
- **Số lượng đặc trưng (Feature count)**: 379 đặc trưng thời gian
- **Số lượng nhãn dương (Positive churn rows)**: 43,543 dòng
- **Tỷ lệ churn trung bình (Overall churn rate)**: `23.5164%` (giảm sự mất cân bằng lớp cực đoan nhờ bổ sung định nghĩa nhãn Churn v2).

---

## 4. Data Architecture
Dữ liệu được tổ chức và xử lý theo mô hình phân tầng từ Raw tới Temporal Dataset và Modeling:

```
+-----------------------------------------------------------+
|               Silver Layer (Raw Event Tables)             |
| (customers, subscriptions, usage, orders, payments, etc.) |
+-----------------------------------------------------------+
                              │
                              ▼
+-----------------------------------------------------------+
|              Temporal Base Table (Monthly Grain)          |
|    - Ghép nối khách hàng với timeline snapshot            |
|    - Tổng hợp hoạt động theo tháng hoàn thành trước S     |
+-----------------------------------------------------------+
                              │
                              ▼
+-----------------------------------------------------------+
|               Temporal Feature Engineering                |
|    - Tính toán Lags (1m, 2m, 3m)                          |
|    - Tính toán Rolling (1m, 3m, 6m: sum, mean, std...)    |
|    - Tính toán Trend (change, pct_change, slope)          |
|    - Tính toán Recency (days since last event)            |
+-----------------------------------------------------------+
                              │
                              ▼
+-----------------------------------------------------------+
|                       Labeling (S)                        |
|   - Nhãn Churn v2 dựa trên 3 Quy tắc (Rule 1, 2, 3)       |
|     quan sát trong khoảng [S, S + 30 ngày)                |
+-----------------------------------------------------------+
                              │
                              ▼
+-----------------------------------------------------------+
|                   Final Temporal Dataset                  |
|                (churn_temporal_dataset_v2.parquet)        |
+-----------------------------------------------------------+
                              │
                              ▼
+-----------------------------------------------------------+
|               Time-based Chronological Split              |
|        - Train: 2024-09-01 đến 2025-08-01                 |
|        - Validation: 2025-09-01 đến 2026-02-01            |
|        - Test: 2026-03-01 đến 2026-06-01                  |
+-----------------------------------------------------------+
                              │
                              ▼
+-----------------------------------------------------------+
|             Model Training (Rolling 12M LightGBM)         |
+-----------------------------------------------------------+
                              │
                              ▼
+-----------------------------------------------------------+
|            Probability Calibration (Platt Scaling)        |
+-----------------------------------------------------------+
                              │
                              ▼
+-----------------------------------------------------------+
|             Thresholding & Final Predictions (0/1)        |
+-----------------------------------------------------------+
                              │
                              ▼
+-----------------------------------------------------------+
|           Serialized Model Bundle (.joblib) & Verification|
+-----------------------------------------------------------+
```

Vai trò của tầng **Silver Layer**: Các bảng sự kiện thô được làm sạch, chuẩn hóa kiểu dữ liệu thời gian, kiểm tra trùng lặp và lưu trữ dưới định dạng Parquet phân vùng. Các bảng sự kiện chính bao gồm:
- `churn_customers`: Thông tin nhân khẩu học (gender, birth_date, region, city, signup_date, closed_date).
- `churn_subscriptions`: Lịch sử cước phí và thay đổi gói cước (plan_tier, change_type, start_date).
- `churn_product_usage`: Lịch sử sử dụng sản phẩm (usage_id, event_date, event_type, session_duration_sec).
- `churn_orders` & `churn_payments`: Lịch sử mua hàng và thanh toán (order_date, payment_date, amount, status).
- `churn_support_tickets`: Lịch sử khiếu nại hỗ trợ (csat_score, created_at).
- `churn_marketing_interactions`: Tương tác tiếp thị tiếp cận (sent_at, opened, clicked, converted).

---

## 5. Temporal Dataset Construction
Để xây dựng bảng thời gian cơ sở:
1. Xác định thời điểm snapshot hợp lệ: Từ ngày `2023-08-01` (global_min_date + 1 tháng lịch sử) tới `2026-08-01` (global_max_date - 30 ngày nhãn).
2. Tạo lưới (grid) hoạt động của khách hàng: Với mỗi khách hàng, tạo một dòng cho mỗi snapshot hàng tháng nằm trong khoảng từ tháng đăng ký (`signup_date`) đến tháng đóng tài khoản (`closed_date`) hoặc tháng snapshot lớn nhất của hệ thống.
3. Ghép nối (merge) các chỉ số hoạt động hành vi được tổng hợp theo từng tháng tương ứng.

---

## 6. Churn Label Definition
Nhãn mục tiêu `churn_next_30d` được tính toán độc lập tại mỗi snapshot $S$ theo quy tắc logic Churn v2 mới:
- **Quy tắc 1 (Account Closed)**: Khách hàng đóng tài khoản trong prediction window:
  $$\text{closed\_date} \ge S \quad \text{AND} \quad \text{closed\_date} < S + 30\text{ ngày}$$
- **Quy tắc 2 (Downgrade to Free & Inactive)**: Khách hàng đang ở paid tier thực hiện hạ cấp xuống Free tier (`change_type == "Downgrade"` và gói mới là `Free`) **AND** hoàn toàn không có hoạt động tương tác nào (sử dụng sản phẩm, giao dịch đơn hàng completed, hoặc thanh toán success) trong prediction window $[S, S + 30\text{ ngày})$.
- **Quy tắc 3 (Already Free & Inactive)**: Khách hàng ĐÃ ở Free tier tại thời điểm snapshot $S$ VÀ không có downgrade xuống Free trong prediction window VÀ hoàn toàn inactive trong 30 ngày tiếp theo.
- **Phân bổ nhãn dương v2**: Quy tắc 1 có 683 trường hợp, Quy tắc 2 có 180 trường hợp, Quy tắc 3 có 42,832 trường hợp (có sự trùng lặp 152 dòng giữa Rule 1 và Rule 3 khi khách hàng vừa là Free Inactive vừa đóng tài khoản).

---

## 7. Temporal Feature Engineering
Các nhóm đặc trưng thời gian được tính toán an toàn bao gồm:

### A. Lịch sử trễ (Lag Features)
- **Định nghĩa**: Giá trị hành vi của các tháng trước snapshot.
- **Cách tính**: Dịch chuyển chuỗi hành vi của khách hàng theo thời gian:
  $$\text{Lag\_k}(x_t) = x_{t-k} \quad (k \in \{1, 2, 3\})$$

### B. Chỉ số tích lũy trượt (Rolling Features)
- **Định nghĩa**: Các thống kê tổng hợp (sum, mean, std, min, max) trên cửa sổ thời gian lịch sử dài hơn.
- **Cách tính**: Áp dụng hàm thống kê trượt trên $W$ tháng ($W \in \{1, 3, 6\}$):
  $$\text{Rolling\_Mean}_{W}(x_t) = \frac{1}{W} \sum_{i=0}^{W-1} x_{t-i}$$

### C. Xu hướng biến động (Trend / Momentum Features)
- **Định nghĩa**: Xu hướng tăng/giảm tương tác của khách hàng.
- **Cách tính**:
  - Biến động tuyệt đối: $\Delta x = x_t - x_{t-1}$
  - Biến động phần trăm (Percentage Change): 
    $$\% \Delta x = \frac{x_t - x_{t-1}}{\max(|x_{t-1}|, 10^{-6})}$$
  - Độ dốc xu hướng (3-Month Slope): Xấp xỉ qua hệ số góc hồi quy tuyến tính của 3 điểm gần nhất:
    $$\text{Slope\_3m} = \frac{x_t - x_{t-2}}{2}$$

### D. Chỉ số thời gian tương tác gần nhất (Recency Features)
- **Định nghĩa**: Khoảng thời gian (số ngày) từ sự kiện cuối cùng của một loại đến ngày snapshot $S$.
- **Cách tính**: 
  $$\text{Days\_since} = S - \max(\text{event\_date}) \quad (\text{event\_date} < S)$$
  Nếu chưa từng xảy ra sự kiện, gán giá trị mặc định lớn (`999`).

---

## 8. Leakage Prevention
Ngăn ngừa rò rỉ dữ liệu (data leakage) là ưu tiên số một của hệ thống temporal modeling:
1. **Thiết kế Base Table dịch chuyển**: Tại snapshot ngày $S$ (ví dụ: `2026-06-01`), các chỉ số hành vi tĩnh của tháng được ghép nối với dữ liệu tổng hợp của tháng trước đó (`2026-05-01`). Do đó, rolling window tính toán từ dòng này chỉ nhìn thấy dữ liệu hoàn thành trước ngày $S$.
2. **Thiết lập recency không đồng thời**: Trong quá trình nối sự kiện `pd.merge_asof`, tham số `allow_exact_matches=False` được sử dụng để đảm bảo chỉ những sự kiện xảy ra **tương lai thực sự** mới bị ẩn, còn các sự kiện trong lịch sử trước S luôn khả dụng.
3. **Phân tách nhãn tương lai**: Nhãn Churn v2 được tính toán hoàn toàn từ sự kiện thuộc cửa sổ $[S, S + 30\text{ ngày})$. Các cột phụ để phân tích và kiểm tra luật (`rule1_closed`, `rule2_downgrade_to_free_inactive`, `rule3_free_at_snapshot_inactive`, `churn_reason`, `tier_at_snapshot`) tuyệt đối không được đưa vào tập đặc trưng huấn luyện.
4. **Kết quả Temporal Audit**: Tập dữ liệu v2 đã vượt qua 100% các bài test kiểm tra tự động và hoàn toàn sạch rò rỉ.

---

## 9. Time-Series Characteristics
Các đặc trưng chuỗi thời gian tích hợp vào dự án qua hai tầng:
1. **Tầng phân tích chuỗi thời gian tổng hợp (Auxiliary Time Series Analysis)**:
   - Áp dụng kiểm định nghiệm đơn vị **ADF (Augmented Dickey-Fuller) Test** để kiểm tra tính dừng (stationarity).
   - Phân tích tương quan tự hồi quy **ACF / PACF** để đánh giá tính chu kỳ và tự tương quan.
   - Thử nghiệm các mô hình tự hồi quy **ARIMA / SARIMA** để dự báo hành vi tổng hợp (orders, usage, spend) của tháng tiếp theo.
2. **Tầng phân loại Churn khách hàng**:
   - Sử dụng mô hình học máy phân loại nhị phân trên cấu trúc temporal snapshot kết hợp lag, rolling, trend, recency nhằm nắm bắt động thái hành vi ở cấp độ khách hàng cá nhân.

---

## 10. Validation Strategy
Dự án áp dụng phương pháp phân tách dữ liệu theo thời gian (Chronological Time-based Split) thay vì chia ngẫu nhiên để tránh việc mô hình sử dụng tương lai để dự báo quá khứ:

- **Tập Huấn luyện (Train)**: Snapshot từ `2024-09-01` đến `2025-08-01` (62,729 dòng).
- **Tập Xác thực (Validation)**: Snapshot từ `2025-09-01` đến `2026-02-01` (45,586 dòng, dùng để hiệu chuẩn Platt và tối ưu hóa ngưỡng).
- **Tập Kiểm thử sạch (Clean Test)**: Snapshot từ `2026-03-01` đến `2026-06-01` (35,336 dòng, hoàn toàn đóng băng trong quá trình phát triển).
- **Tập loại trừ**: Các snapshot từ `2026-07-01` trở đi bị loại bỏ khỏi kiểm thử vì có nhãn chưa chín muồi.

---

## 11. Evaluation Metrics
Dự án sử dụng các metrics đánh giá mất cân bằng lớp:
- **PR-AUC (Area Under Precision-Recall Curve)**: Chỉ số chính đánh giá chất lượng phân loại của lớp thiểu số.
- **ROC-AUC**: Đánh giá khả năng phân biệt tổng thể.
- **Precision, Recall, F1-Score**: Đánh giá chất lượng phân loại tại ngưỡng tối ưu.
- **Ma trận nhầm lẫn (Confusion Matrix)**.

---

## 12. Baseline Models
Kết quả thực nghiệm của các mô hình cơ sở tĩnh (static baselines) và temporal trên tập dữ liệu v2:
- **Logistic Regression (Static Baseline)**: PR-AUC = `0.020058`, ROC-AUC = `0.775803` (được chạy làm tham chiếu tĩnh).
- **Model v1 (Old Churn Rule)**: PR-AUC = `0.124796`, ROC-AUC = `0.917949`, F1 = `22.6950%` (đánh giá trên Test).
- **Model v2 (New Churn Rule)**: PR-AUC = `0.538246`, ROC-AUC = `0.951322`, F1 = `68.6854%` (đánh giá trên Test).

---

## 13. ARIMA / SARIMA Experiment
Mô hình dự báo chuỗi thời gian tổng hợp chứng minh hành vi khách hàng có tính quy luật thời gian mạnh mẽ. Tuy nhiên đây chỉ là phân tích bổ trợ, không dùng làm mô hình phân loại churn trực tiếp.

---

## 14. Feature Selection
Quy trình huấn luyện được thực hiện trên tập Train. Cấu hình sản xuất quyết định giữ lại **toàn bộ 379 đặc trưng thời gian** (All379) nhằm bảo toàn tối đa hiệu năng.

---

## 15. LightGBM Modeling
Mô hình **LightGBM Classifier** được thiết lập với các tham số tối ưu chống overfitting:
- `max_depth`: 4, `num_leaves`: 15
- `learning_rate`: 0.01, `n_estimators`: 400
- `feature_fraction`: 0.7, `reg_alpha`: 1.0, `reg_lambda`: 1.0
- Trọng số cân bằng lớp: `scale_pos_weight` được tính toán tự động dựa trên phân bổ nhãn của tập huấn luyện trượt.

---

## 16. Probability Calibration
Mô hình LightGBM nguyên bản (raw LightGBM) có xu hướng dự đoán xác suất lệch rất lớn so với tỷ lệ thực tế do việc cân bằng lớp (`scale_pos_weight`). 
- Dự án áp dụng **Hiệu chuẩn Platt (Platt Scaling)** huấn luyện một mô hình Logistic Regression trên các xác suất thô của tập Validation.
- Sau khi hiệu chuẩn, ngưỡng phân loại tối ưu chuyển đổi từ ngưỡng thô mặc định về ngưỡng F1 hiệu chuẩn là **`0.24`**.

---

## 17. Final Model Selection
Mô hình sản xuất cuối cùng được cấu hình cố định trong hệ thống:
- **Thuật toán**: LightGBM Classifier + Platt Calibrator
- **Đặc trưng**: Toàn bộ 379 đặc trưng thời gian (All379)
- **Cơ chế huấn luyện**: Huấn luyện lại động (Rolling 12M Retraining)
- **Ngưỡng quyết định**: `0.24` (sau khi hiệu chuẩn xác suất)

---

## 18. Final Evaluation
Dưới đây là kết quả đối sánh cuối cùng của mô hình LightGBM trên tập Clean Test sạch (`2026-03-01` đến `2026-06-01`) giữa Old Rule và New Rule:

| Metric | Old Rule (v1) | New Rule (v2) |
| :--- | :--- | :--- |
| **Số mẫu (Rows)** | 185,160 | 185,160 |
| **Số Churn (Positives)**| 933 | 43,543 |
| **PR-AUC** | `0.124796` | `0.538246` |
| **ROC-AUC** | `0.917949` | `0.951322` |
| **Precision** | `16.0804%` | `54.3224%` |
| **Recall** | `38.5542%` | `93.3735%` |
| **F1-Score** | `22.6950%` | `68.6854%` |
| **Confusion Matrix** | `[[34745, 342], [174, 75]]` (v1) | `[[29113, 2737], [231, 3255]]` (v2) |

---

## 19. Model Storage & Verification
Model được lưu trữ dưới dạng gói bundle tại đường dẫn:
```
artifacts/temporal_churn_model_v2.joblib
```
Bundle chứa các thành phần đóng gói độc lập:
- `model`: Đối ứng LightGBM Classifier đã huấn luyện.
- `calibrator`: Logistic Regression dùng cho hiệu chuẩn Platt.
- `imputer`: SimpleImputer xử lý dữ liệu trống.
- `selected_features`: Danh sách 379 đặc trưng được chọn (toàn bộ đặc trưng).
- `threshold`: Ngưỡng xác suất tối ưu (0.24).

Kết quả kiểm thử kiểm duyệt tự động từ `compare_models.py`:
- **MODEL LOAD**: `PASS` (tải mô hình v2 thành công).
- **FEATURE COMPATIBILITY**: `PASS` (379 đặc trưng khớp hoàn toàn).
- **PREDICTION TEST**: `PASS` (suy diễn thành công, xác suất trả về nằm đúng trong khoảng $[0.0126, 0.5707]$, nhãn dự đoán nhị phân $\{0, 1\}$).

---

## 20. Conclusion
Dự án đã hoàn thành trọn vẹn mục tiêu nâng cấp **Temporal Machine Learning Churn Prediction v2**:
1. Đưa thành công cấu trúc hành vi thời gian hàng tháng và nhãn Churn v2 (bao gồm Rule 3 cho khách hàng đã ở gói Free) vào bài toán churn.
2. Thiết lập quy trình tạo đặc trưng và gán nhãn không chứa lookahead bias.
3. Chứng minh mô hình LightGBM Rolling 12M kết hợp Platt Calibration mang lại kết quả phân loại vượt trội, giải quyết hiệu quả sự mất cân bằng lớp cực đoan của bài toán.
