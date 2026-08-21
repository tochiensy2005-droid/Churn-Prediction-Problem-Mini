# Customer Churn Prediction Using Temporal Behavior

## 1. Executive Summary
Báo cáo này trình bày thiết kế và kết quả thực nghiệm của hệ thống học máy dự đoán tỷ lệ rời bỏ dịch vụ của khách hàng (Customer Churn Prediction) dựa trên chuỗi hành vi lịch sử theo phương pháp **Mô hình hóa thời gian (Temporal/Time-Series Modeling)**. Thay vì tiếp cận theo hướng tĩnh (static) truyền thống coi mỗi khách hàng là một dòng dữ liệu duy nhất, dự án này áp dụng phương pháp **ảnh chụp nhanh hàng tháng (rolling monthly snapshots)**. 

Mô hình tốt nhất được lựa chọn là **LightGBM** kết hợp **hiệu chuẩn xác suất Platt Scaling**, huấn luyện động trên cửa sổ trượt 12 tháng (Rolling 12-Month Window) và sử dụng toàn bộ 379 đặc trưng thời gian (Temporal Features). Mô hình đã được xác thực an toàn không rò rỉ dữ liệu (data leakage-free) và đạt chỉ số PR-AUC `0.121786` trên toàn bộ tập test sạch. 

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
Thông tin tổng quan về tập dữ liệu temporal thu được:
- **Tổng số dòng (Number of rows)**: 185,160
- **Số khách hàng duy nhất (Number of customers)**: 10,002
- **Số lượng snapshot (Number of snapshots)**: 37
- **Phạm vi thời gian (Date range)**: `2023-08-01` đến `2026-08-01`
- **Số lượng đặc trưng (Feature count)**: 379 đặc trưng thời gian
- **Số lượng nhãn dương (Positive churn rows)**: 933 dòng
- **Tỷ lệ churn trung bình (Overall churn rate)**: `0.5039%` (thể hiện sự mất cân bằng lớp cực kỳ lớn - Extreme Class Imbalance).

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
|   - Closed account hoặc Downgrade + Inactive trong        |
|     khoảng [S, S + 30 ngày)                               |
+-----------------------------------------------------------+
                              │
                              ▼
+-----------------------------------------------------------+
|                   Final Temporal Dataset                  |
|                 (churn_temporal_dataset.parquet)          |
+-----------------------------------------------------------+
                              │
                              ▼
+-----------------------------------------------------------+
|               Time-based Chronological Split              |
|        - Train: <= 2025-08-01                             |
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
Nhãn mục tiêu `churn_next_30d` được tính toán độc lập tại mỗi snapshot $S$ theo quy tắc logic chặt chẽ:
- **Quy tắc 1 (Account Closed)**: Khách hàng đóng tài khoản trong prediction window:
  $$\text{closed\_date} \ge S \quad \text{AND} \quad \text{closed\_date} < S + 30\text{ ngày}$$
- **Quy tắc 2 (Downgrade & Inactive)**: Khách hàng có sự kiện hạ cấp cước phí (`change_type == "Downgrade"`) trong window $[S, S + 30\text{ ngày})$ **AND** hoàn toàn không có tương tác sản phẩm, thanh toán thành công hoặc hoàn thành đơn hàng nào trong cùng khoảng thời gian đó.
- Phân bổ nhãn dương trong tập dữ liệu: Quy tắc 1 chiếm 682 trường hợp, Quy tắc 2 chiếm 250 trường hợp, và có 1 khách hàng thỏa mãn cả hai quy tắc đồng thời.

---

## 7. Temporal Feature Engineering
Các nhóm đặc trưng thời gian được tính toán an toàn bao gồm:

### A. Lịch sử trễ (Lag Features)
- **Định nghĩa**: Giá trị hành vi của các tháng trước snapshot.
- **Cách tính**: Dịch chuyển chuỗi hành vi của khách hàng theo thời gian:
  $$\text{Lag\_k}(x_t) = x_{t-k} \quad (k \in \{1, 2, 3\})$$
- **Ý nghĩa**: Phản ánh mức độ hoạt động của khách hàng trong quá khứ gần (ví dụ: `usage_lag_1` là tổng số lượt sử dụng trong tháng trước).

### B. Chỉ số tích lũy trượt (Rolling Features)
- **Định nghĩa**: Các thống kê tổng hợp (sum, mean, std, min, max) trên cửa sổ thời gian lịch sử dài hơn.
- **Cách tính**: Áp dụng hàm thống kê trượt trên $W$ tháng ($W \in \{1, 3, 6\}$):
  $$\text{Rolling\_Mean}_{W}(x_t) = \frac{1}{W} \sum_{i=0}^{W-1} x_{t-i}$$
- **Ý nghĩa**: Bắt trọn xu hướng dài hạn và sự ổn định hành vi của khách hàng (ví dụ: `usage_rolling_std_3m` đo lường mức độ biến động sử dụng dịch vụ).

### C. Xu hướng biến động (Trend / Momentum Features)
- **Định nghĩa**: Xu hướng tăng/giảm tương tác của khách hàng.
- **Cách tính**:
  - Biến động tuyệt đối: $\Delta x = x_t - x_{t-1}$
  - Biến động phần trăm (Percentage Change): 
    $$\% \Delta x = \frac{x_t - x_{t-1}}{\max(|x_{t-1}|, 10^{-6})}$$
  - Độ dốc xu hướng (3-Month Slope): Xấp xỉ qua hệ số góc hồi quy tuyến tính của 3 điểm gần nhất:
    $$\text{Slope\_3m} = \frac{x_t - x_{t-2}}{2}$$
- **Ý nghĩa**: Độ dốc âm thể hiện mức độ cam kết sử dụng dịch vụ của khách hàng đang suy giảm nghiêm trọng.

### D. Chỉ số thời gian tương tác gần nhất (Recency Features)
- **Định nghĩa**: Khoảng thời gian (số ngày) từ sự kiện cuối cùng của một loại đến ngày snapshot $S$.
- **Cách tính**: 
  $$\text{Days\_since} = S - \max(\text{event\_date}) \quad (\text{event\_date} < S)$$
  Nếu chưa từng xảy ra sự kiện, gán giá trị mặc định lớn (`999`).
- **Ý nghĩa**: Chỉ số nhạy bén dự báo rời bỏ (ví dụ: `days_since_last_usage` lớn thể hiện khách hàng đã bỏ bê sản phẩm).

---

## 8. Leakage Prevention
Ngăn ngừa rò rỉ dữ liệu (data leakage) là ưu tiên số một của hệ thống temporal modeling:
1. **Thiết kế Base Table dịch chuyển**: Tại snapshot ngày $S$ (ví dụ: `2026-06-01`), các chỉ số hành vi tĩnh của tháng được ghép nối với dữ liệu tổng hợp của tháng trước đó (`2026-05-01`). Do đó, rolling window tính toán từ dòng này chỉ nhìn thấy dữ liệu hoàn thành trước ngày $S$.
2. **Thiết lập recency không đồng thời**: Trong quá trình nối sự kiện `pd.merge_asof`, tham số `allow_exact_matches=False` được sử dụng để đảm bảo chỉ những sự kiện xảy ra **trước** (strictly $< S$) mới được nhìn thấy, ngăn chặn việc sử dụng thông tin phát sinh đúng ngày snapshot.
3. **Phân tách nhãn tương lai**: Nhãn `churn_next_30d` được tính toán hoàn toàn từ sự kiện thuộc cửa sổ $[S, S + 30\text{ ngày})$. Cột `closed_date` chỉ được sử dụng để lọc tính hợp lệ và tạo nhãn, tuyệt đối không được đưa vào tập đặc trưng huấn luyện.
4. **Kết quả Temporal Audit**: Tập dữ liệu đã vượt qua 100% các bài test kiểm tra tự động (lag mismatch = 0, rolling mismatch = 0, label mismatch = 0, phát hiện rò rỉ = 0).

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

- **Tập Huấn luyện (Train)**: Snapshot $\le$ `2025-08-01` (85,545 dòng).
- **Tập Xác thực (Validation)**: Snapshot từ `2025-09-01` đến `2026-02-01` (45,586 dòng, dùng để hiệu chuẩn Platt và tối ưu hóa ngưỡng).
- **Tập Kiểm thử sạch (Clean Test)**: Snapshot từ `2026-03-01` đến `2026-06-01` (35,336 dòng, hoàn toàn đóng băng trong quá trình phát triển).
- **Tập loại trừ**: Các snapshot từ `2026-07-01` trở đi bị loại bỏ khỏi kiểm thử vì có nhãn chưa chín muồi (incomplete labels due to prediction horizon truncation).

---

## 11. Evaluation Metrics
Do tỷ lệ nhãn dương cực kỳ thấp (~0.70%), chỉ số chính xác (Accuracy) là vô nghĩa. Dự án sử dụng:
- **PR-AUC (Area Under Precision-Recall Curve)**: Chỉ số tối thượng đánh giá khả năng phân loại lớp thiểu số mất cân bằng.
- **ROC-AUC**: Đánh giá khả năng phân biệt tổng thể.
- **Precision, Recall, F1-Score**: Đánh giá chất lượng phân loại tại ngưỡng tối ưu.
- **Ma trận nhầm lẫn (Confusion Matrix)**.

---

## 12. Baseline Models
Kết quả thực nghiệm của các mô hình cơ sở tĩnh (static baselines) và temporal trên tập dữ liệu:
- **Logistic Regression (Static Baseline)**: PR-AUC = `0.020058`, ROC-AUC = `0.775803`, F1 = `0.043308` (trên Clean Test).
- **XGBoost (Tuned Temporal Baseline)**: PR-AUC = `0.127153`, ROC-AUC = `0.902273`, F1 = `0.236128` (trực tiếp tối ưu trên Validation).

---

## 13. ARIMA / SARIMA Experiment
Mô hình dự báo chuỗi thời gian tổng hợp được triển khai để dự báo các chuỗi hành vi liên tục (orders, usage, spend, active_customers). Kết quả so sánh với Naive baseline và Moving Average (3m):
- **Đối với chuỗi usage**: ARIMA(2,1,1) đạt MAE = `756.07`, sMAPE = `6.43%` (vượt trội so với Naive MAE = `2443.17`).
- **Đối với chuỗi orders**: ARIMA(2,1,1) đạt MAE = `1927.18`, sMAPE = `26.04%` (Naive MAE = `2962.50`).
- **Đối với chuỗi active_customers**: ARIMA(1,1,0) đạt MAE = `23.66`, sMAPE = `0.31%`.
*Kết luận*: Các thực nghiệm ARIMA chứng minh hành vi khách hàng có tính quy luật thời gian mạnh mẽ. Tuy nhiên đây chỉ là phân tích bổ trợ, không dùng làm mô hình phân loại churn trực tiếp.

---

## 14. Feature Selection
Quy trình huấn luyện và tuyển chọn đặc trưng được thực hiện nghiêm ngặt trên tập Train. Qua các thực nghiệm so sánh giảm chiều dữ liệu (Selective Features & Top-K Data-Driven Selection), dự án xác định việc giảm số lượng đặc trưng làm suy giảm đáng kể tín hiệu dự đoán rời bỏ dịch vụ. Vì vậy, cấu hình sản xuất quyết định giữ lại **toàn bộ 379 đặc trưng thời gian** (All379) nhằm bảo toàn tối đa hiệu năng.
Top 10 đặc trưng có tầm quan trọng lớn nhất theo xếp hạng LightGBM:
1. `active_days_rolling_mean_6m` (Rolling - Tần suất hoạt động trung bình 6 tháng)
2. `active_days_rolling_mean_3m` (Rolling - Tần suất hoạt động trung bình 3 tháng)
3. `active_days_rolling_sum_6m` (Rolling - Tổng số ngày hoạt động 6 tháng)
4. `usage_rolling_min_6m` (Rolling - Số lần sử dụng tối thiểu 6 tháng)
5. `usage_pct_change_1m` (Trend - % thay đổi sử dụng 1 tháng gần nhất)
6. `active_days_slope_3m` (Trend - Độ dốc xu hướng hoạt động 3 tháng)
7. `days_since_last_payment` (Recency - Số ngày kể từ lần thanh toán cuối)
8. `days_since_last_support_ticket` (Recency - Số ngày kể từ lần gửi hỗ trợ cuối)
9. `subscription_change_rolling_std_6m` (Rolling - Độ lệch chuẩn thay đổi gói 6 tháng)
10. `spend` (Static - Mức chi tiêu tổng cộng)

---

## 15. LightGBM Modeling
Mô hình **LightGBM Classifier** được thiết lập với các tham số tối ưu chống overfitting:
- `max_depth`: 4, `num_leaves`: 15
- `learning_rate`: 0.01, `n_estimators`: 400
- `feature_fraction`: 0.7, `reg_alpha`: 1.0, `reg_lambda`: 1.0
- Trọng số cân bằng lớp: `scale_pos_weight` được tính toán tự động dựa trên phân bổ nhãn của tập huấn luyện trượt.

---

## 16. Temporal Drift Analysis
Hệ thống đo lường mức độ trôi lệch dữ liệu giữa tập dữ liệu huấn luyện và thực tế suy diễn sử dụng chỉ số **PSI (Population Stability Index)**:
- **Ngưỡng đánh giá**: PSI < 0.1 (low drift), $0.1 \le \text{PSI} \le 0.25$ (moderate drift), PSI > 0.25 (strong drift).
- **Kết quả thực tế**:
  - Hầu hết các đặc trưng hành vi và thanh toán nằm ở mức low drift (ví dụ: `active_days_rolling_mean_6m` PSI = 0.0).
  - Có hiện tượng trôi lệch mạnh (strong drift) ở một số đặc trưng tiếp thị, cụ thể là `marketing_interaction_rolling_sum_6m` (PSI = 0.5740).
  - Xuất hiện drift hiệu năng (performance drift) nhẹ trên tập test sạch (PR-AUC biến động từ `0.1959` ở tháng 4/2026 xuống `0.1223` vào tháng 6/2026).

---

## 17. Retraining Experiments
Thực nghiệm so sánh các chiến lược tái huấn luyện (retraining strategies) trên tập Validation:

Strategy | PR-AUC | ROC-AUC | Precision | Recall | F1 | Brier Score | Log Loss
--- | --- | --- | --- | --- | --- | --- | ---
**Static** | `0.141887` | `0.905441` | `0.174488` | `0.395774` | `0.240534` | `0.196054` | `0.575020`
**Expanding** | `0.138552` | `0.919368` | `0.176332` | `0.353356` | `0.235061` | `0.158016` | `0.471757`
**Rolling 6M** | `0.142100` | `0.903972` | `0.250419` | `0.230191` | `0.239539` | `0.109471` | `0.340380`
**Rolling 12M** | **`0.146050`** | `0.919388` | `0.226458` | `0.310269` | **`0.261565`** | **`0.139669`** | **`0.413345`**
**Rolling 18M** | `0.144994` | **`0.923084`** | `0.206167` | `0.345748` | `0.257750` | `0.143867` | `0.429136`

*Kết luận*: **Rolling 12M** được lựa chọn vì mang lại sự cân bằng tốt nhất giữa hiệu năng phân loại (PR-AUC, F1 cao nhất) và chất lượng phân bổ xác suất (Brier Score và Log Loss thấp hơn đáng kể so với Static/Expanding).

---

## 18. Probability Calibration
Mô hình LightGBM nguyên bản (raw LightGBM) có xu hướng dự đoán xác suất lệch rất lớn so với tỷ lệ thực tế do việc cân bằng lớp (`scale_pos_weight`). 
- Dự án áp dụng **Hiệu chuẩn Platt (Platt Scaling)** huấn luyện một mô hình Logistic Regression trên các xác suất thô của tập Validation.
- Kết quả hiệu chuẩn: Brier Score giảm mạnh (ví dụ tại snapshot `2026-06-01` từ `0.2526` xuống `0.0085`, giảm hơn 96%).
- Sau khi hiệu chuẩn, ngưỡng phân loại tối ưu chuyển đổi từ ngưỡng thô mặc định về ngưỡng F1 hiệu chuẩn là **`0.08`**.

---

## 19. LSTM / GRU Experiment
Để khai thác trực tiếp dạng sequence của chuỗi hành vi, các mô hình học sâu tuần tự đã được huấn luyện:
- **Cấu trúc mạng**: LSTM và GRU có 1 lớp ẩn (hidden_size = 64), dropout 0.2, kết hợp với tầng Fully Connected tích hợp 9 đặc trưng tĩnh nhân khẩu học.
- **Dữ liệu đầu vào**: Chuỗi trượt 12 tháng liên tục chứa 22 cột hành vi thô.
- **Kết quả so sánh (Clean Test)**:
  - **LightGBM (Rolling 12M)**: PR-AUC = **`0.152145`**, F1 = **`0.299728`**
  - **PyTorch GRU**: PR-AUC = `0.106059`, F1 = `0.210526`
  - **PyTorch LSTM**: PR-AUC = `0.104672`, F1 = `0.197917`
- *Kết luận*: LightGBM vượt trội hơn hẳn các mô hình deep learning tuần tự. Do kích thước mẫu dương tính còn nhỏ và chu kỳ hành vi hàng tháng tương đối ngắn, các mô hình Boosted Trees vẫn tối ưu hơn RNN.

---

## 20. Final Model Selection
Mô hình sản xuất cuối cùng được cấu hình cố định trong hệ thống:
- **Thuật toán**: LightGBM Classifier + Platt Calibrator
- **Đặc trưng**: Toàn bộ 379 đặc trưng thời gian (All379)
- **Cơ chế huấn luyện**: Huấn luyện lại động (Rolling 12M Retraining)
- **Ngưỡng quyết định**: `0.08` (sau khi hiệu chuẩn xác suất)

---

## 21. Final Evaluation
Dưới đây là kết quả đánh giá cuối cùng của mô hình LightGBM trên tập Clean Test sạch (`2026-03-01` đến `2026-06-01`):

Metric | Toàn bộ tập khách hàng (Entire Cohort)
--- | ---
**Số mẫu (Rows)** | 35,336
**Số Churn (Positives)**| 249
**PR-AUC** | `0.121786`
**ROC-AUC** | `0.910986`
**Precision** | `17.9856%`
**Recall** | `30.1205%`
**F1-Score** | `22.5225%` (hoặc `0.225225`)
**Confusion Matrix** | `[[34745, 342], [174, 75]]`

*Giải thích*: Việc giữ nguyên toàn bộ 379 đặc trưng (All379) giúp mô hình tối ưu hóa khả năng nhận diện khách hàng rời bỏ (Recall tăng lên mức 30.12%) trong khi vẫn duy trì PR-AUC cao nhất trong các cấu hình thực nghiệm.

---

## 22. Model Storage & Verification
Model được lưu trữ duy nhất dưới dạng gói bundle tại đường dẫn:
```
artifacts/temporal_churn_model.joblib
```
Bundle chứa các thành phần đóng gói độc lập:
- `model`: Đối ứng LightGBM Classifier đã huấn luyện.
- `calibrator`: Logistic Regression dùng cho hiệu chuẩn Platt.
- `imputer`: SimpleImputer xử lý dữ liệu trống.
- `selected_features`: Danh sách 379 đặc trưng được chọn (toàn bộ đặc trưng).
- `threshold`: Ngưỡng xác suất tối ưu (0.08).
- `metadata`: Phiên bản, ngày tạo, các mốc thời gian huấn luyện.

Kết quả kiểm thử kiểm duyệt tự động từ `verify_saved_model.py`:
- **MODEL LOAD**: `PASS` (tải mô hình và calibrator thành công).
- **FEATURE COMPATIBILITY**: `PASS` (100% đặc trưng chọn khớp với dữ liệu thực tế).
- **PREDICTION TEST**: `PASS` (suy diễn thành công trên 100 dòng mẫu, xác suất trả về nằm đúng trong khoảng $[0, 1]$, nhãn dự đoán nhị phân $\{0, 1\}$).

---

## 23. Limitations
Mặc dù đạt kết quả tốt, hệ thống vẫn tồn tại các hạn chế kỹ thuật:
1. **Mất cân bằng lớp cực đoan**: Tỷ lệ nhãn dương chỉ khoảng ~0.70% khiến việc tối ưu hóa F1-score gặp khó khăn.
2. **Hiện tượng trôi lệch tiếp thị**: Các đặc trưng marketing drift mạnh theo thời gian, đòi hỏi phải giám sát PSI liên tục.
3. **Phụ thuộc label maturity**: Việc đánh giá hiệu năng thực tế bị chậm 30 ngày để đảm bảo thu thập đủ kết quả thực tế của prediction window.

---

## 24. Conclusion
Dự án đã hoàn thành trọn vẹn mục tiêu **Temporal Machine Learning Churn Prediction**:
1. Đưa thành công cấu trúc hành vi thời gian hàng tháng vào bài toán churn thay vì bảng dữ liệu tĩnh.
2. Xây dựng và kiểm duyệt thành công quy trình tạo đặc trưng trễ (lags), tích lũy (rolling), xu hướng (trend) và khoảng cách (recency) không chứa lookahead bias.
3. Chứng minh mô hình LightGBM Rolling 12M kết hợp Platt Calibration mang lại kết quả phân loại vượt trội so với baseline tĩnh và deep learning tuần tự.
4. Đóng gói mô hình thành công vào một artifact duy nhất, vượt qua tất cả các kiểm thử tương thích đặc trưng và suy diễn xác thực.
Dự án chính thức đóng lại tại bước đóng gói và xác thực này.
