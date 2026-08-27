# Phân Tích Chi Tiết: Baseline_LogisticRegression_v2.ipynb

Tài liệu này phân tích cấu trúc, cơ sở lý thuyết, ý nghĩa toán học và logic xử lý của từng khối mã trong file notebook **`Baseline_LogisticRegression_v2.ipynb`**.

---

## 1. Mục Đích & Ý Nghĩa Kiến Trúc

### 1.1. Thiết lập điểm sàn hiệu năng (Benchmark Baseline)
Trong quy trình phát triển mô hình học máy, việc xây dựng một mô hình cơ sở đơn giản (Linear/Logistic Baseline) là bắt buộc. Nó đóng vai trò là "mức chuẩn tối thiểu" để đo lường xem các kiến trúc phi tuyến phức tạp (LightGBM, XGBoost, Neural Networks) sau này có thực sự tạo ra giá trị gia tăng tương xứng với chi phí tính toán hay không.

### 1.2. Cơ chế phát hiện rò rỉ dữ liệu (Data Leakage Detection)
Data Leakage là hiện tượng dữ liệu đầu vào vô tình chứa các thông tin thuộc về tương lai (sau mốc snapshot) hoặc chứa chính các thành phần cấu tạo nên nhãn mục tiêu.
- **Quy luật kiểm thử:** Logistic Regression là mô hình tuyến tính với không gian phân tách phẳng (hyperplane). Với một bài toán thực tế phức tạp như hành vi rời bỏ dịch vụ của khách hàng, Logistic Regression thông thường chỉ đạt $ROC-AUC \approx 0.80 - 0.90$.
- **Dấu hiệu cảnh báo:** Nếu một mô hình tuyến tính đạt $ROC-AUC > 0.99$ hoặc $PR-AUC \approx 1.0$, điều này gần như **chắc chắn 100%** là do Data Leakage (chứa biến "tiết lộ tương lai" như ngày đóng tài khoản, cờ hủy gói trong 30 ngày tới).

---

## 2. Phân Tích Chi Tiết Từng Phần (Sections / Cells)

---

### Section 1: Thiết Lập Môi Trường & Thư Viện
- **Mục tiêu phân tích:** Đảm bảo môi trường thực nghiệm có tính tái lập (Reproducibility).
- **Cơ chế:**
  - Khởi tạo seed ngẫu nhiên cố định (`SEED = 42`) cho bộ sinh số ngẫu nhiên của `numpy`.
  - Cấu hình chuẩn hóa xuất nhập `utf-8` để xử lý chính xác các chuỗi ký tự tiếng Việt trên môi trường hệ điều hành Windows.

---

### Section 2: Đọc Dữ Liệu & Phân Tích Phân Phối Nhãn (Dataset v2)
- **Tệp dữ liệu:** `output/churn_temporal_dataset_v2.parquet` gồm **185,160 dòng** và **387 cột**.
- **Phân tích nhãn mục tiêu (`churn_next_30d`):**
  - Khách hàng không rời bỏ (`0`): **141,617 mẫu (~76.48%)**.
  - Khách hàng rời bỏ (`1`): **43,543 mẫu (~23.52%)**.
- **Phân tích cấu trúc quy tắc Churn v2:**
  - Nhãn Churn v2 được tổng hợp từ 3 điều kiện kinh doanh thực tế:
    1. `rule1_closed`: Khách hàng thực hiện đóng tài khoản hoàn toàn trong vòng 30 ngày tới.
    2. `rule2_downgrade_to_free_inactive`: Khách hàng hạ từ gói trả phí (Plus/Pro) xuống gói Free và hoàn toàn không có tương tác/hoạt động trong 30 ngày tới.
    3. `rule3_free_at_snapshot_inactive`: Khách hàng đang ở gói Free sẵn và hoàn toàn không có tương tác/hoạt động trong 30 ngày tới.

---

### Section 3: Xác Định Tập Đặc Trưng & Lọc Chống Rò Rỉ Nhãn
- **Danh sách loại trừ bắt buộc (`META_COLS`):**
  - Nhóm định danh: `customer_id`, `snapshot_date`.
  - Nhóm nhãn và thành phần nhãn: `churn_next_30d`, `rule1_closed`, `rule2_downgrade_to_free_inactive`, `rule3_free_at_snapshot_inactive`.
  - Nhóm thông tin hậu kỳ: `churn_reason`, `tier_at_snapshot`, `closed_date`, `label_complete`.
- **Cơ chế lọc tự động:** Loại bỏ tất cả các cột có tiền tố `future_` hoặc hậu tố `_future` để ngăn chặn tuyệt đối việc sử dụng thông tin sau mốc thời gian snapshot.
- **Kết quả:** Giữ lại chính xác **379 đặc trưng thời gian** (RFM, Recency, Lag 1-3 tháng, Rolling 1-6 tháng, Trend/Slope).

---

### Section 4: Phân Chia Dữ Liệu Dòng Thời Gian (Chronological Split)
- **Phân tích phương pháp:** Không sử dụng phép chia ngẫu nhiên (Random K-Fold) vì trong dữ liệu chuỗi thời gian, việc trộn lẫn các mốc thời gian sẽ gây rò rỉ xu hướng vĩ mô từ tương lai vào quá khứ.
- **Phân bổ 3 tập dữ liệu độc lập:**
  1. **Tập Huấn luyện (Train):** Từ `2024-09-01` đến `2025-08-01` (12 tháng snapshot) — Dùng để fit mô hình và bộ tiền xử lý.
  2. **Tập Hiệu chuẩn (Validation):** Từ `2025-09-01` đến `2026-02-01` (6 tháng snapshot) — Dùng để dò tìm ngưỡng phân loại tối ưu (Threshold Tuning).
  3. **Tập Kiểm thử (Test):** Từ `2026-03-01` đến `2026-06-01` (4 tháng snapshot hoàn toàn mới) — Dùng để đánh giá độc lập và phát hiện rò rỉ dữ liệu.

---

### Section 5: Xây Dựng Pipeline Tiền Xử Lý & Huấn Luyện
- **`SimpleImputer(strategy='median')`:**
  - *Ý nghĩa:* Điền các giá trị thiếu bằng giá trị trung vị (Median) được tính toán **chỉ trên tập Train**, sau đó áp dụng phép biến đổi này lên Validation và Test để chống rò rỉ phân phối (Distribution Leakage).
- **`StandardScaler()`:**
  - *Ý nghĩa:* Chuẩn hóa từng đặc trưng về phân phối chuẩn $z = \frac{x - \mu}{\sigma}$. Phép biến đổi này là bắt buộc đối với Logistic Regression để gradient hội tụ nhanh và giúp các trọng số hồi quy $w_i$ có thể so sánh trực tiếp độ quan trọng với nhau.
- **`LogisticRegression(penalty='l2', C=1.0)`:**
  - *Ý nghĩa:* Hàm mất mát Log-loss kết hợp với thành phần phạt $L_2$ (Ridge Regularization):
    $$\mathcal{L}(w) = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \ln(p_i) + (1-y_i)\ln(1-p_i) \right] + \frac{1}{2C} \|w\|_2^2$$
  - Giữ $C=1.0$ tiêu chuẩn để đo lường độ phân tách tự nhiên của dữ liệu.

---

### Section 6: Đánh Giá Hiệu Năng & Logic Phát Hiện Data Leakage
- **Hệ thống chỉ số đánh giá:**
  - **PR-AUC (Average Precision Score):** Diện tích dưới đường cong Precision-Recall — chỉ số quan trọng nhất cho dữ liệu mất cân bằng lớp.
  - **ROC-AUC:** Đo lường khả năng phân tách tổng thể giữa 2 lớp $0$ và $1$.
  - **F1-Score:** Trung bình điều hòa giữa Precision và Recall.
  - **Brier Score Loss:** Đo lường sai số bình phương trung bình giữa xác suất dự đoán $p_i$ và nhãn thực tế $y_i$ (giá trị càng gần 0 càng tốt).
- **Thang đo rủi ro Data Leakage:**
  - $\mathbf{AUC > 0.99}$: **Đỏ (Nghiêm trọng)** — Có biến rò rỉ thông tin tương lai trực tiếp.
  - $\mathbf{0.95 \le AUC \le 0.99}$: **Vàng (Cảnh báo)** — Cần rà soát các biến có trọng số áp đảo.
  - $\mathbf{0.80 \le AUC < 0.95}$: **Xanh (Bình thường)** — Bộ đặc trưng lành mạnh, mô hình học được quy luật thực tế.
  - $\mathbf{AUC < 0.80}$: **Trắng (Yếu)** — Không có rò rỉ, cần cải thiện Feature Engineering.

---

### Section 7: Phân Tích Đường Cong ROC & Precision-Recall
- **ROC Curve:** So sánh tỷ lệ bắt đúng (True Positive Rate) với tỷ lệ báo động giả (False Positive Rate) trên toàn bộ dải ngưỡng xác suất.
- **PR Curve:** Phản ánh độ chính xác thực tế khi mô hình dự báo một khách hàng sẽ rời bỏ dịch vụ so với tỷ lệ nền (Baseline Rate $\approx 23.5\%$).

---

### Section 8 & 9: Phân Tích Trọng Số Hồi Quy (Coefficients Analysis)
- **Cơ sở toán học:** Trong Logistic Regression, xác suất được tính bằng:
  $$P(Y=1|X) = \sigma\left(w_0 + \sum_{j=1}^M w_j x_j\right)$$
  - $w_j > 0$: Đặc trưng $x_j$ càng tăng thì nguy cơ khách hàng rời bỏ càng cao (ví dụ: `days_since_last_activity`, `payment_failure_count`).
  - $w_j < 0$: Đặc trưng $x_j$ càng tăng thì khách hàng càng gắn kết, giảm nguy cơ rời bỏ (ví dụ: `spend_rolling_sum_6m`, `usage_rolling_mean_3m`).
- **Phát hiện bất thường:** Nếu tồn tại đặc trưng có $|w_j| > 3.0$, đặc trưng đó đang chi phối toàn bộ hàm sigmoid $\rightarrow$ cần kiểm tra lại công thức tính toán của đặc trưng đó.

---

### Section 10: Tối Ưu Hóa Ngưỡng Phân Loại (Threshold Tuning)
- Mặc định ngưỡng phân loại là $0.5$. Tuy nhiên, với bài toán mất cân bằng lớp và chi phí bỏ sót khách hàng rời bỏ (False Negative) rất lớn, ngưỡng $0.5$ thường không tối ưu.
- Thực hiện quét 99 giá trị ngưỡng trên tập **Validation** để tìm ngưỡng tối đa hóa điểm $F_1$, sau đó cố định ngưỡng này để đánh giá ma trận nhầm lẫn (Confusion Matrix: TP, FP, TN, FN) trên tập **Test**.

---

### Section 11: Phân Tích Khoảng Cách Hiệu Năng (Gap Analysis)
- Tính toán độ suy giảm:
  $$\Delta AUC = AUC_{Train} - AUC_{Test}$$
- **Phân tích:**
  - Nếu $\Delta AUC \approx 0$ và cả hai đều $> 0.99 \rightarrow$ Data Leakage mang tính hệ thống trên toàn bộ dataset.
  - Nếu $\Delta AUC > 0.10 \rightarrow$ Mô hình bị Overfitting nặng do quá nhiều biến nhiễu.
  - Nếu $\Delta AUC \le 0.05$ và $AUC_{Test} \in [0.80, 0.95] \rightarrow$ Mô hình tổng quát hóa tốt.

---

### Section 12: Đánh Giá Độ Ổn Định Theo Từng Snapshot Tháng
- Tính toán ROC-AUC riêng biệt cho từng tháng snapshot trong toàn bộ chuỗi lịch sử (2023 - 2026).
- **Mục tiêu phân tích:** Kiểm tra xem chất lượng dự đoán có bị suy thoái theo thời gian (Concept Drift) hoặc có xuất hiện tháng nào có AUC vọt lên $1.0$ bất thường hay không.

---

### Section 13: Bảng Tổng Kết Phán Quyết (Final Verdict)
- Tổng hợp toàn bộ các bằng chứng định lượng để đưa ra kết luận dứt khoát: Tập dữ liệu có sạch để tiếp tục đầu tư xây dựng các mô hình phức tạp hơn (Ensemble, Deep Learning) hay phải quay lại bước Feature Engineering để sửa lỗi rò rỉ.
