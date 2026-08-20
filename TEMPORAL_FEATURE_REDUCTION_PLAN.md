# Temporal Feature Reduction Plan (V2 - Approved & Implemented)

Bản kế hoạch này trình bày thiết kế tối ưu hóa (refactor) đặc trưng thời gian (Temporal Feature Engineering) và tích hợp **Pipeline Tiền xử lý (Preprocessing Pipeline)** chuẩn học máy. Hệ thống đã được triển khai chạy song song và vượt qua 100% các bước kiểm thử kiểm duyệt.

---

## 1. Audit & Analysis of Current Feature Importance
Dựa trên tệp xếp hạng tầm quan trọng đặc trưng hiện tại ([final_feature_importance.csv](file:///d:/Intern%20Data/output/modeling/tuning/final_feature_importance.csv)), tín hiệu dự báo churn tập trung mạnh vào các nhóm sau:
*   **Đặc trưng hoạt động (`active_days`, `usage`)**: Chiếm trọn 3 vị trí dẫn đầu (`active_days_rolling_mean_6m`, `active_days_rolling_mean_3m`, `active_days_rolling_sum_6m`).
*   **Đặc trưng khoảng cách (`days_since_last_payment`, `days_since_last_support_ticket`)**: Đóng vai trò cực kỳ quan trọng (nằm trong Top 10).
*   **Đặc trưng chi tiêu (`spend`) và biến động nâng/hạ cấp (`subscription_change`, `upgrade`, `downgrade`)**: Chiếm ưu thế lớn ở phân khúc tầm trung.
*   **Đặc trưng thanh toán (`payment_success_rolling_mean_6m`)**: Đóng góp tín hiệu ổn định.
*   **Đặc trưng tiếp thị (`marketing_interaction_slope_3m`)**: Có đóng góp nhưng có độ dư thừa (redundancy) rất cao do sinh quá nhiều biến thể tương tự nhau.

---

## 2. Temporal Features Reduction Mapping

| Base Variable | Business Meaning | Current Lag Features | Current Rolling Features | Current Importance | Keep / Remove | Recommended Temporal Features | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **`active_days`** | Tần suất hoạt động trong tháng | lag 1, 2, 3 | 1m, 3m, 6m (sum, mean, std, min, max) | Rất Cao (Rank 1, 2, 3) | **Keep** | lag 1, 2, 3;<br>3m mean, 6m mean, 3m min;<br>3m slope | Trụ cột hành vi chính xác định churn. |
| **`usage`** | Khối lượng sử dụng sản phẩm | lag 1, 2, 3 | 1m, 3m, 6m (sum, mean, std, min, max) | Cao (Rank 4, 5) | **Keep** | lag 1, 2, 3;<br>3m mean, 6m mean, 3m std;<br>1m pct_change | Phản ánh mức độ tương tác trực tiếp với sản phẩm. |
| **`spend`** | Giá trị chi tiêu của đơn hàng | lag 1, 2, 3 | 1m, 3m, 6m (sum, mean, std, min, max) | Cao (Rank 10, 17, 18) | **Keep** | lag 1, 2, 3;<br>3m mean, 6m mean, 3m std, 3m min;<br>1m change, 1m pct_change, 3m slope | Khách hàng giảm chi tiêu là tín hiệu rời bỏ rõ ràng. |
| **`orders`** | Tổng số đơn đặt hàng | lag 1, 2, 3 | 1m, 3m, 6m (sum, mean, std, min, max) | Trung Bình (Rank 15, 32) | **Keep** | lag 1, 2, 3;<br>3m mean, 6m mean;<br>1m pct_change | Tần suất mua hàng giảm liên quan trực tiếp đến churn. |
| **`completed_orders`**| Số đơn hàng hoàn thành thành công| lag 1, 2, 3 | 1m, 3m, 6m (sum, mean, std, min, max) | Trung Bình (Rank 13, 55) | **Keep** | lag 1, 2, 3;<br>3m mean, 6m std | Đánh giá giao dịch thành công thực tế của khách hàng. |
| **`payment_count`** | Tổng số lượt thanh toán | lag 1, 2, 3 | 1m, 3m, 6m (sum, mean, std, min, max) | Trung Bình (Rank 38, 49) | **Keep** | lag 1;<br>6m mean, 6m min | Số lượt thanh toán giảm phản ánh chu kỳ sử dụng sắp kết thúc. |
| **`payment_success`**| Số lượt thanh toán thành công | lag 1, 2, 3 | 1m, 3m, 6m (sum, mean, std, min, max) | Trung Bình (Rank 17, 53) | **Keep** | 6m mean, 3m sum, 6m sum | Đảm bảo tính liên tục của gói dịch vụ. |
| **`payment_failure`**| Số lượt thanh toán thất bại | lag 1, 2, 3 | 1m, 3m, 6m (sum, mean, std, min, max) | Thấp (Rank 76) | **Keep** | 6m max | Thanh toán thất bại nhiều lần dễ dẫn đến churn tự động. |
| **`payment_success_rate`**| Tỷ lệ thanh toán thành công | lag 1, 2, 3 | 1m, 3m, 6m (sum, mean, std, min, max) | Trung Bình (Rank 24, 33, 37) | **Keep** | lag 3;<br>3m mean, 3m min, 6m std;<br>3m slope (count), 3m slope (rate) | Tỷ lệ lỗi thanh toán cao là rủi ro churn kỹ thuật. |
| **`support_ticket`** | Số khiếu nại hỗ trợ khách hàng | lag 1, 2, 3 | 1m, 3m, 6m (sum, mean, std, min, max) | Trung Bình (Rank 20, 58) | **Keep** | lag 1;<br>3m sum, 6m std | Phản ánh trực tiếp mức độ không hài lòng của khách hàng. |
| **`csat`** | Điểm số đánh giá dịch vụ | lag 1, 2, 3 | 1m, 3m, 6m (sum, mean, std, min, max) | Thấp-Trung Bình (Rank 19) | **Keep** | 3m mean, 6m min | Điểm CSAT thấp báo hiệu chất lượng dịch vụ đi xuống. |
| **`marketing_interaction`**| Số lượt tương tác tiếp thị | lag 1, 2, 3 | 1m, 3m, 6m (sum, mean, std, min, max) | Trung Bình (Rank 14, 21, 22, 25) | **Keep** | lag 1;<br>6m mean, 3m std;<br>3m slope | Phản ánh mức độ phản hồi của khách hàng với chiến dịch. |
| **`marketing_click_rate`**| Tỷ lệ nhấp chuột tiếp thị | lag 1, 2, 3 | 1m, 3m, 6m (sum, mean, std, min, max) | Trung Bình (Rank 30, 31, 45) | **Keep** | 3m mean, 6m mean, 6m min | Tỷ lệ tương tác giảm thể hiện khách hàng thờ ơ với thương hiệu. |
| **`downgrade`** | Số lần hạ cấp gói cước | lag 1, 2, 3 | 1m, 3m, 6m (sum, mean, std, min, max) | Cao-Trung Bình (Rank 36) | **Keep** | lag 1;<br>3m count, 6m count | Trực tiếp liên quan đến định nghĩa churn của business. |
| **`upgrade`** | Số lần nâng cấp gói cước | lag 1, 2, 3 | 1m, 3m, 6m (sum, mean, std, min, max) | Trung Bình (Rank 12) | **Keep** | 6m count | Thể hiện mức độ gia tăng gắn kết của khách hàng. |
| **`subscription_change`**| Số lần thay đổi gói cước chung | lag 1, 2, 3 | 1m, 3m, 6m (sum, mean, std, min, max) | Cao (Rank 9, 11) | **Keep** | 3m count, 6m std | Thay đổi liên tục thể hiện gói cước hiện tại không ổn định. |
| **`marketing_click`**| Số click tiếp thị | lag 1, 2, 3 | 1m, 3m, 6m (sum, mean, std, min, max) | Thấp (Rank 73) | **Remove** | None | Đã được thay thế hoàn toàn bằng `marketing_click_rate`. |

---

## 3. Preprocessing Logic (Tree-based & Generic ML)

Chúng tôi đã đóng gói quy trình xử lý thành module `src/preprocessing/build_preprocessor.py` hỗ trợ hai chế độ độc lập:

| Đặc tính Tiền xử lý | LightGBM Mode (Pipeline chính) | Generic ML Mode (Optional Utility) |
| --- | --- | --- |
| **Xử lý giá trị trống (Numeric)** | Giữ nguyên NaN (Native handling) | `SimpleImputer(strategy="median")` |
| **Mã hóa Categorical** | `OneHotEncoder(handle_unknown="ignore")` | `OneHotEncoder(handle_unknown="ignore")` |
| **Chuẩn hóa tỷ lệ (Scaling)** | Không áp dụng | `StandardScaler()` |
| **Làm sạch dữ liệu** | Gom cụm `HaNoi` $\rightarrow$ `Ha Noi` | Gom cụm `HaNoi` $\rightarrow$ `Ha Noi` |
| **Chuyển đổi phân phối** | Giữ nguyên thô | Áp dụng `log1p` cho Spend/Usage |

---

## 4. Final Proposed Feature List (76 Features)

### Group 1: Engagement & Usage (14 features)
1.  `usage_lag_1`
2.  `usage_lag_2`
3.  `usage_lag_3`
4.  `active_days_lag_1`
5.  `active_days_lag_2`
6.  `active_days_lag_3`
7.  `usage_rolling_mean_3m`
8.  `usage_rolling_mean_6m`
9.  `usage_rolling_std_3m`
10. `active_days_rolling_mean_3m`
11. `active_days_rolling_mean_6m`
12. `active_days_rolling_min_3m`
13. `active_days_slope_3m`
14. `usage_pct_change_1m`

### Group 2: Spend & Orders (21 features)
15. `spend_lag_1`
16. `spend_lag_2`
17. `spend_lag_3`
18. `spend_rolling_mean_3m`
19. `spend_rolling_mean_6m`
20. `spend_rolling_std_3m`
21. `spend_rolling_min_3m`
22. `spend_change_1m`
23. `spend_pct_change_1m`
24. `spend_slope_3m`
25. `orders_lag_1`
26. `orders_lag_2`
27. `orders_lag_3`
28. `completed_orders_lag_1`
29. `completed_orders_lag_2`
30. `completed_orders_lag_3`
31. `orders_rolling_mean_3m`
32. `orders_rolling_mean_6m`
33. `completed_orders_rolling_mean_3m`
34. `completed_orders_rolling_std_6m`
35. `orders_pct_change_1m`

### Group 3: Payments (13 features)
36. `payment_count_lag_1`
37. `payment_count_rolling_mean_6m`
38. `payment_count_rolling_min_6m`
39. `payment_success_rolling_mean_6m`
40. `payment_success_rolling_sum_3m`
41. `payment_success_rolling_sum_6m`
42. `payment_failure_rolling_max_6m`
43. `payment_success_rate_rolling_mean_3m`
44. `payment_success_rate_rolling_min_3m`
45. `payment_success_rate_rolling_std_6m`
46. `payment_success_rate_lag_3`
47. `payment_success_count_slope_3m` (Đổi tên từ `payment_success_slope_3m`)
48. `payment_success_rate_slope_3m` (Thêm mới)

### Group 4: Support & Satisfaction (5 features)
49. `support_ticket_lag_1`
50. `support_ticket_rolling_sum_3m`
51. `support_ticket_rolling_std_6m`
52. `csat_rolling_mean_3m`
53. `csat_rolling_min_6m` (Thay thế cho `csat_rolling_max_6m`)

### Group 5: Subscription & Upgrades/Downgrades (6 features)
54. `downgrade_lag_1`
55. `downgrade_count_3m` (Thay thế cho `downgrade_rolling_mean_6m`)
56. `downgrade_count_6m` (Thêm mới)
57. `upgrade_count_6m` (Thay thế cho `upgrade_rolling_mean_6m`)
58. `subscription_change_count_3m` (Thay thế cho `subscription_change_rolling_mean_3m`)
59. `subscription_change_rolling_std_6m`

### Group 6: Marketing (7 features)
60. `marketing_interaction_lag_1`
61. `marketing_interaction_rolling_mean_6m`
62. `marketing_interaction_rolling_std_3m`
63. `marketing_interaction_slope_3m`
64. `marketing_click_rate_rolling_mean_3m`
65. `marketing_click_rate_rolling_mean_6m`
66. `marketing_click_rate_rolling_min_6m`

### Group 7: Recency & Static Features (10 features)
67. `days_since_last_usage`
68. `days_since_last_order`
69. `days_since_last_payment`
70. `days_since_last_support_ticket`
71. `days_since_last_downgrade`
72. `age`
73. `gender`
74. `city`
75. `region`
76. `customer_tenure`
