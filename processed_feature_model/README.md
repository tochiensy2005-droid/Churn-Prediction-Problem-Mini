# Mô Hình Churn Prediction Trên Tập 41 Đặc Trưng Đã Qua Xử Lý

Thư mục này chứa toàn bộ pipeline mô hình hóa bằng các **file Python độc lập** (không dùng Jupyter Notebook) để huấn luyện, đánh giá và giải thích mô hình trên tệp dữ liệu **`churn_feature_dataset_processed.csv`** (166,084 dòng, 41 đặc trưng số).

---

## 1. Cấu Trúc Thư Mục

```
processed_feature_model/
│
├── run_pipeline.py              # Bộ điều phối thực thi toàn bộ pipeline
│
├── scripts/                     # 5 bước thực thi Python độc lập
│   ├── step0_check_env.py       # Bước 0: Kiểm tra môi trường, thư viện và dữ liệu local
│   ├── step1_prepare_data.py    # Bước 1: Kiểm tra chất lượng & phân chia dòng thời gian
│   ├── step2_train_model.py     # Bước 2: Huấn luyện (LightGBM/XGBoost) + Platt Scaling
│   ├── step3_evaluate_model.py  # Bước 3: Đánh giá trên tập Test độc lập & Sanity Checks
│   └── step4_feature_report.py  # Bước 4: Báo cáo Feature Importance & Tổng kết output
│
├── artifacts/                   # Chứa model bundle đã đóng gói (.joblib)
│   ├── processed_churn_lightgbm_model.joblib
│   └── processed_churn_xgboost_model.joblib
│
└── output/                      # Báo cáo, bảng phân phối và bảng chỉ số đầu ra
    ├── snapshot_distribution.csv
    ├── feature_importance_lightgbm.csv
    ├── feature_importance_xgboost.csv
    ├── eval_metrics_lightgbm.csv
    └── eval_metrics_xgboost.csv
```

---

## 2. Danh Sách 41 Đặc Trưng Đầu Vào

Tập dữ liệu bao gồm 4 nhóm đặc trưng:

1. **Hành vi & Tương tác gần (Recency & Activity):**
   - `days_since_last_activity`, `days_since_last_activity_lag1m`, `days_since_last_login`, `days_since_last_usage_event`
   - `total_active_days_30d`, `total_active_days_60d`, `total_active_days_90d`
   - `activity_slope_3m`, `is_declining_engagement`, `reactivation_flag`

2. **Mức độ sử dụng dịch vụ (Usage):**
   - `num_usage_events_30d`, `num_usage_events_30d_lag1m`, `num_usage_events_60d`, `num_usage_events_roll3m_sum`
   - `avg_session_duration_30d`, `avg_session_duration_roll3m_mean`, `total_session_time_30d`
   - `session_duration_trend`, `usage_trend_30d`, `event_type_diversity_30d`

3. **Giao dịch & Đơn hàng (Orders & Payments):**
   - `orders_last_30d`, `orders_last_90d`, `orders_roll3m_sum`
   - `avg_spend_to_date_per_month`
   - `payments_success_rate`, `payments_success_rate_missing`

4. **Khách hàng, Gói dịch vụ & Hỗ trợ (Customer, Subscription & Support):**
   - `age`, `gender`, `city`, `region`, `tenure_days`
   - `subscription_tier`, `is_paid_tier`, `auto_renew`
   - `num_tickets_90d`, `has_unresolved_ticket`, `avg_csat_score`, `avg_csat_score_missing`
   - `open_rate_30d`, `has_marketing_click_30d`

---

## 3. Cách Sử Dụng Dòng Lệnh

```powershell
# Chạy toàn bộ pipeline từ bước 0 đến bước 4 (LightGBM):
python processed_feature_model/run_pipeline.py

# Chạy với thuật toán XGBoost:
python processed_feature_model/run_pipeline.py --model xgboost

# Chỉ chạy bước kiểm tra môi trường:
python processed_feature_model/run_pipeline.py --steps 0

# Huấn luyện và đánh giá lại (bước 2, 3, 4):
python processed_feature_model/run_pipeline.py --steps 2 3 4
```
