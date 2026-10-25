# Expanded-feature electricity forecasting results

These are actual seed-42 CPU results. The target remains the next 24 hourly demand values, using the preceding 168 hours. Test origins and target values match the original experiment exactly.

| Model | Demand-only RMSE (kW) | Expanded RMSE (kW) | RMSE reduction (%) |
| --- | --- | --- | --- |
| CatBoost | 0.5859 | 0.5656 | 3.4725 |
| TCN | 0.6039 | 0.5782 | 4.2544 |
| GRU | 0.5876 | 0.5837 | 0.6702 |
| MLP | 0.5886 | 0.5849 | 0.6184 |
| LSTM | 0.5864 | 0.5900 | -0.6233 |
| N-BEATS-style | 0.5882 | 0.5907 | -0.4284 |
| Linear | 0.5916 | 0.6017 | -1.7123 |
| Patch Transformer | 0.5935 | 0.6053 | -1.9939 |
| Transformer | 0.5957 | 0.6231 | -4.5970 |
| CNN1D | 0.6267 | 0.6636 | -5.8892 |

The original neural results are saved reference measurements; the expanded neural fits and both CatBoost baselines were executed for this update. Neural models retain the original training settings and widths, with larger input projections and shared context. CatBoost uses a separate fixed baseline budget. A single seed and feature bundle do not isolate individual feature effects or establish a general architecture ranking.
