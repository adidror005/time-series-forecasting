# Executed comparison

All scores below were measured in the supplied notebook execution.

| Model | Test RMSE (kW) | Observed-only RMSE (kW) | Parameters |
|---|---:|---:|---:|
| LSTM | 0.586379 | 0.593479 | 18,712 |
| GRU | 0.587642 | 0.594803 | 14,424 |
| N-BEATS-style | 0.588154 | 0.595787 | 672,576 |
| MLP | 0.588570 | 0.596071 | 79,256 |
| Linear | 0.591601 | 0.598057 | 4,056 |
| Patch Transformer | 0.593467 | 0.602240 | 70,360 |
| Transformer | 0.595687 | 0.602044 | 79,512 |
| TCN | 0.603921 | 0.612295 | 38,872 |
| CNN1D | 0.626667 | 0.633757 | 32,600 |
| Yesterday | 0.749631 | 0.761792 | 0 |
| Last week | 0.802534 | 0.806139 | 0 |
| Last observed value | 0.917863 | 0.935385 | 0 |

A single seed and chronological split do not establish a stable architectural ranking.
The original article supplied no training logs or weights; these are fresh results.
The TCN has a 31-hour receptive field. All inputs contain 168 hourly observations.
Data: UCI Individual Household Electric Power Consumption, DOI 10.24432/C58K54 (CC BY 4.0).
