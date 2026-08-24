# Domain-Shift Analysis Report: Synthetic vs. Real-World Fitbit

Quantifies distribution differences between synthetic training data and real Fitbit wearable data on the harmonized primary feature space:

| feature              |   synthetic_mean |   synthetic_std |   real_mean |   real_std |   cohens_d |   wasserstein_distance | shift_severity   |
|:---------------------|-----------------:|----------------:|------------:|-----------:|-----------:|-----------------------:|:-----------------|
| resting_hr_bpm       |           64.53  |           8.18  |      66.09  |      7.304 |     -0.192 |                  1.772 | Low              |
| avg_hr_day_bpm       |           87.105 |          10.347 |      79.023 |      8.865 |      0.788 |                  8.143 | Moderate         |
| steps                |         9282.01  |        4016.07  |    8448.07  |   5503.53  |      0.202 |               1511.07  | Low              |
| distance_km          |            7.429 |           3.267 |       5.96  |      3.961 |      0.443 |                  1.601 | Low              |
| calories_kcal        |         2074.19  |         259.12  |    2375.67  |    718.635 |     -0.962 |                393.393 | High             |
| sleep_duration_hours |            6.99  |           0.842 |       7.474 |      1.908 |     -0.507 |                  0.914 | Moderate         |
| severity_score       |            4.992 |           1.679 |       5.261 |      2.095 |     -0.157 |                  0.367 | Low              |