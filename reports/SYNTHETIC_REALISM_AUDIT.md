# Synthetic Realism Audit Report

Comparing statistical properties of 6 common signals between Synthetic (55,200 obs) and Real Fitbit (4,159 obs) cohorts.

| Feature              | Real Mean ± SD    |   Real Skew |   Real AR(1) | Synth Mean ± SD   |   Synth Skew |   Synth AR(1) |
|:---------------------|:------------------|------------:|-------------:|:------------------|-------------:|--------------:|
| resting_hr_bpm       | 66.10 ± 7.29      |      -0.271 |        0.842 | 64.53 ± 8.18      |        0.071 |        -0.006 |
| avg_hr_day_bpm       | 79.08 ± 8.89      |       0.654 |        0.277 | 87.11 ± 10.35     |        0.05  |         0.014 |
| steps                | 8448.83 ± 5498.07 |       1.049 |        0.196 | 9282.01 ± 4016.04 |        0.536 |         0.022 |
| distance_km          | 5.96 ± 3.96       |       1.101 |        0.188 | 7.43 ± 3.27       |        0.59  |         0.022 |
| calories_kcal        | 2377.23 ± 717.70  |       0.901 |        0.245 | 2074.19 ± 259.12  |        0.099 |         0.013 |
| sleep_duration_hours | 7.43 ± 1.96       |      -0.427 |        0.134 | 6.99 ± 0.84       |        0.01  |         0.022 |

**Correlation Matrix Frobenius Norm Difference**: $\|R_{synth} - R_{real}\|_F = 1.0309$
