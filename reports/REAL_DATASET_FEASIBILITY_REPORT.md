# Real Fitbit Dataset Feasibility & Scientific Authenticity Report

**Project**: Wearable Health Unsupervised Latent State Discovery  
**Evaluation Scope**: Scientific Authenticity, Data Quality, Feature Harmonization, and Methodology Compatibility Assessment  
**Target Dataset**: `data/daily_fitbit_sema_df_unprocessed.csv`  
**Baseline Dataset**: `data/wearables_health_6mo_daily.csv` (Synthetic)  
**Primary Objective**: **MAXIMUM SCIENTIFIC AUTHENTICITY AND DEFENSIBILITY (NOT ARTIFICIAL 100% RETENTION)**  
**Status**: FEASIBILITY STUDY RE-EVALUATION COMPLETE — NO RAW DATA ALTERED / NO MODEL RETRAINED  

---

## 1. Executive Summary

This revised feasibility study re-evaluates the newly introduced real-world Fitbit dataset (`daily_fitbit_sema_df_unprocessed.csv`) with **scientific authenticity as the primary objective**. 

Key Scientific Findings & Methodological Pivot:
1. **Rejection of Artificial 100% Retention**: Mass cohort imputation for highly incomplete signals (SpO2 at 82.86% missing and HRV RMSSD at 66.60% missing) manufactures artificial data points that degrade scientific integrity for IEEE publication. Mass filling is strictly rejected.
2. **Authentic Dual-Tier Architecture (Scenario 4 - RECOMMENDED)**:
   - **Tier 1 (Primary Model Space - 4,159 rows / 69 users [97.18% cohort coverage])**: Uses high-fidelity core wearable features (`resting_hr_bpm`, `avg_hr_day_bpm`, `steps`, `distance_km`, `calories_kcal`, `sleep_duration_hours`) with a strict $\le 2$-day causal within-user forward fill limit. Zero cohort mean filling.
   - **Tier 2 (Nocturnal Sub-Cohort Analysis - 1,066 rows / 23 users [32.39% cohort coverage])**: Evaluates the full 8-feature space (including SpO2 and HRV RMSSD) strictly on authentic days/users where nocturnal sensor measurements actually exist.
3. **Data Retention & Validity Quantified**:
   - **Strict Complete Case (All 8 Features)**: Yields **1,066 authentic rows (14.39%) across 23 users (32.39%)**.
   - **Authentic Core Complete Case (6 Core Features, Zero Imputation)**: Yields **3,498 authentic rows (47.21%) across 69 users (97.18%)**.
   - **Conservative Causal Core (6 Core Features, ffill limit 2d)**: Yields **4,159 authentic rows (56.13%) across 69 users (97.18%)**.
4. **Feasibility Assessment**: The real Fitbit dataset is **FULLY GO** under the Authentic Dual-Tier Architecture.

---

## 2. Existing Project Methodology

The existing repository implements an unsupervised longitudinal physiological state-discovery framework:
- **Personalized Baselines**: Calculates causal rolling baselines ($\mu_{i,t}$) over a 7-day window ($W=7$) per user $i$ with a mandatory warm-up period ($W_{\text{warmup}}=7$ days).
- **Personalized Deviations**: Computes raw deviations ($d_{i,t} = x_{i,t} - \mu_{i,t}$) and z-normalized deviations ($z_{i,t} = d_{i,t} / \sigma_{i,t}$) relative to past history.
- **Dynamic Slopes**: Computes causal 7-day rolling slopes ($\beta_{7\text{d}}$) using first-order polynomial regression on past window observations.
- **Composite Scores**: Derives `severity_score` ($\sum |z_{i,t}|$) and `trend_score` ($\sum |\beta_{7\text{d}}|$) for anomaly detection and latent state transitions.
- **Unsupervised Modeling**: Fits Gaussian Mixture Models (GMM) and K-Means for latent cluster state discovery, followed by Hidden Markov Models (HMM) for temporal state transitions (Recovery, Baseline, Strain).
- **Causal Safeguards**: Strictly avoids backward filling (`bfill`) or future-looking transformations. All baseline windows and expanding stats operate strictly backward in time.

---

## 3. New Dataset Overview

- **Source File**: `data/daily_fitbit_sema_df_unprocessed.csv`
- **File Size**: 2,066,609 bytes (~2.07 MB)
- **Total Rows**: 7,410
- **Total Columns**: 64 (53 Numerical, 11 Object/Categorical)
- **Unique Users**: 71 (identifier column: `id`)
- **Unique Dates**: 283 calendar days
- **Date Range**: 2021-04-08 to 2022-01-22
- **Duplicate Rows**: 0 (0.00%)
- **Duplicate User-Date Pairs**: 0 (0.00%)
- **User Observation Span**:
  - Minimum days/user: 64
  - Maximum days/user: 244
  - Mean days/user: 104.37
  - Median days/user: 88.00

---

## 4. Dataset Provenance

> **"Provenance could not be independently verified from the available project files."**

Analysis of the local repository files confirms that while column names match standard Fitbit API structure and SEMA research schemas, no explicit metadata file or API provenance log was packaged with `daily_fitbit_sema_df_unprocessed.csv`.

---

## 5. User Coverage

- **Total User Count**: 71
- **Min Observations per User**: 64
- **25th Percentile**: 73.5
- **Median (50th Percentile)**: 88.0
- **75th Percentile**: 114.0
- **Max Observations per User**: 244

No user has fewer than 64 recorded days.

---

## 6. Temporal Coverage

- **Overall Date Span**: 283 days (2021-04-08 to 2022-01-22).
- **Cohort Continuous Run**: Median max continuous daily run per user is **87.0 days**.
- **Minimum Max Continuous Run**: 62 days (User `621e2e9867b776a24055be2f`).
- **Maximum Max Continuous Run**: 244 days (User `621e2e9567b776a24055bce5`).

---

## 7. EDA Findings

### Physiological Distributions:
1. **Resting Heart Rate (`resting_hr`)**: Range 44.12 - 86.00 bpm, mean 66.27 bpm, median 66.50 bpm.
2. **Daily Average Heart Rate (`bpm`)**: Range 43.50 - 154.00 bpm, mean 80.25 bpm, median 79.79 bpm.
3. **Heart Rate Variability (`rmssd`)**: Range 0.00 - 122.09 ms, mean 40.02 ms, median 34.07 ms. (Nocturnal sensor only).
4. **Blood Oxygen Saturation (`spo2`)**: Range 89.40% - 100.00%, mean 95.89%, median 95.90%. (Nocturnal sensor only).
5. **Daily Step Count (`steps`)**: Range 0 - 43,112 steps, mean 8,261.64, median 8,833.00.
6. **Sleep Duration (`sleep_duration`)**: Recorded in **milliseconds** (Range 3.6e6 - 7.45e7 ms, corresponding to 1.00 - 20.68 hours, mean 7.50 hours, median 7.63 hours). Divide by $3.6 \times 10^6$ for hours.
7. **Distance (`distance`)**: Recorded in **meters** (Range 0 - 29,850.7 m, mean 5,839.38 m = 5.84 km, median 5.21 km). Divide by $1000.0$ for km.
8. **Energy Expenditure (`calories`)**: Range 0.69 - 8,387.03 kcal, mean 2,182.96 kcal, median 2,073.60 kcal.

---

## 8. Missingness Analysis

Missingness breakdown across core candidate features:

| Column Name | Real Missing Count | Real Missing % | Cause of Missingness | Authenticity Assessment |
|---|---|---|---|---|
| `calories` / `calories_kcal` | 750 | 10.12% | Partial sync loss | High Fidelity (Easily bridged via $\le 2$d causal ffill) |
| `bpm` / `avg_hr_day_bpm` | 2,606 | 35.17% | Daytime non-wear | High Fidelity |
| `resting_hr` / `resting_hr_bpm` | 2,988 | 40.32% | Non-wear / rest span | High Fidelity |
| `steps` | 2,633 | 35.53% | Daytime non-wear | High Fidelity |
| `distance` | 2,633 | 35.53% | Daytime non-wear | High Fidelity |
| `sleep_duration` | 3,859 | 52.08% | Unrecorded sleep | Moderate Fidelity |
| `rmssd` / `hrv_rmssd_ms` | 4,935 | 66.60% | Nocturnal PPG requirement | **HIGH MISSINGNESS — MUST NOT BE MASS-FILLED COHORT-WIDE** |
| `spo2` / `spo2_avg_pct` | 6,140 | 82.86% | Nocturnal SpO2 hardware | **HIGH MISSINGNESS — MUST NOT BE MASS-FILLED COHORT-WIDE** |

---

## 9. Data Quality Analysis

- **Implausible Values**: Zero instances of negative heart rates, negative step counts, or SpO2 $> 100\%$.
- **Constant / Near-Constant Columns**: None.
- **Outliers**: Outliers in step counts ($> 40,000$) and sleep durations ($> 18$ hours) capped causally per user using expanding 1.5x IQR bounds.
- **Target Leakage**: `stress_score` is NOT used as a target label.

---

## 10. Feature Compatibility Matrix

Complete mapping table saved to `reports/FEATURE_COMPATIBILITY_MATRIX.csv`.

Summary Classification:
- **Core Common Space (Tier 1)**: `user_id`, `date`, `resting_hr_bpm`, `avg_hr_day_bpm`, `steps`, `distance_km`, `calories_kcal`, `sleep_duration_hours`, `sleep_efficiency`, `sleep_latency_min`, `wake_after_sleep_onset_min`.
- **Nocturnal Common Sub-Cohort (Tier 2 Only)**: `hrv_rmssd_ms`, `spo2_avg_pct` (Evaluated ONLY on authentic observed days; no mass cohort imputation).
- **Synthetic-Only (Dropped from Common Space)**: `sbp_mmHg`, `dbp_mmHg`, `caffeine_mg`, `alcohol_units`, `screen_time_min`, `mindfulness_minutes`.

---

## 11. Proposed Common Feature Space

The final harmonized **Authentic Common Feature Space** consists of:

### Tier 1 Primary Signals (6 core physiological & behavioral features):
1. `resting_hr_bpm`
2. `avg_hr_day_bpm`
3. `steps`
4. `distance_km`
5. `calories_kcal`
6. `sleep_duration_hours`

### Derived ML Feature Space (Used for Clustering & HMM):
- Personalized Baselines: `baseline_hr`, `baseline_sleep`, `baseline_steps`
- Raw Deviations: `hr_dev`, `sleep_dev`, `steps_dev`
- Z-Normalized Deviations: `hr_dev_z`, `sleep_dev_z`, `steps_dev_z`
- 7-Day Slopes: `hr_dev_slope_7d`, `sleep_dev_slope_7d`, `steps_dev_slope_7d`
- Composite Indicators: `severity_score` ($\sum |z_{i,t}|$), `trend_score` ($\sum |\beta_{7\text{d}}|$)

---

## 12. Unit Harmonization Requirements

1. **`sleep_duration`**: Divide by $3.6 \times 10^6$ (convert ms to hours).
2. **`distance`**: Divide by $1000.0$ (convert meters to kilometers).
3. **`wake_after_sleep_onset_min`**: Compute as `minutesAwake.fillna(0) + minutesAfterWakeup.fillna(0)`.
4. **`gender`**: Lowercase string conversion (`'MALE'` -> `'male'`).
5. **`age` & `bmi`**: Retained as categorical string metadata.

---

## 13. Conservative Missingness Scenarios & Empirical Comparison

We evaluated five conservative missingness scenarios with **authenticity as the primary objective** (Saved in `reports/DATA_RETENTION_SCENARIOS.csv`):

| Scenario | Description | Rows Retained | % Rows Retained | Users Retained | % Users Retained | Scientific Authenticity Assessment | Recommendation |
|---|---|---:|---:|---:|---:|---|---|
| **Scenario 1** | Strict Complete Case All 8 Features (inc. SpO2/HRV) | 1,066 | 14.39% | 23 | 32.39% | 100% Authentic (Severe user loss due to 82.9% SpO2 missingness) | Not Recommended (Sample size too small) |
| **Scenario 2** | Authentic Core Features (Excl. SpO2/HRV) Complete Case | 3,498 | 47.21% | 69 | 97.18% | 100% Authentic (Zero imputation of any kind) | Acceptable Pure Baseline |
| **Scenario 3** | Conservative Causal FFill (limit=2d) Core Features | 4,159 | 56.13% | 69 | 97.18% | High Authenticity (Max 48h within-user past fill, zero cohort fill) | Strong Candidate |
| **Scenario 4** | **Authentic Dual-Tier Architecture (Tier 1 Core + Tier 2 Nocturnal)** | **4,159 / 1,066** | **56.13% / 14.39%** | **69 / 23** | **97.18% / 32.39%** | **Maximum Scientific Authenticity (Primary model on 4,159 core rows; Nocturnal model on 1,066 authentic rows)** | **RECOMMENDED FOR PUBLICATION** |
| **Scenario 5** | Core Features (ffill limit=2d) + User History Filter ($\ge 14$d/user) | 4,115 | 55.53% | 61 | 85.92% | High Authenticity + Guaranteed baseline calibration history | High-Quality Alternative |

---

## 14. 7-Day Baseline Feasibility Under Authentic Filtering

Under **Scenario 4 (Tier 1 Core Space)**:
- **Users with $\ge 7$ observations**: 69 / 71 (**97.18%**)
- **Users with $\ge 14$ observations**: 61 / 71 (**85.92%**)
- **Users with $\ge 30$ observations**: 53 / 71 (**74.65%**)
- **Mean Usable History per User**: 60.3 days.
- **Usable Post-Warmup Observations**: **3,745 out of 4,159 rows (90.05%)** have a fully calibrated 7-day rolling baseline (`baseline_valid = True`).

---

## 15. Leakage Audit

1. **Temporal / Future Leakage**: NO backward filling (`bfill`) is allowed. Forward fill is capped at $\le 2$ days within-user. Rolling baselines operate strictly backward in time.
2. **Subject / User Leakage**: Cross-validation splitting must be conducted via `subject_independent_split` (by `user_id`).
3. **Target / Label Leakage**: `stress_score` is NOT used as an input feature for state-discovery clustering or HMM transitions.

---

## 16. Scientific Limitations

1. **Absence of Blood Pressure**: BP signals (`sbp_mmHg`, `dbp_mmHg`) are absent in Fitbit data and excluded from the Common Feature Space.
2. **Nocturnal Sensor Sparsity**: SpO2 and HRV are evaluated strictly in Tier 2 nocturnal sub-cohort experiments to prevent artificial imputation.

---

## 17. Final Recommendation

Adopt **Scenario 4 (Authentic Dual-Tier Architecture)**:
- **Primary Unsupervised Model**: Trained & evaluated on **4,159 authentic core observations across 69 users (97.18% cohort coverage)**.
- **Secondary Nocturnal Model**: Trained & evaluated on **1,066 authentic nocturnal observations across 23 users (32.39% cohort coverage)**.

---

## 18. Go/No-Go Decision

### **FINAL DECISION: GO (Under Authentic Dual-Tier Architecture)**

---

# FINAL GO / NO-GO ASSESSMENT

### 1. Is the real Fitbit dataset compatible with the existing feature methodology?
**YES**. Under the authentic 6-signal core feature space, all mathematical operations (baselines, deviations, slopes, severity) execute identically.

### 2. Can the existing 7-day personalized baseline be applied?
**YES**. 69 out of 71 real users (97.2%) retain sufficient authentic observations to calibrate personalized baselines.

### 3. What common features are available?
- **Core Tier 1 Space**: `resting_hr_bpm`, `avg_hr_day_bpm`, `steps`, `distance_km`, `calories_kcal`, `sleep_duration_hours`.
- **Nocturnal Tier 2 Space**: `hrv_rmssd_ms`, `spo2_avg_pct` (Authentic observed days only).

### 4. What features must be removed?
`sbp_mmHg`, `dbp_mmHg`, `caffeine_mg`, `alcohol_units`, `screen_time_min`, and `mindfulness_minutes` must be excluded. Cohort mean mass-imputation for SpO2 and HRV is explicitly removed.

### 5. What missing-value strategy is recommended?
**Scenario 4 (Authentic Dual-Tier)**: Conservative within-user forward fill ($\le 2$ days) on core signals. Zero cohort mean filling.

### 6. Approximately how many rows and users can be retained?
- **Tier 1 Core Space**: **4,159 authentic rows (56.13%) across 69 users (97.18%)**.
- **Tier 2 Nocturnal Space**: **1,066 authentic rows (14.39%) across 23 users (32.39%)**.

### 7. Can the real dataset support a valid real-data state-discovery experiment?
**YES**.

### 8. Can synthetic $\rightarrow$ real evaluation be performed?
**YES**. Synthetic models trained on the 6 core common features can be directly evaluated on the 4,159 authentic real Fitbit rows.

### 9. Can synthetic + real $\rightarrow$ real evaluation be performed?
**YES**.

### 10. What must be changed before implementation?
- Update preprocessing pipeline to support the 6-signal Core Feature Space (`resting_hr_bpm`, `avg_hr_day_bpm`, `steps`, `distance_km`, `calories_kcal`, `sleep_duration_hours`).
- Enforce conservative $\le 2$-day within-user causal ffill limit.
- Generate authentic processed datasets: `data/processed/synthetic_common.csv` and `data/processed/real_common.csv`.
