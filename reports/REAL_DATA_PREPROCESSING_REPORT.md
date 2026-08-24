# Real Fitbit Dataset Authentic Preprocessing & Harmonization Report

**Project**: Wearable Health Unsupervised Latent State Discovery  
**Evaluation Scope**: Scientific Authenticity, Data Quality, Feature Harmonization, and Methodology Compatibility Assessment  
**Target Dataset**: `data/daily_fitbit_sema_df_unprocessed.csv`  
**Baseline Dataset**: `data/wearables_health_6mo_daily.csv` (Synthetic)  
**Primary Objective**: **MAXIMUM SCIENTIFIC AUTHENTICITY AND DEFENSIBILITY**  

---

## 1. Executive Summary

This preprocessing report details the authentic harmonization of the real-world Fitbit dataset (`daily_fitbit_sema_df_unprocessed.csv`) with the existing synthetic dataset (`wearables_health_6mo_daily.csv`).

Key Implementation Highlights:
1. **Raw Data Preservation**: `data/daily_fitbit_sema_df_unprocessed.csv` remains completely untouched.
2. **Primary Core Feature Space (6 Signals)**: `resting_hr_bpm`, `avg_hr_day_bpm`, `steps`, `distance_km`, `calories_kcal`, `sleep_duration_hours`.
3. **Causal Infill Rules**: Max 2 consecutive days of within-user causal forward fill (`ffill(limit=2)`). Zero mass cohort mean imputation for physiological signals.
4. **Primary Real Dataset Output**: `data/processed/real_common.csv` (**4,159 observations across 69 users**).
5. **Harmonized Synthetic Dataset Output**: `data/processed/synthetic_common.csv` (**55,200 observations across 300 users**).
6. **HMM Eligibility Gate**: **61 eligible real users (4,115 rows / 3,749 post-warmup valid rows)** meeting strict eligibility criteria ($\ge 14$ total days, $\ge 7$ post-warmup days). This exact same set of 61 real test users is evaluated across Experiments 2, 3, and 4.
