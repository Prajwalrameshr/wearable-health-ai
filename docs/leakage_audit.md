# Data Leakage Audit & Verification Report

| Leakage Flaw Identified in Original Pipeline | Scientific Risk | Rectification Implemented | Verification Status |
| :--- | :--- | :--- | :--- |
| **Backward Filling (`bfill()`)** | Future observations used to fill past missing values. | Removed `bfill()`. Implemented causal forward fill (`ffill()`) and initial cohort medians. | **FIXED & VERIFIED** |
| **Full-Timeline Outlier Statistics** | Full user series quantiles used future values to cap past outliers. | Replaced global quantiles with causal expanding IQR bounds calculated up to time $t$. | **FIXED & VERIFIED** |
| **Full-Dataset Scaler Fitting** | `StandardScaler.fit_transform(all_data)` before train/test split. | Separated fitting onto `train_df` only; validation and test sets are transformed using fitted scaler. Persisted to `scaler.pkl`. | **FIXED & VERIFIED** |
| **Rolling Baseline Initialization (`min_periods=1`)** | Single-day noise created uncalibrated baselines. | Enforced 7-day warm-up requirement (`min_periods=7`). Days 1–6 explicitly marked as warm-up. | **FIXED & VERIFIED** |
| **Cross-User Sequence Contamination in HMM** | Concatenating users into one sequence caused fake inter-user transitions. | Grouped observations by user and passed explicit sequence `lengths` array to `GaussianHMM`. | **FIXED & VERIFIED** |
| **Circular Pre-Generated Future Risk Targets** | Training against circular synthetic string labels. | Removed pre-generated risk targets and circular training logic completely. | **FIXED & VERIFIED** |
| **Environmental Context External Dependencies** | External OpenWeather API and non-physiological outdoor weather adjustments. | Completely eliminated `environment.py`, OpenWeather API, AQI, PM2.5, and outdoor weather adjustments across backend and UI. | **FIXED & VERIFIED** |
