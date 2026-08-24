# Cross-Domain Experiment Design & Protocol Specification

**Research Project**: Unsupervised Longitudinal Wearable Health State Discovery  
**Primary Objective**: Investigate cross-domain latent state transfer from synthetic to real-world wearable data and evaluate synthetic data augmentation when real training data is limited (RQ3).

---

## 1. Experimental Methodology Overview

All experiments enforce strict subject-independent (user-level) cross-validation and causal preprocessing to eliminate data leakage.

### Key Safeguards:
1. **User-Level Split**: Users are partitioned into 5 balanced folds. No daily observation from a user in training appears in test data.
2. **Identical Test User Alignment**: For each fold, Experiments 2, 3, and 4 evaluate on the **EXACT SAME held-out eligible real test users**.
3. **Frozen Model Hyperparameters**: GMM component count ($K=3$) and HMM state count are frozen across all ratio conditions.
4. **Independent State Naming**: Cluster IDs are sorted strictly by median harmonized `severity_score`:
   - Lowest Median Severity = **Recovery**
   - Middle Median Severity = **Baseline**
   - Highest Median Severity = **Strain**

---

## 2. Four Core Experiments

### Experiment 1: Synthetic -> Synthetic Baseline
- **Training Set**: Synthetic training users (4/5 of synthetic cohort, ~44,160 rows).
- **Test Set**: Held-out synthetic test users (1/5 of synthetic cohort, ~11,040 rows).
- **Feature Space**: Harmonized Primary 6-Core Feature Matrix (`resting_hr_bpm`, `avg_hr_day_bpm`, `steps`, `distance_km`, `calories_kcal`, `sleep_duration_hours`).

### Experiment 2: Real -> Real Baseline
- **Training Set**: Real training users (4/5 of eligible real cohort, ~3,292 rows across 48.8 users).
- **Test Set**: Held-out real test users (1/5 of eligible real cohort, ~823 rows across 12.2 users).
- **Feature Space**: Harmonized Primary 6-Core Feature Matrix.

### Experiment 3: Synthetic -> Real Zero-Shot Transfer
- **Training Set**: Synthetic training users ONLY (~44,160 rows). Zero real training data used.
- **Test Set**: The EXACT SAME held-out real test users as Experiment 2 (~823 rows per fold).
- **Feature Space**: Harmonized Primary 6-Core Feature Matrix.

### Experiment 4: Synthetic + Real -> Real Augmentation Sweep (RQ3)
- **Ratio Conditions**:
  - `0:1`: Real-Only Baseline (Identical to Exp 2).
  - `1:1`: Controlled 1:1 Synthetic:Real training user ratio.
  - `2:1`: Controlled 2:1 Synthetic:Real training user ratio.
  - `4:1`: Controlled 4:1 Synthetic:Real training user ratio.
- **Sampling Strategy**: User-balanced controlled sampling from synthetic training pool to prevent synthetic row count from dominating real domain representation.
- **Test Set**: Evaluated on the EXACT SAME held-out real test users across all ratio conditions.

---

## 3. Secondary HRV / SpO2 Analysis

- Conducted on authentic nocturnal sub-cohort (**1,066 authentic rows across 23 users**).
- Evaluates whether adding genuinely measured HRV RMSSD and SpO2 alters the discovered latent state boundaries.
- Zero fabricated SpO2 or HRV values.

---

## 4. Evaluation Metrics

1. **Clustering Quality**: Silhouette Score, Calinski-Harabasz Index, Davies-Bouldin Index.
2. **Model Likelihood**: GMM Log-Likelihood per sample.
3. **State Distribution**: Occupancy percentages for Recovery, Baseline, Strain.
4. **Statistical Uncertainty**: 5-Fold User-Level Mean, Standard Deviation, and 95% Confidence Intervals via user-level bootstrapping.
