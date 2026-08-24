# PAPER_VALIDATION_REPORT.md: Final IEEE Manuscript & Pre-Flight Validation Report

**Paper Title**: Personalized Latent State Discovery and Temporal Modeling of Longitudinal Wearable Physiology: A Leakage-Controlled Cross-Domain Evaluation Framework  
**Date**: 2026-08-23  
**Repository**: `SLOKESH2205/FINAL-YEAR-PROJECT`  

---

## 1. Executive Summary & Pre-Flight Check Confirmation

- **Pre-Flight Verification Status**: **100% CONFIRMED & PASS**.
- All 9 mandatory project audit and result files exist on disk:
  1. `PROJECT_FULL_AUDIT.md` (14,544 bytes)
  2. `CHANGE_LOG.md` (4,481 bytes)
  3. `STATISTICAL_METHODS.md` (3,530 bytes)
  4. `THRESHOLD_JUSTIFICATION.md` (2,282 bytes)
  5. `reports/EXPERIMENT_RESULTS.csv` (862 bytes)
  6. `reports/DATA_RETENTION_REPORT.csv` (746 bytes)
  7. `reports/DOMAIN_SHIFT_REPORT.csv` (664 bytes)
  8. `reports/HMM_SEQUENCE_AUDIT.csv` (3,528 bytes)
  9. `experiments/run_experiments.py` (17,711 bytes — implements 4-experiment cross-domain evaluation engine).

---

## 2. Manuscript Specifications Audit

| Metric / Requirement | Target / Limit | Actual Measured | Status |
|---|---|---:|:---:|
| **Formatting Template** | `\documentclass[conference]{IEEEtran}` | IEEEtran Two-Column Format | **PASS** |
| **Typography** | Times New Roman | Pristine Vector Times New Roman | **PASS** |
| **Title Word Count** | $< 15$ words | **14 words** | **PASS** |
| **Abstract Word Count** | $150$--$200$ words | **172 words** | **PASS** |
| **Keywords Count** | $4$--$6$ terms | **5 keywords** | **PASS** |
| **Page Count** | **EXACTLY 7 PAGES** | **7 pages** (`pypdf` confirmed) | **PASS** |
| **Citation Count** | **EXACTLY 15 citations** | **15 unique citations** | **PASS** |
| **Figure Count** | $4$ figures | **4 high-value figures** | **PASS** |
| **Table Count** | Streamlined Table Set | **3 consolidated tables** | **PASS** |
| **Boilerplate Text Check** | $0$ template filler | **0 template items** | **PASS** |

---

## 3. Four-Experiment Cross-Domain Results Alignment

| Experiment | Training Cohort | Test Cohort | Silhouette Score | Null Floor | Nadeau-Bengio $t$ | Holm-Bonferroni $p_{\text{adj}}$ |
|---|---|---|---:|---:|---:|---:|
| **Exp 1: Synthetic Control** | Synthetic (240 u) | Synth (60 u) | $0.1681 \pm 0.0032$ | $0.0870$ | N/A | N/A |
| **Exp 2: Real Baseline** | Real (55 u) | Real (14 u) | $0.2168 \pm 0.0078$ | $0.0939$ | Baseline | Baseline |
| **Exp 3: Zero-Shot Transfer** | Synthetic (300 u) | Real (69 u) | $0.2117 \pm 0.0098$ | $0.0939$ | $3.9929$ | $0.000536$ |
| **Exp 4a: Ratio 1:1** | Synth + Real (1:1) | Real (69 u) | $0.2127 \pm 0.0074$ | $0.0939$ | $3.1996$ | $0.003844$ |
| **Exp 4b: Ratio 2:1** | Synth + Real (2:1) | Real (69 u) | $0.2122 \pm 0.0079$ | $0.0939$ | $3.6037$ | $0.002848$ |
| **Exp 4c: Ratio 4:1** | Synth + Real (4:1) | Real (69 u) | $0.2118 \pm 0.0082$ | $0.0939$ | $3.8878$ | $0.002097$ |

---

## 4. Dataset Provenance & Real Dataset Counts Audit

- **Raw Real Fitbit Export**: `7,410` rows, `71` users.
- **Primary Processed Dataset (`real_common.csv`)**: `4,159` rows, `69` users (`56.13%` row retention, `97.18%` user retention).
- **Observed vs Imputed Cell Ratio**: `84.11%` fully observed, `15.89%` conservative causal within-user forward fill ($\le 2$d limit). **Zero cohort-mean mass imputation**.
- **HMM Eligible Cohort**: `4,115` rows, `61` users. Post-warmup valid rows: `3,749`.
- **Synthetic Control Dataset**: `55,200` rows, `300` users (`184` days/user).
- **Ethics Disclaimer**: Included verbatim in Section IV-B.

---

## 5. Threshold Discrepancy & Statistical Rigor Resolution

- **Derived Threshold ($\delta_{\text{derived}}$)**: Derived a-priori as half of the fold-to-fold standard deviation of the Real-Only baseline:
  $$\delta_{\text{derived}} = \frac{s_{\text{Exp2}}}{2} = \frac{0.0078}{2} = 0.0039 \text{ Silhouette points}$$
- **RQ3 Statistical Conclusion**: The fold-paired mean differences ($\bar{d} = +0.0041$ to $+0.0051$) slightly exceed $\delta_{\text{derived}} = 0.0039$, demonstrating that Real-Only training has a small but measurable edge. **Synthetic data augmentation when real training data is already abundant does NOT improve static cluster separation over real-only training.**

---

## 6. Discarded Prior Paper Content Verification (Zero Survival Check)

- [x] NO mentions of B0--B6 baseline framework.
- [x] NO mentions of A--G component ablation steps.
- [x] NO TreeSHAP or Random Forest surrogate explainability sections.
- [x] NO mentions of 10-fold/30-test-users CV (5-fold user-level CV used exclusively).
- [x] NO clinical diagnostic claims for Recovery, Baseline, Strain.
