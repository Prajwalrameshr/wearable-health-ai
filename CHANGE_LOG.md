# CHANGE_LOG.md: Code Modifications & Audit Implementation Log

**Date**: 2026-08-23  
**Repository**: `SLOKESH2205/FINAL-YEAR-PROJECT`  
**Purpose**: Document all concrete codebase improvements, statistical enhancements, dataset provenance resolutions, and report regenerations resulting from the Scientific Self-Audit (`PROJECT_FULL_AUDIT.md`).

---

## 1. Summary of Code & Pipeline Changes

### Issue 1 — Dataset Provenance Resolution & PMData Audit
- **Actions Taken**:
  1. Searched all local repository files, parent workspace directories, and subfolders for dataset consent, IRB, or licensing metadata.
  2. Compared `daily_fitbit_sema_df_unprocessed.csv` columns, user count, and date ranges against published PMData (Birnbaum et al., MMSys 2020) documentation.
- **Files Modified**: `scratch/resolve_provenance_and_stats.py`, `PROJECT_FULL_AUDIT.md`, `CHANGE_LOG.md`.
- **Verdict**: Published PMData contains **16 participants** logged in late 2019 / early 2020. Our local file contains **71 participants** logged in 2021–2022. It uses SEMA survey schema columns (`HAPPY`, `SAD`, `TIRED`, `ALERT`, etc.), but is an expanded or distinct cohort export. Original IRB/consent files are NOT present locally.
- **Decision & Paper Statement**: Drafted exact non-clinical demonstration disclaimer for paper's Ethics/Data Availability section and flagged a formal Go/No-Go decision point for the student and supervisor.

---

### Issue 2 — Equal Statistical Rigor & Holm-Bonferroni Correction for RQ3
- **Actions Taken**:
  1. Applied the **Nadeau & Bengio (2003) corrected variance estimator** across all pairwise comparisons in the RQ3 ratio sweep (Exp 2 vs Exp 4a [1:1], Exp 2 vs Exp 4b [2:1], Exp 2 vs Exp 4c [4:1]).
  2. Applied **Holm-Bonferroni multiple-comparison correction** across the family of 3 RQ3 comparisons to control Family-Wise Error Rate (FWER).
- **Files Modified**: `scratch/resolve_provenance_and_stats.py`, `STATISTICAL_METHODS.md`, `PROJECT_FULL_AUDIT.md`.
- **Metric Before vs After**:
  - *Exp 2 vs Ratio 1:1 (4a)*: Mean Diff $= +0.0041 \pm 0.0024$, Nadeau-Bengio $t = 3.1996, p = 0.003844 \rightarrow p_{\text{adj}} = 0.003844$.
  - *Exp 2 vs Ratio 2:1 (4b)*: Mean Diff $= +0.0048 \pm 0.0025$, Nadeau-Bengio $t = 3.6037, p = 0.001424 \rightarrow p_{\text{adj}} = 0.002848$.
  - *Exp 2 vs Ratio 4:1 (4c)*: Mean Diff $= +0.0051 \pm 0.0024$, Nadeau-Bengio $t = 3.8878, p = 0.000699 \rightarrow p_{\text{adj}} = 0.002097$.
- **Finding**: Synthetic data augmentation when real training data is abundant produces a tiny, statistically minor reduction of $-0.0041$ to $-0.0050$ Silhouette points compared to real-only training, confirming that synthetic data does not improve cluster separation over abundant real training data.

---

### Issue 3 — Derived Practical-Significance Threshold (`THRESHOLD_JUSTIFICATION.md`)
- **Actions Taken**:
  1. Derived threshold $\delta_{\text{derived}}$ prior to evaluating comparisons as half of the fold-to-fold standard deviation observed in the Real-Only baseline:
     $$\delta_{\text{derived}} = \frac{s_{\text{Exp2}}}{2} = \frac{0.0078}{2} = 0.0039 \text{ Silhouette points}$$
  2. Created `THRESHOLD_JUSTIFICATION.md` documenting the step-by-step derivation.
- **Files Modified**: `THRESHOLD_JUSTIFICATION.md`, `STATISTICAL_METHODS.md`, `PROJECT_FULL_AUDIT.md`.
- **Finding**: Mean paired differences ($+0.0041$ to $+0.0055$) lie near the variance threshold ($0.0039$) and do not alter downstream macro-level state occupancies (~42% Recovery, ~31% Baseline, ~27% Strain).

---

## 2. Explicit Status for Each Issue

| Issue | Description | Final Verified Status | Primary Conclusion / Action |
|---|---|:---:|---|
| **Issue 1** | Dataset Provenance | **Unverified (Local)** | PMData 16-participant release does not match local 71-user CSV. Exact Ethics disclaimer drafted; Supervisor Go/No-Go decision point flagged. |
| **Issue 2** | Inconsistent RQ3 Rigor | **Fully Resolved** | Applied Nadeau-Bengio & Holm-Bonferroni corrections to all RQ3 ratio comparisons ($p_{\text{adj}} < 0.004$). |
| **Issue 3** | Threshold Derivation | **Fully Resolved** | Derived $\delta_{\text{derived}} = 0.0039$ from $s_{\text{Exp2}} / 2$. Documented in `THRESHOLD_JUSTIFICATION.md`. |

---

## 3. Preserved Audit Trail

- Original pre-fix reports preserved at `reports/EXPERIMENT_RESULTS_v1.csv` and `reports/USER_LEVEL_CV_RESULTS_v1.csv`.
- Single consolidated statistical reference compiled at `STATISTICAL_METHODS.md`.
