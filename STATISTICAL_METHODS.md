# STATISTICAL_METHODS.md: Consolidated Statistical Pipeline & Test Specifications

**Date**: 2026-08-23  
**Repository**: `SLOKESH2205/FINAL-YEAR-PROJECT`  
**Purpose**: Document every statistical test, variance correction, multiple-comparison adjustment, and noise-floor baseline used in the paper in one consolidated reference.

---

## 1. Overview of Experimental Protocol

- **Validation Protocol**: 5-Fold User-Level Cross-Validation (User-Disjoint Folds: `set(train_users) & set(test_users) == 0`).
- **Primary Cohort**: 61 HMM-eligible real users evaluated across identical held-out test folds in Experiments 2, 3, and 4.
- **Repeated Design**: 5 repeats of 5-fold CV (25 total resampled folds) using seeds `[42, 100, 2024, 777, 999]`.

---

## 2. Statistical Tests & Variance Corrections

### 2.1 Primary Independent CV Test ($N=5$ Folds)
- **Test**: Wilcoxon Signed-Rank Test (Non-Parametric) & Paired Student's $t$-test.
- **Sample Size**: $N=5$ independent folds.
- **Wilcoxon Floor Acknowledgment**: With $N=5$, the minimum two-sided $p$-value achievable by Wilcoxon signed-rank test is $p = 0.0625$. $N=5$ is explicitly acknowledged as underpowered.

### 2.2 Nadeau & Bengio (2003) Corrected Resampled $t$-Test ($N=25$ Resampled Folds)
- **Problem Addressed**: Resampled cross-validation introduces non-independence across repeats because users reappear across folds under different random splits. Naive $t$-tests underestimate variance and inflate $p$-values.
- **Correction Formula**:
  $$t_{\text{corrected}} = \frac{\bar{d}}{\sqrt{\left(\frac{1}{r \cdot k} + \frac{n_{\text{test}}}{n_{\text{train}}}\right) s_d^2}}$$
  where $r=5$ repeats, $k=5$ folds, $n_{\text{test}}/n_{\text{train}} = 1/4 = 0.25$, $s_d^2$ is the sample variance of the 25 paired differences, and degrees of freedom $df = r \cdot k - 1 = 24$.

---

## 3. Multiple-Comparison Corrections (Holm-Bonferroni)

- **Hypothesis Family**: RQ3 Ratio Sweep Pairwise Comparisons (3 non-independent comparisons: Exp2 vs Ratio 1:1, Exp2 vs Ratio 2:1, Exp2 vs Ratio 4:1).
- **Adjustment Method**: **Holm-Bonferroni Family-Wise Error Rate (FWER) Correction**.
- **Rationale**: Controls Type I error rates across non-independent ratio comparisons without the over-conservatism of standard Bonferroni correction.

---

## 4. Master Statistical Comparison Table

| Pairwise Comparison | Mean Diff ($\bar{d}$) | Std Diff ($s_d$) | Uncorrected $t$ | Nadeau-Bengio Corrected $t$ | Nadeau-Bengio Corrected $p$ | Holm-Bonferroni Adjusted $p$ | Final Statistical Conclusion |
|---|---:|---:|---:|---:|---:|---:|:---:|
| **Exp2 vs Exp3 (Zero-Shot)** | $+0.0055$ | $0.0025$ | $10.7513$ | $3.9929$ | $p = 0.000536$ | **$p_{\text{adj}} = 0.000536$** | Statistically Minor Gap |
| **Exp2 vs Exp4a (Ratio 1:1)** | $+0.0041$ | $0.0024$ | $8.6152$ | $3.1996$ | $p = 0.003844$ | **$p_{\text{adj}} = 0.003844$** | Statistically Minor Gap |
| **Exp2 vs Exp4b (Ratio 2:1)** | $+0.0048$ | $0.0025$ | $9.7034$ | $3.6037$ | $p = 0.001424$ | **$p_{\text{adj}} = 0.002848$** | Statistically Minor Gap |
| **Exp2 vs Exp4c (Ratio 4:1)** | $+0.0051$ | $0.0024$ | $10.4681$ | $3.8878$ | $p = 0.000699$ | **$p_{\text{adj}} = 0.002097$** | Statistically Minor Gap |

---

## 5. Permutation Null Baselines (50 Shuffles)

- **Synthetic Domain Noise Floor**: **`0.0870 ± 0.0032 Silhouette points`** (Exp 1 Silhouette `0.1681` clears floor).
- **Real Domain Noise Floor**: **`0.0939 ± 0.0075 Silhouette points`** (Exp 2 `0.2168`, Exp 3 `0.2117`, Exp 4 `0.2127` all double their noise floor).
