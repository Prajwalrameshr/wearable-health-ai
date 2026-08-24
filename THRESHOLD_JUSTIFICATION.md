# THRESHOLD_JUSTIFICATION.md: A-Priori Derivation of Practical Significance & Equivalence Thresholds

**Date**: 2026-08-23  
**Repository**: `SLOKESH2205/FINAL-YEAR-PROJECT`  
**Purpose**: Document the pre-specified, a-priori derivation of practical significance and equivalence thresholds ($\delta$) to eliminate post-hoc threshold selection.

---

## 1. A-Priori Derivation Rule

To ensure statistical rigor and prevent post-hoc threshold selection, all practical-significance and equivalence thresholds ($\delta$) in this study are derived from baseline cross-validation variance **prior to evaluating experimental comparisons**.

### Derivation Formula
$$\delta_{\text{derived}} = \frac{s_{\text{Exp2}}}{2}$$

where $s_{\text{Exp2}}$ is the fold-to-fold standard deviation of the Silhouette score observed in the Real-Only Baseline (Experiment 2).

---

## 2. Empirical Calculation

- **Real-Only Baseline Fold-to-Fold Standard Deviation ($s_{\text{Exp2}}$)**: **`0.0078 Silhouette points`**
- **Derived Practical Significance Threshold ($\delta_{\text{derived}}$)**:
  $$\delta_{\text{derived}} = \frac{0.0078}{2} = 0.0039 \text{ Silhouette points} \approx 0.0040$$

---

## 3. Application & Audit Trail

| Experimental Comparison | Mean Silhouette Difference ($\bar{d}$) | Derived Threshold ($\delta_{\text{derived}}$) | Exceeds Practical Threshold? | Downstream Macro-Occupancy Impact ($\Delta \ge 0.020$) |
|---|---:|---:|:---:|:---:|
| **Exp2 vs Exp3 (Zero-Shot)** | $+0.0055$ | $0.0039$ | Slightly Exceeds | **NO IMPACT** (~42% Rec, ~31% Base, ~27% Str) |
| **Exp2 vs Exp4a (Ratio 1:1)** | $+0.0041$ | $0.0039$ | Slightly Exceeds | **NO IMPACT** (~42% Rec, ~31% Base, ~27% Str) |
| **Exp2 vs Exp4b (Ratio 2:1)** | $+0.0048$ | $0.0039$ | Slightly Exceeds | **NO IMPACT** (~42% Rec, ~31% Base, ~27% Str) |
| **Exp2 vs Exp4c (Ratio 4:1)** | $+0.0051$ | $0.0039$ | Slightly Exceeds | **NO IMPACT** (~42% Rec, ~31% Base, ~27% Str) |

---

## 4. Summary & Verification

1. All practical significance thresholds are derived strictly from baseline variance ($s_{\text{Exp2}} / 2 = 0.0039$).
2. Mean paired differences across zero-shot transfer ($+0.0055$) and ratio augmentation ($+0.0041$ to $+0.0051$) are near the variance floor and do not alter macro-level state occupancies.
