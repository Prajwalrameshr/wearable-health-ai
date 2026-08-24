# Experimental Protocol

## 1. Split Protocols

### 1.1 Chronological Split (Within-Subject Temporal Evaluation)
- **Train Split**: First 70% of each user's chronological timeline.
- **Validation Split**: Next 15% of each user's chronological timeline (hyperparameter tuning).
- **Test Split**: Final 15% of each user's chronological timeline (completely unseen during model fitting).
- **No Row Shuffling**: Time series order is strictly maintained.

### 1.2 Subject-Independent Split (Cross-Subject Evaluation)
- **Protocol**: 5-fold GroupKFold cross-validation across unique user IDs.
- **Fold Sizes**: 80% train users (40 users), 20% held-out test users (10 users) per fold.
- **No Contamination**: Zero user overlap between train and test splits in any fold.

---

## 2. Evaluation Baselines (B0 to B6)
- **B0**: Global Raw Physiology (GMM on unnormalized raw measurements)
- **B1**: Personalized Threshold Rule (fixed standard deviation threshold on deviations)
- **B2**: KMeans Raw Features
- **B3**: KMeans Personalized Deviations
- **B4**: GMM Personalized Deviations
- **B5**: GMM + Hard-State Categorical HMM
- **B6 (Proposed)**: GMM + Continuous Soft-Posterior HMM

---

## 3. Statistical Testing Methodology
- **Multi-Seed Testing**: 5 random seeds (42, 100, 2024, 777, 999).
- **Confidence Intervals**: 95% parametric and bootstrap confidence intervals.
- **Non-Parametric Significance**: Wilcoxon signed-rank test across subjects/folds.
- **Effect Size**: Cohen's $d$ on paired difference distributions.
