# Research Methodology — Wearable Health AI

## 1. Primary Research Contribution

This research presents a scientifically rigorous framework for **Personalized Temporal Modeling of Physiological States from Longitudinal Wearable Data**.

### Core Architecture:
$$\text{Longitudinal Wearable Data} \longrightarrow \text{Causal Preprocessing} \longrightarrow \text{Personal Baseline} \longrightarrow \text{Deviations \& Slopes} \longrightarrow \text{GMM Soft Probabilities} \longrightarrow \text{Soft-Posterior Temporal HMM} \longrightarrow \text{Physiological State / Strain Assessment}$$

---

## 2. Causal Preprocessing & Baseline Formulation

To eliminate data leakage, all preprocessing transforms operate strictly on historical information available up to observation time $t$ ($X_{\le t}$):

### 2.1 Causal Missing-Value Imputation
- No backward filling (`bfill`) is permitted, as backward filling uses future observations to impute past gaps.
- Missing values are causally forward-filled (`ffill`). Initial unobserved prefix values are initialized with historical cohort medians without look-ahead.

### 2.2 Causal Outlier Capping
- Causal expanding IQR bounds are calculated per user:
  $$\text{IQR}(t) = Q_3(X_{\le t}) - Q_1(X_{\le t})$$
  $$\text{Bound}_{\text{upper}}(t) = Q_3(X_{\le t}) + 1.5 \times \text{IQR}(t)$$
  Observations at time $t$ are capped using $\text{Bound}_{\text{upper}}(t)$ without incorporating future timeline statistics.

### 2.3 7-Day Baseline Warm-Up
- Personalized physiological baselines are computed using a 7-day rolling window with a mandatory warm-up window ($min\_periods=7$).
- Days 1–6 are explicitly flagged as warm-up mode (`baseline_valid = False`), preventing uncalibrated early observations from distorting deviation features.

---

## 3. Dynamic Unsupervised State Discovery (GMM)

Rather than enforcing an arbitrary $k=3$ cluster bias, Gaussian Mixture Models evaluate $k \in \{2, 3, 4, 5\}$ and select $k$ dynamically by minimizing the Bayesian Information Criterion (BIC):
$$\text{BIC} = -2 \ln(\widehat{L}) + k_{\text{params}} \ln(N)$$

Instead of discretizing observations into hard state labels, the model retains continuous soft posterior probability vectors $P(S_i | x_t)$:
$$P(S_i | x_t) = \frac{\pi_i \mathcal{N}(x_t | \mu_i, \Sigma_i)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_t | \mu_j, \Sigma_j)}$$

---

## 4. Continuous Soft-Posterior Temporal HMM

The continuous HMM models temporal transitions over GMM posterior probability vectors directly:
- **State Transition Matrix**: $A_{ij} = P(S_t = j | S_{t-1} = i)$
- **Sequence Isolation**: Every user timeline is treated as a separate observation sequence with explicit sequence lengths, preventing artificial cross-user transition boundaries.
- **Uncertainty Preservation**: Continuous Gaussian emission density over probability vectors prevents information loss caused by hard integer state discretization.

---

## 5. Model-Native Explainability & Surrogate SHAP

- **Model-Native GMM Attribution**: Feature contribution is derived directly from component parameter distances weighted by component covariance:
  $$I(f) = \frac{\text{Var}_{k}(\mu_{k, f})}{\text{Mean}_{k}(\sigma_{k, f}^2) + \epsilon}$$
- **Surrogate-Model SHAP**: For compatibility with tree-based explainability visual interfaces, a Random Forest surrogate classifier is trained on GMM latent labels. Surrogate classification fidelity is explicitly calculated and reported (Accuracy, Balanced Accuracy, Macro-F1, Confusion Matrix).
