from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def compute_size_matched_calinski_harabasz(feature_matrix: pd.DataFrame, labels: np.ndarray, target_n: int = 823, n_reps: int = 20, random_state: int = 42) -> float:
    """Compute size-matched Calinski-Harabasz score via subsampling when N > target_n."""
    n_samples = len(feature_matrix)
    if n_samples <= target_n:
        return float(calinski_harabasz_score(feature_matrix, labels))
    
    rng = np.random.default_rng(random_state)
    ch_scores = []
    for rep in range(n_reps):
        sub_idx = rng.choice(n_samples, size=target_n, replace=False)
        sub_X = feature_matrix.iloc[sub_idx] if isinstance(feature_matrix, pd.DataFrame) else feature_matrix[sub_idx]
        sub_y = labels[sub_idx]
        if len(np.unique(sub_y)) > 1:
            ch_scores.append(float(calinski_harabasz_score(sub_X, sub_y)))
    return float(np.mean(ch_scores)) if ch_scores else float(calinski_harabasz_score(feature_matrix, labels))


def compute_clustering_metrics(feature_matrix: pd.DataFrame, labels: np.ndarray, target_n: int = 823) -> dict[str, float]:
    """Calculate Silhouette, Davies-Bouldin, raw Calinski-Harabasz, and size-matched Calinski-Harabasz metrics."""
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return {
            "silhouette": float("nan"),
            "davies_bouldin": float("nan"),
            "calinski_harabasz": float("nan"),
            "calinski_harabasz_size_matched": float("nan"),
        }

    raw_ch = float(calinski_harabasz_score(feature_matrix, labels))
    sm_ch = compute_size_matched_calinski_harabasz(feature_matrix, labels, target_n=target_n)

    return {
        "silhouette": float(silhouette_score(feature_matrix, labels)),
        "davies_bouldin": float(davies_bouldin_score(feature_matrix, labels)),
        "calinski_harabasz": raw_ch,
        "calinski_harabasz_size_matched": sm_ch,
    }


def compute_entropy(probabilities: np.ndarray) -> float:
    """Compute mean Shannon entropy across posterior probability distributions (in bits)."""
    eps = 1e-12
    p = np.clip(probabilities, eps, 1.0)
    entropy = -np.sum(p * np.log2(p), axis=1)
    return float(np.mean(entropy))


def compute_hmm_metrics(
    decoded_df: pd.DataFrame,
    state_column: str = "soft_hmm_state",
    user_column: str = "user_id",
) -> dict[str, float]:
    """Calculate transition rate, mean state duration, and state persistence metrics."""
    if state_column not in decoded_df.columns:
        return {"transition_rate": 0.0, "mean_duration_days": 0.0}

    total_days = 0
    total_transitions = 0
    durations = []

    for _, group in decoded_df.groupby(user_column, sort=False):
        states = group[state_column].to_numpy()
        total_days += len(states)
        if len(states) > 1:
            total_transitions += int(np.sum(states[1:] != states[:-1]))

        # Calculate run length durations
        current_dur = 1
        for i in range(1, len(states)):
            if states[i] == states[i - 1]:
                current_dur += 1
            else:
                durations.append(current_dur)
                current_dur = 1
        durations.append(current_dur)

    transition_rate = float(total_transitions / total_days) if total_days > 0 else 0.0
    mean_duration = float(np.mean(durations)) if durations else 0.0

    return {
        "transition_rate": round(transition_rate, 4),
        "mean_duration_days": round(mean_duration, 2),
    }


def compute_confidence_intervals(
    data: np.ndarray | list[float],
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Compute sample mean and exact parametric/bootstrap 95% confidence interval."""
    arr = np.array(data, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return 0.0, 0.0, 0.0

    mean_val = float(np.mean(arr))
    if len(arr) < 2:
        return mean_val, mean_val, mean_val

    std_err = float(stats.sem(arr))
    margin = std_err * stats.t.ppf((1 + confidence) / 2.0, len(arr) - 1)
    return round(mean_val, 4), round(mean_val - margin, 4), round(mean_val + margin, 4)


def perform_wilcoxon_test(
    scores_a: list[float] | np.ndarray,
    scores_b: list[float] | np.ndarray,
) -> dict[str, float]:
    """
    Perform Wilcoxon signed-rank test for non-parametric paired comparison across users/folds.
    Calculates p-value and Cohen's d effect size.
    """
    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)
    diff = a - b
    diff = diff[~np.isnan(diff)]

    if len(diff) < 5 or np.all(diff == 0):
        return {"statistic": 0.0, "p_value": 1.0, "cohens_d": 0.0}

    try:
        res = stats.wilcoxon(diff)
        stat = float(res.statistic)
        p_val = float(res.pvalue)
    except Exception:
        stat = 0.0
        p_val = 1.0

    # Cohen's d
    std_diff = np.std(diff, ddof=1)
    cohens_d = float(np.mean(diff) / std_diff) if std_diff > 0 else 0.0

    return {
        "statistic": round(stat, 4),
        "p_value": round(p_val, 6),
        "cohens_d": round(cohens_d, 4),
    }
