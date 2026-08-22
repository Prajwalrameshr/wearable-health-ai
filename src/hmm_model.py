from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
try:
    from hmmlearn.hmm import CategoricalHMM
except ImportError:
    try:
        from hmmlearn.hmm import MultinomialHMM as CategoricalHMM
    except ImportError:
        from hmmlearn.hmm import GaussianHMM as CategoricalHMM
from hmmlearn.hmm import GaussianHMM

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
DEFAULT_USER_COLUMN = "user_id"
DEFAULT_DATE_COLUMN = "date"
STANDARD_STATES = ["Recovery", "Baseline", "Strain"]


def load_hmm_model(model_name: str = "hmm.pkl"):
    model_path = MODEL_DIR / model_name
    if model_path.exists() and model_path.stat().st_size > 0:
        try:
            return joblib.load(model_path)
        except Exception:
            return None
    return None


def save_hmm_model(model: Any, model_name: str = "hmm.pkl") -> Path:
    model_path = MODEL_DIR / model_name
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model_path


def validate_hmm_input(df: pd.DataFrame, observation_column: str, user_column: str = DEFAULT_USER_COLUMN, date_column: str = DEFAULT_DATE_COLUMN) -> pd.DataFrame:
    required_columns = [user_column, date_column, observation_column]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required HMM input columns: {', '.join(missing_columns)}")
    ordered = df.copy()
    ordered[date_column] = pd.to_datetime(ordered[date_column], errors="coerce")
    ordered = ordered.dropna(subset=[date_column, observation_column])
    ordered[observation_column] = pd.to_numeric(ordered[observation_column], errors="coerce").astype(int)
    return ordered.sort_values([user_column, date_column]).reset_index(drop=True)


def build_hmm_sequences(df: pd.DataFrame, observation_column: str, user_column: str = DEFAULT_USER_COLUMN) -> tuple[np.ndarray, list[int], pd.Index, list[int]]:
    observed_labels = sorted(df[observation_column].dropna().astype(int).unique().tolist())
    label_to_index = {label: idx for idx, label in enumerate(observed_labels)}
    sequences: list[np.ndarray] = []
    lengths: list[int] = []
    ordered_index_parts: list[pd.Index] = []
    for _, group in df.groupby(user_column, sort=False):
        group_obs = group[observation_column].map(label_to_index).to_numpy(dtype=int).reshape(-1, 1)
        sequences.append(group_obs)
        lengths.append(len(group_obs))
        ordered_index_parts.append(group.index)
    concatenated = np.vstack(sequences) if sequences else np.empty((0, 1), dtype=int)
    ordered_index = ordered_index_parts[0].append(ordered_index_parts[1:]) if len(ordered_index_parts) > 1 else (ordered_index_parts[0] if ordered_index_parts else pd.Index([]))
    return concatenated, lengths, ordered_index, observed_labels


def train_hmm(df: pd.DataFrame, observation_column: str, n_components: int = 3, user_column: str = DEFAULT_USER_COLUMN, date_column: str = DEFAULT_DATE_COLUMN, random_state: int = 42) -> tuple[Any, pd.DataFrame, np.ndarray, list[int], list[int]]:
    ordered_df = validate_hmm_input(df, observation_column=observation_column, user_column=user_column, date_column=date_column)
    observations, lengths, _, observed_labels = build_hmm_sequences(ordered_df, observation_column=observation_column, user_column=user_column)
    effective_components = max(1, min(n_components, len(observed_labels), len(observations)))
    try:
        model = CategoricalHMM(n_components=effective_components, n_iter=300, tol=1e-2, random_state=random_state)
        model.fit(observations, lengths)
    except Exception:
        model = GaussianHMM(n_components=effective_components, covariance_type="diag", n_iter=300, tol=1e-2, random_state=random_state, min_covar=1e-3)
        model.fit(observations.astype(float), lengths)
    return model, ordered_df, observations, lengths, observed_labels


def decode_states(model: Any, df: pd.DataFrame, observation_column: str, output_column: str = "hmm_state_sequence", user_column: str = DEFAULT_USER_COLUMN, date_column: str = DEFAULT_DATE_COLUMN) -> pd.DataFrame:
    ordered_df = validate_hmm_input(df, observation_column=observation_column, user_column=user_column, date_column=date_column)
    observations, lengths, _, _ = build_hmm_sequences(ordered_df, observation_column=observation_column, user_column=user_column)
    try:
        hidden_states = model.predict(observations, lengths)
        posteriors = model.predict_proba(observations, lengths)
    except Exception:
        hidden_states = model.predict(observations.astype(float), lengths)
        posteriors = model.predict_proba(observations.astype(float), lengths)
    decoded_df = ordered_df.copy()
    decoded_df[output_column] = hidden_states.astype(int)
    for state_index in range(posteriors.shape[1]):
        decoded_df[f"{output_column}_prob_{state_index}"] = posteriors[:, state_index]
    decoded_df[f"{output_column}_confidence"] = posteriors.max(axis=1)
    return decoded_df


def build_hmm_state_map(decoded_df: pd.DataFrame, state_column: str, severity_column: str = "severity_score") -> dict[int, str]:
    if severity_column not in decoded_df.columns:
        unique_states = sorted(decoded_df[state_column].dropna().astype(int).unique().tolist())
        return {state: STANDARD_STATES[min(index, len(STANDARD_STATES) - 1)] for index, state in enumerate(unique_states)}
    ordered_states = decoded_df.groupby(state_column)[severity_column].mean().sort_values().index.astype(int).tolist()
    return {state: STANDARD_STATES[min(index, len(STANDARD_STATES) - 1)] for index, state in enumerate(ordered_states)}


def attach_state_persistence(decoded_df: pd.DataFrame, state_column: str = "hmm_state_sequence", user_column: str = DEFAULT_USER_COLUMN, output_column: str | None = None) -> pd.DataFrame:
    enriched = decoded_df.copy()
    duration_column = output_column or f"{state_column}_duration"
    enriched[duration_column] = 0
    for _, group in enriched.groupby(user_column, sort=False):
        states = group[state_column].to_numpy(dtype=int)
        durations = np.zeros(len(states), dtype=int)
        run_start = 0
        for index in range(1, len(states) + 1):
            if index < len(states) and states[index] == states[run_start]:
                continue
            run_length = index - run_start
            durations[run_start:index] = run_length
            run_start = index
        enriched.loc[group.index, duration_column] = durations
    return enriched


def compute_transition_matrix(model: GaussianHMM, state_map: dict[int, str] | None = None) -> pd.DataFrame:
    state_names = [state_map.get(idx, f"state_{idx}") for idx in range(model.n_components)] if state_map else [f"state_{idx}" for idx in range(model.n_components)]
    return pd.DataFrame(model.transmat_, index=state_names, columns=state_names).round(4)


def compute_transition_rate(decoded_df: pd.DataFrame, state_column: str = "hmm_state_sequence", user_column: str = DEFAULT_USER_COLUMN) -> float:
    total_days = 0
    total_transitions = 0
    for _, group in decoded_df.groupby(user_column, sort=False):
        states = group[state_column].to_numpy()
        total_days += len(states)
        if len(states) > 1:
            total_transitions += int(np.sum(states[1:] != states[:-1]))
    return float(total_transitions / total_days) if total_days else 0.0


def compute_state_distribution(decoded_df: pd.DataFrame, state_column: str = "hmm_state_sequence", state_map: dict[int, str] | None = None) -> pd.Series:
    distribution = decoded_df[state_column].value_counts(normalize=True).sort_index() * 100.0
    distribution.index = [state_map.get(int(idx), f"state_{int(idx)}") if state_map else f"state_{int(idx)}" for idx in distribution.index]
    return distribution.round(2)


def run_hmm_pipeline(df: pd.DataFrame, observation_column: str, n_components: int = 3, user_column: str = DEFAULT_USER_COLUMN, date_column: str = DEFAULT_DATE_COLUMN, model_name: str | None = None, save_model: bool = False) -> dict[str, Any]:
    model, ordered_df, observations, lengths, observed_labels = train_hmm(df, observation_column=observation_column, n_components=n_components, user_column=user_column, date_column=date_column)
    output_column = f"{observation_column}_hmm_state_sequence"
    decoded_df = decode_states(model, ordered_df, observation_column=observation_column, output_column=output_column, user_column=user_column, date_column=date_column)
    state_map = build_hmm_state_map(decoded_df, state_column=output_column)
    decoded_df[f"{output_column}_label"] = decoded_df[output_column].map(state_map)
    renamed_probability_columns: list[str] = []
    for state_idx, state_label in state_map.items():
        source_column = f"{output_column}_prob_{int(state_idx)}"
        target_column = f"{output_column}_{str(state_label).lower().replace(' ', '_')}_prob"
        if source_column in decoded_df.columns:
            decoded_df[target_column] = decoded_df[source_column]
            renamed_probability_columns.append(target_column)
    duration_column = f"{output_column}_duration"
    decoded_df = attach_state_persistence(decoded_df, state_column=output_column, user_column=user_column, output_column=duration_column)
    transition_matrix = compute_transition_matrix(model, state_map=state_map)
    saved_model_path = save_hmm_model(model, model_name=model_name or f"{observation_column}_hmm.pkl") if save_model else None
    return {
        "decoded_df": decoded_df,
        "transition_matrix": transition_matrix,
        "transition_rate": compute_transition_rate(decoded_df, state_column=output_column, user_column=user_column),
        "state_distribution": compute_state_distribution(decoded_df, state_column=output_column, state_map=state_map),
        "state_map": state_map,
        "duration_column": duration_column,
        "probability_columns": [f"{output_column}_prob_{state_idx}" for state_idx in range(model.n_components)],
        "named_probability_columns": renamed_probability_columns,
        "state_column": output_column,
        "model_path": saved_model_path,
    }


def compute_transition_entropy(probabilities: np.ndarray) -> float:
    """Computes Shannon Entropy (bits) across state probability vectors to quantify transition uncertainty."""
    eps = 1e-12
    p = np.clip(probabilities, eps, 1.0)
    entropy = -np.sum(p * np.log2(p), axis=1)
    return float(np.mean(entropy))


def run_soft_probability_hmm(
    labeled_df: pd.DataFrame,
    probability_columns: list[str] | None = None,
    n_components: int = 3,
    user_column: str = DEFAULT_USER_COLUMN,
    date_column: str = DEFAULT_DATE_COLUMN,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Fits continuous HMM directly on continuous GMM posterior probability distributions [prob_recovery, prob_baseline, prob_strain]
    without discarding uncertainty via hard integer discretization.
    """
    prob_cols = probability_columns or [f"prob_{state.lower().replace(' ', '_')}" for state in STANDARD_STATES]
    missing_cols = [c for c in prob_cols if c not in labeled_df.columns]
    if missing_cols:
        return {"fitted": False, "reason": f"Missing probability columns: {missing_cols}"}

    ordered = labeled_df.sort_values([user_column, date_column]).reset_index(drop=True)
    sequences: list[np.ndarray] = []
    lengths: list[int] = []
    for _, group in ordered.groupby(user_column, sort=False):
        prob_matrix = group[prob_cols].to_numpy(dtype=float)
        sequences.append(prob_matrix)
        lengths.append(len(prob_matrix))

    if not sequences:
        return {"fitted": False, "reason": "No sequences available."}

    X = np.vstack(sequences)
    model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=200, random_state=random_state, min_covar=1e-3)
    model.fit(X, lengths)

    states = model.predict(X, lengths)
    posteriors = model.predict_proba(X, lengths)
    entropy = compute_transition_entropy(posteriors)

    decoded = ordered.copy()
    decoded["soft_hmm_state"] = states.astype(int)
    decoded["soft_hmm_confidence"] = posteriors.max(axis=1)

    transmat_df = pd.DataFrame(model.transmat_, index=STANDARD_STATES[:n_components], columns=STANDARD_STATES[:n_components]).round(4)

    return {
        "fitted": True,
        "model": model,
        "decoded_df": decoded,
        "transition_matrix": transmat_df,
        "transition_entropy_bits": round(entropy, 4),
        "mean_confidence": round(float(posteriors.max(axis=1).mean()) * 100.0, 1),
    }


MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def save_hmm_model(model: Any, model_path: str | Path | None = None, model_name: str | Path | None = None) -> Path:
    """Save trained HMM model to disk."""
    target_path = Path(model_path or model_name or (MODEL_DIR / "hmm.pkl"))
    if not target_path.is_absolute():
        target_path = MODEL_DIR / target_path.name
    target_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target_path)
    return target_path



def load_hmm_model(model_path: str | Path | None = None) -> Any:
    """Load persisted HMM model from disk."""
    target_path = Path(model_path) if model_path is not None else MODEL_DIR / "hmm.pkl"
    return joblib.load(target_path)


def run_all_hmm_modes(df: pd.DataFrame, n_components: int = 3) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for observation_column in ["gmm_cluster", "kmeans_cluster"]:
        if observation_column in df.columns:
            results[observation_column] = run_hmm_pipeline(df, observation_column=observation_column, n_components=n_components)
    results["soft_probability_hmm"] = run_soft_probability_hmm(df, n_components=n_components)
    return results


