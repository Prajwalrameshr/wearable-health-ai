from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DATASET_ALIASES = [
    "dwearables.csv",
    "wearables_hea;th_6mo_daily.csv",
    "wearables_health_6mo_daily.csv",
]

DAILY_REQUIRED_COLUMNS = [
    "user_id",
    "date",
    "age",
    "gender",
    "region",
    "device_model",
    "height_cm",
    "weight_kg",
    "bmi",
    "resting_hr_bpm",
    "avg_hr_day_bpm",
    "hrv_rmssd_ms",
    "spo2_avg_pct",
    "sbp_mmHg",
    "dbp_mmHg",
    "steps",
    "distance_km",
    "calories_kcal",
    "workout_type",
    "workout_minutes",
    "caffeine_mg",
    "alcohol_units",
    "screen_time_min",
    "sleep_duration_hours",
    "sleep_efficiency",
    "sleep_latency_min",
    "wake_after_sleep_onset_min",
    "sleep_stage_rem_pct",
    "sleep_stage_deep_pct",
    "sleep_stage_light_pct",
    "stress_score",
    "mindfulness_minutes",
    "mood",
]

MINIMAL_DAILY_COLUMNS = [
    "user_id",
    "date",
    "resting_hr_bpm",
    "hrv_rmssd_ms",
    "spo2_avg_pct",
    "steps",
    "sleep_duration_hours",
]

OPTIONAL_DAILY_DEFAULTS = {
    "age": 35,
    "gender": "unknown",
    "region": "unknown",
    "device_model": "generic",
    "height_cm": 170.0,
    "weight_kg": 70.0,
    "bmi": 24.2,
    "avg_hr_day_bpm": np.nan,
    "sbp_mmHg": 120.0,
    "dbp_mmHg": 80.0,
    "distance_km": 0.0,
    "calories_kcal": 0.0,
    "workout_type": "unknown",
    "workout_minutes": 0.0,
    "caffeine_mg": 0.0,
    "alcohol_units": 0.0,
    "screen_time_min": 0.0,
    "sleep_efficiency": 85.0,
    "sleep_latency_min": 0.0,
    "wake_after_sleep_onset_min": 0.0,
    "sleep_stage_rem_pct": 20.0,
    "sleep_stage_deep_pct": 15.0,
    "sleep_stage_light_pct": 65.0,
    "stress_score": 0.0,
    "mindfulness_minutes": 0.0,
    "mood": "unknown",
}

LEGACY_REQUIRED_COLUMNS = [
    "timestamp",
    "heart_rate",
    "steps",
    "sleep_hours",
    "spo2",
    "stress_level",
]

PHYSIOLOGICAL_COLUMNS = [
    "resting_hr_bpm",
    "avg_hr_day_bpm",
    "hrv_rmssd_ms",
    "spo2_avg_pct",
    "sbp_mmHg",
    "dbp_mmHg",
    "sleep_duration_hours",
    "steps",
]

BASELINE_COLUMN_MAP = {
    "baseline_hr": "resting_hr_bpm",
    "baseline_hrv": "hrv_rmssd_ms",
    "baseline_spo2": "spo2_avg_pct",
    "baseline_sleep": "sleep_duration_hours",
    "baseline_steps": "steps",
    "baseline_sbp": "sbp_mmHg",
    "baseline_dbp": "dbp_mmHg",
}

STD_COLUMN_MAP = {
    "std_hr": "resting_hr_bpm",
    "std_hrv": "hrv_rmssd_ms",
    "std_spo2": "spo2_avg_pct",
    "std_sleep": "sleep_duration_hours",
    "std_steps": "steps",
    "std_sbp": "sbp_mmHg",
    "std_dbp": "dbp_mmHg",
}

DEVIATION_COLUMN_MAP = {
    "hr_dev": ("resting_hr_bpm", "baseline_hr", "std_hr"),
    "hrv_dev": ("hrv_rmssd_ms", "baseline_hrv", "std_hrv"),
    "spo2_dev": ("spo2_avg_pct", "baseline_spo2", "std_spo2"),
    "sleep_dev": ("sleep_duration_hours", "baseline_sleep", "std_sleep"),
    "steps_dev": ("steps", "baseline_steps", "std_steps"),
    "sbp_dev": ("sbp_mmHg", "baseline_sbp", "std_sbp"),
    "dbp_dev": ("dbp_mmHg", "baseline_dbp", "std_dbp"),
}

SLOPE_INPUT_MAP = {
    "hr_dev_slope_7d": "hr_dev",
    "hrv_dev_slope_7d": "hrv_dev",
    "sleep_dev_slope_7d": "sleep_dev",
    "steps_dev_slope_7d": "steps_dev",
}

CLUSTER_FEATURE_COLUMNS = [
    "hr_dev",
    "hrv_dev",
    "sleep_dev",
    "severity_score",
]

RECOMMENDATION_FEATURE_COLUMNS = [
    "steps",
    "spo2_avg_pct",
    "caffeine_mg",
    "screen_time_min",
]

EARLY_WARNING_FEATURE_COLUMNS = [
    "hrv_dev_slope_7d",
    "sleep_dev_slope_7d",
]

HMM_FEATURE_COLUMNS = [
    "hr_dev_z",
    "hrv_dev_z",
    "spo2_dev_z",
    "sleep_dev_z",
    "steps_dev_z",
    "sbp_dev_z",
    "dbp_dev_z",
    "hr_dev_slope_7d",
    "hrv_dev_slope_7d",
    "sleep_dev_slope_7d",
    "steps_dev_slope_7d",
    "severity_score",
    "trend_score",
]

ROLLING_WINDOW = 7
CATEGORICAL_COLUMNS = ["workout_type", "mood"]
USER_CONTEXT_CATEGORICAL_COLUMNS = ["gender", "region", "device_model"]


def resolve_dataset_path(
    source: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> Path:
    """Resolve the daily wearable dataset path across known filename variants."""
    if source is not None:
        source_path = Path(source)
        if source_path.exists():
            return source_path

    base_dir = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parents[1] / "data"
    for alias in DATASET_ALIASES:
        candidate = base_dir / alias
        if candidate.exists():
            return candidate

    checked = ", ".join(DATASET_ALIASES)
    raise FileNotFoundError(f"Dataset not found. Checked: {checked}")


def is_daily_wearable_dataset(df: pd.DataFrame) -> bool:
    """Identify the daily wearable dataset using the minimal physiological schema."""
    return set(MINIMAL_DAILY_COLUMNS).issubset(df.columns)


def validate_required_columns(df: pd.DataFrame) -> list[str]:
    """Allow either the full daily schema or the minimal physiological schema."""
    if set(MINIMAL_DAILY_COLUMNS).issubset(df.columns):
        return []
    if set(LEGACY_REQUIRED_COLUMNS).issubset(df.columns):
        return []
    if {"user_id", "date"}.issubset(df.columns):
        return [column for column in MINIMAL_DAILY_COLUMNS if column not in df.columns]
    return [column for column in LEGACY_REQUIRED_COLUMNS if column not in df.columns]


def ensure_daily_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Fill optional daily columns so the pipeline can run on minimal uploads."""
    prepared = df.copy()
    missing_minimal = [column for column in MINIMAL_DAILY_COLUMNS if column not in prepared.columns]
    if missing_minimal:
        missing = ", ".join(missing_minimal)
        raise ValueError(f"Daily wearable dataset is missing required columns: {missing}")

    for column, default_value in OPTIONAL_DAILY_DEFAULTS.items():
        if column not in prepared.columns:
            prepared[column] = default_value

    if "avg_hr_day_bpm" in prepared.columns:
        prepared["avg_hr_day_bpm"] = prepared["avg_hr_day_bpm"].fillna(prepared["resting_hr_bpm"])

    for column in DAILY_REQUIRED_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = OPTIONAL_DAILY_DEFAULTS.get(column, np.nan)

    return prepared


def load_user_data(source) -> pd.DataFrame:
    """Load a CSV from a path-like object or file-like upload object."""
    return pd.read_csv(source)


def get_daily_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Collect numeric columns excluding the date field."""
    return [
        column
        for column in df.columns
        if column != "date" and pd.api.types.is_numeric_dtype(df[column])
    ]


def load_data(source: str | Path | None = None, data_dir: str | Path | None = None) -> pd.DataFrame:
    """Load, type-cast, sort, and deduplicate the daily wearable dataset."""
    dataset_path = resolve_dataset_path(source=source, data_dir=data_dir)
    df = pd.read_csv(dataset_path)
    df = ensure_daily_schema(df)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    non_numeric_columns = {"user_id", "date", "gender", "region", "device_model", "workout_type", "mood"}
    for column in DAILY_REQUIRED_COLUMNS:
        if column in non_numeric_columns or column not in df.columns:
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_values(["user_id", "date"]).drop_duplicates(subset=["user_id", "date"], keep="last")
    return df.reset_index(drop=True)


def handle_missing(
    df: pd.DataFrame,
    interpolation_limit: int = 2,
    categorical_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Fill short gaps per user using ffill, interpolation, and per-user categorical mode."""
    filled = df.copy()
    categorical_targets = list(CATEGORICAL_COLUMNS)
    categorical_targets.extend(USER_CONTEXT_CATEGORICAL_COLUMNS)
    if categorical_columns is not None:
        categorical_targets = categorical_columns

    numeric_columns = get_daily_numeric_columns(filled)

    filled[numeric_columns] = filled.groupby("user_id", group_keys=False)[numeric_columns].apply(
        lambda group: group.ffill(limit=interpolation_limit)
    )
    filled[numeric_columns] = filled.groupby("user_id", group_keys=False)[numeric_columns].apply(
        lambda group: group.interpolate(method="linear", limit_direction="forward", limit=interpolation_limit)
    )
    filled[numeric_columns] = filled.groupby("user_id", group_keys=False)[numeric_columns].apply(
        lambda group: group.bfill(limit=1)
    )

    for column in categorical_targets:
        if column not in filled.columns:
            continue

        def fill_mode(series: pd.Series) -> pd.Series:
            mode = series.mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "unknown"
            return series.fillna(fill_value)

        filled[column] = filled.groupby("user_id")[column].transform(fill_mode).fillna("unknown")

    return filled


def remove_outliers(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    whisker_width: float = 1.5,
) -> pd.DataFrame:
    """Cap extreme per-user physiological outliers using the IQR rule."""
    capped = df.copy()
    target_columns = PHYSIOLOGICAL_COLUMNS if columns is None else columns

    for column in target_columns:
        if column not in capped.columns:
            continue

        def cap_group(series: pd.Series) -> pd.Series:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                return series
            lower = q1 - whisker_width * iqr
            upper = q3 + whisker_width * iqr
            return series.clip(lower=lower, upper=upper)

        capped[column] = capped.groupby("user_id")[column].transform(cap_group)

    return capped


def compute_baseline(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Compute rolling baselines and rolling standard deviations per user."""
    baseline_df = df.copy()

    for baseline_name, source_column in BASELINE_COLUMN_MAP.items():
        baseline_df[baseline_name] = baseline_df.groupby("user_id")[source_column].transform(
            lambda series: series.rolling(window=window, min_periods=1).mean()
        )

    for std_name, source_column in STD_COLUMN_MAP.items():
        baseline_df[std_name] = baseline_df.groupby("user_id")[source_column].transform(
            lambda series: series.rolling(window=window, min_periods=1).std(ddof=0).fillna(0.0)
        )

    return baseline_df


def compute_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """Create raw deviations and z-normalized deviation features from personalized baselines."""
    deviation_df = df.copy()

    for deviation_name, (source_column, baseline_column, std_column) in DEVIATION_COLUMN_MAP.items():
        deviation_df[deviation_name] = deviation_df[source_column] - deviation_df[baseline_column]
        safe_std = deviation_df[std_column].replace(0, np.nan)
        deviation_df[f"{deviation_name}_z"] = (deviation_df[deviation_name] / safe_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return deviation_df


def rolling_slope(series: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    """Compute an ordered rolling slope via linear convolution weights."""
    values = series.to_numpy(dtype=float)
    slopes = np.zeros(len(values), dtype=float)

    for index in range(len(values)):
        start = max(0, index - window + 1)
        window_values = values[start : index + 1]
        if len(window_values) < 2 or np.isnan(window_values).any():
            slopes[index] = 0.0
            continue
        x_values = np.arange(len(window_values), dtype=float)
        slopes[index] = float(np.polyfit(x_values, window_values, 1)[0])

    return pd.Series(slopes, index=series.index)


def compute_temporal_features(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Add 7-day rolling slopes and rolling HRV coefficient of variation."""
    temporal_df = df.copy()

    for slope_column, source_column in SLOPE_INPUT_MAP.items():
        temporal_df[slope_column] = temporal_df.groupby("user_id")[source_column].transform(
            lambda series: rolling_slope(series, window=window)
        )

    hrv_mean = temporal_df.groupby("user_id")["hrv_rmssd_ms"].transform(
        lambda series: series.rolling(window=window, min_periods=1).mean()
    )
    hrv_std = temporal_df.groupby("user_id")["hrv_rmssd_ms"].transform(
        lambda series: series.rolling(window=window, min_periods=1).std(ddof=0).fillna(0.0)
    )
    temporal_df["hrv_cv_7d"] = (hrv_std / hrv_mean.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return temporal_df


def compute_severity(df: pd.DataFrame) -> pd.DataFrame:
    """Create composite severity and trend scores, then derive anomaly flags."""
    scored = df.copy()
    deviation_z_columns = [f"{name}_z" for name in DEVIATION_COLUMN_MAP]
    slope_columns = list(SLOPE_INPUT_MAP.keys())

    scored["severity_score"] = scored[deviation_z_columns].abs().sum(axis=1)
    scored["trend_score"] = scored[slope_columns].abs().sum(axis=1)

    severity_mean = scored.groupby("user_id")["severity_score"].transform("mean")
    severity_std = scored.groupby("user_id")["severity_score"].transform("std").fillna(0.0)
    threshold = severity_mean + (3 * severity_std)
    scored["anomaly_flag"] = (scored["severity_score"] > threshold).astype(int)

    return scored


def build_recommendation_context(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Select recommendation-only context variables after feature creation."""
    selected_columns = RECOMMENDATION_FEATURE_COLUMNS if feature_columns is None else feature_columns
    missing_columns = [column for column in selected_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing recommendation feature columns: {missing}")

def preprocess_for_modeling(
    source: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> dict:
    """Run the full preprocessing pipeline for KMeans, GMM, and HMM."""
    cleaned_df = load_data(source=source, data_dir=data_dir)
    cleaned_df = handle_missing(cleaned_df)
    cleaned_df = remove_outliers(cleaned_df)

    feature_df = compute_baseline(cleaned_df)
    feature_df = compute_deviation(feature_df)
    feature_df = compute_temporal_features(feature_df)
    feature_df = compute_severity(feature_df)
    feature_df, encoded_columns = encode_features(feature_df)

    scaled_df, scaler, cluster_features = scale_features(feature_df)
    hmm_sequences, hmm_lengths, hmm_features = build_hmm_sequences(feature_df)
    summary_df = summarize_preprocessed_data(feature_df)

    return {
        "cleaned_df": cleaned_df,
        "feature_df": feature_df,
        "scaled_feature_matrix": scaled_df,
        "hmm_sequences": hmm_sequences,
        "hmm_lengths": hmm_lengths,
        "cluster_feature_columns": cluster_features,
        "hmm_feature_columns": hmm_features,
        "encoded_feature_columns": encoded_columns,
        "summary_df": summary_df,
        "preview_df": feature_df.head(),
        "scaler": scaler,
    }


def preprocess_daily_wearables(
    source: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compatibility wrapper returning final dataframe, preview, and summary."""
    outputs = preprocess_for_modeling(source=source, data_dir=data_dir)
    return outputs["feature_df"], outputs["preview_df"], outputs["summary_df"]


def clean_wearable_data(df: pd.DataFrame) -> pd.DataFrame:
    """Retain legacy cleaning for the small demo dataset and route daily data to the full pipeline."""
    if is_daily_wearable_dataset(df):
        prepared = ensure_daily_schema(df)
        prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
        prepared = prepared.dropna(subset=["date"]).sort_values(["user_id", "date"]).reset_index(drop=True)
        non_numeric_columns = {"user_id", "date", "gender", "region", "device_model", "workout_type", "mood"}
        for column in DAILY_REQUIRED_COLUMNS:
            if column in prepared.columns and column not in non_numeric_columns:
                prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        prepared = handle_missing(prepared)
        prepared = remove_outliers(prepared)
        prepared = compute_baseline(prepared)
        prepared = compute_deviation(prepared)
        prepared = compute_temporal_features(prepared)
        prepared = compute_severity(prepared)
        prepared, _ = encode_features(prepared)
        return prepared



import joblib

def save_scaler(scaler: StandardScaler, filepath: str | Path) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)

def load_scaler(filepath: str | Path) -> StandardScaler:
    return joblib.load(filepath)

def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values(["user_id", "date"]).reset_index(drop=True)
    train_list, val_list, test_list = [], [], []

    for _, user_df in ordered.groupby("user_id", sort=False):
        n = len(user_df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_list.append(user_df.iloc[:train_end])
        val_list.append(user_df.iloc[train_end:val_end])
        test_list.append(user_df.iloc[val_end:])

    train_df = pd.concat(train_list, ignore_index=True) if train_list else pd.DataFrame()
    val_df = pd.concat(val_list, ignore_index=True) if val_list else pd.DataFrame()
    test_df = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame()

    return train_df, val_df, test_df

def encode_features(df: pd.DataFrame, categorical_columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    encoded = df.copy()
    columns_to_encode = CATEGORICAL_COLUMNS if categorical_columns is None else categorical_columns
    dummy_frame = pd.get_dummies(encoded[columns_to_encode], prefix=columns_to_encode, dtype=int)
    encoded = pd.concat([encoded, dummy_frame], axis=1)
    return encoded, dummy_frame.columns.tolist()

def scale_features(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    scaler: StandardScaler | None = None,
) -> tuple[pd.DataFrame, StandardScaler, list[str]]:
    selected_columns = CLUSTER_FEATURE_COLUMNS if feature_columns is None else feature_columns
    if scaler is None:
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(df[selected_columns])
    else:
        scaled_values = scaler.transform(df[selected_columns])
    scaled_df = pd.DataFrame(scaled_values, columns=selected_columns, index=df.index)
    return scaled_df, scaler, selected_columns

def build_hmm_sequences(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> tuple[list[np.ndarray], list[int], list[str]]:
    selected_columns = HMM_FEATURE_COLUMNS if feature_columns is None else feature_columns
    ordered = df.sort_values(["user_id", "date"]).reset_index(drop=True)

    sequences: list[np.ndarray] = []
    lengths: list[int] = []
    for _, group in ordered.groupby("user_id", sort=False):
        values = group[selected_columns].to_numpy(dtype=float)
        sequences.append(values)
        lengths.append(len(values))

    return sequences, lengths, selected_columns

def summarize_preprocessed_data(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.describe().transpose()
