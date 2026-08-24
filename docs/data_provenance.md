# Dataset Provenance & Methodology Disclosure

## 1. Dataset Characteristics & Provenance
- **Type**: Synthetic Longitudinal Wearable Physiological Benchmark Dataset
- **Total Cohort Size**: 300 individual longitudinal user timelines
- **Observation Frequency**: 184 daily observations per user
- **Total Dataset Observations**: 55,200 total daily records
- **Temporal Coverage**: ~6 consecutive months (184 calendar days) per user timeline

---

## 2. Physiological Signals Schema
The dataset contains daily synthetic wearable physiological and lifestyle readings:
- `resting_hr_bpm`: Resting heart rate (beats per minute)
- `hrv_rmssd_ms`: Heart rate variability RMSSD (milliseconds)
- `spo2_avg_pct`: Blood oxygen saturation percentage (%)
- `sleep_duration_hours`: Nightly sleep duration (hours)
- `steps`: Daily step count
- `sbp_mmHg` / `dbp_mmHg`: Systolic and diastolic blood pressure (mmHg)
- `caffeine_mg` / `screen_time_min`: Daily lifestyle metrics

---

## 3. Ground Truth & Clinical Scope Disclosures
- **Clinical Ground Truth**: Absent. The dataset is fully synthetic and contains no real-patient medical diagnostic ground truth labels.
- **Model-Derived Latent States**: Health states (*Recovery*, *Baseline*, *Strain*) represent unsupervised model-derived clustering interpretations based on physiological deviations.
- **Methodological Scope**: The framework is presented strictly as a methodological benchmark study for longitudinal wearable analytics, state discovery, and probabilistic temporal modeling.
- **Non-Clinical Disclaimer**: This software and dataset do **NOT** provide medical diagnosis, disease prediction, or clinical validation.
