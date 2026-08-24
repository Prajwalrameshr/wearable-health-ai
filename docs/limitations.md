# Methodological Limitations

1. **Synthetic Data Benchmark**:
   - The evaluation is conducted exclusively on synthetic longitudinal wearable data.
   - Real-world physiological signals contain complex artifacts, sensor disconnections, and non-stationary baseline drift not fully reflected in synthetic distributions.

2. **Absence of Ground-Truth Clinical Diagnostic Labels**:
   - The latent states ("Recovery", "Baseline", "Strain") are human-readable interpretations derived from unsupervised Gaussian component means and severity scores.
   - These labels are not verified against clinical diagnostic markers (e.g., blood biomarkers, ECG, PSG).

3. **Heuristic Nature of Physiological Strain Index**:
   - The composite physiological strain index combines normalized deviation z-scores using heuristic scaling coefficients.
   - It serves as a continuous physiological monitoring index rather than a clinically calibrated diagnostic risk score.

4. **Temporal Windowing**:
   - The personalized baseline utilizes a fixed 7-day rolling window. Certain physiological dynamics (e.g., menstrual cycles, seasonal changes) operate on multi-week or multi-month scales requiring longer historical baselines.
