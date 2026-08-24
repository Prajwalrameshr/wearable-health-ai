# Reproducibility Guide

## 1. Environment Requirements
- Python 3.10+
- Dependencies listed in `requirements.txt`:
  - `pandas`, `numpy`, `scikit-learn`, `hmmlearn`, `joblib`, `plotly`, `matplotlib`, `seaborn`, `streamlit`

---

## 2. Command Execution Sequence

### Step 1: Train & Serialize Leakage-Free Models
```bash
py train_and_save_models.py
```
This trains `scaler.pkl`, `kmeans.pkl`, `gmm.pkl`, and `hmm.pkl` strictly on the chronological training split (70%) and saves artifacts to `models/`.

### Step 2: Run Inference
```bash
py run_inference.py data/wearables_health_6mo_daily.csv --model gmm --output outputs/final_results.csv
```
Runs inference loading persisted models without refitting.

### Step 3: Run Research Experiment Suite & Paper Artifact Generator
```bash
py experiments/run_all.py
```
Executes baselines B0–B6, ablation study A–G, subject-independent 5-fold cross-validation, multi-seed statistical testing, robustness simulations, and exports Tables 1–9 and Figures 1–12 into `experiments/results/`.

### Step 4: Run Automated Tests
```bash
py -m pytest tests/
```

### Step 5: Launch Streamlit UI
```bash
py -m streamlit run app/main.py
```
