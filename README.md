<<<<<<< HEAD
# Wearable Health AI

A modular Python project for analyzing wearable device data with Streamlit and a separable ML pipeline.

## Project Structure

```text
wearable-health-ai/
├── app/
│   ├── main.py
│   └── pages/
│       ├── upload.py
│       ├── analysis.py
│       └── recommendations.py
├── data/
│   └── sample_user.csv
├── models/
│   ├── gmm.pkl
│   ├── hmm.pkl
│   └── kmeans.pkl
├── src/
│   ├── baseline.py
│   ├── clustering.py
│   ├── features.py
│   ├── hmm_model.py
│   ├── preprocessing.py
│   ├── recommendation_engine.py
│   └── risk_scoring.py
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

## Architecture

- `app/` contains the Streamlit UI and multipage workflow.
- `src/` contains reusable preprocessing, feature engineering, scoring, and recommendation logic.
- `data/` contains sample wearable input data.
- `models/` is reserved for trained serialized artifacts.

## Notes

- The `.pkl` files are placeholders and can be replaced with trained models.
- The current pipeline uses rule-based fallbacks when trained models are not present.
=======
# FINAL-YEAR-PROJECT
>>>>>>> 37ffc0fc018febe81e7d2fab47b2b943f552de55
