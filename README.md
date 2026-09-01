# Anxiety Level Prediction — Streamlit App

A machine-learning Streamlit application that predicts an **Anxiety Level (1-10)** from the supplied dataset.

## Dataset

- File: `enhanced_anxiety_dataset.csv`
- Rows: 11,000
- Columns: 19
- Target: `Anxiety Level (1-10)`
- Missing values: 0

## Model

The application uses:
- One-hot encoding for categorical variables
- A Random Forest regression model
- Prediction constrained to the 1–10 target range

Validation results on a held-out 20% test split:
- MAE: 0.816
- R²: 0.780
- Predictions within ±1 point: 65.4%

These metrics are validation results for this supplied dataset and do not establish clinical accuracy.

## Files

- `app.py` — Streamlit application
- `best_anxiety_model.pkl` — trained Random Forest model
- `preprocessor.pkl` — fitted preprocessing pipeline
- `enhanced_anxiety_dataset.csv` — source dataset
- `train_model.py` — reproducible training script
- `requirements.txt` — Python dependencies
- `runtime.txt` — Python 3.11 runtime

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

For Streamlit Community Cloud, use:
- Repository: your GitHub repository
- Branch: `main`
- Main file: `app.py`
- Python: `3.11`

The dependency file should be in the repository root with the entrypoint, which is the layout used here.
