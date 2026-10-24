# FlowCast - AI-Powered Smart Traffic Flow Prediction Platform

Forecasts **traffic volume, travel time, congestion level and accident risk** 30-120 minutes ahead for the 25 segments of the
Northline Corridor, from sensor, weather and calendar data - and serves everything through a Streamlit dashboard.

## Quick start (one setup command + one run command)

```bash
python -m venv .venv                      # Python 3.11 recommended
source .venv/bin/activate                 # Windows:  .venv\Scripts\activate
pip install -r requirements.txt           # setup
python run_pipeline.py                    # raw CSVs -> cleaned data -> models -> reports (30-60 min, mostly Random Forest tuning)
streamlit run dashboard/app.py            # open the dashboard
pytest -q                                 # sanity tests (cleaning, leakage, gradients)
```

Quicker first look: `python run_pipeline.py --only clean`, then `--only eda`. Use `--skip dl` to leave the LSTM out.
Full beginner walkthrough: **STEP_BY_STEP_GUIDE.md**. Written results: **reports/FlowCast_Final_Report.md**.

## Layout

| Path | What it is | PRD module |
|---|---|---|
| `data/raw/` | the 3 source CSVs (read-only) | - |
| `src/ingest.py` | load + validate + quarantine bad rows | M1 |
| `src/clean.py` | dedup, outliers, imputation, harmonisation, merge, data-quality report | M2 |
| `src/features.py` | lags, rolling stats, encodings, time split, scaler | M3 |
| `src/eda.py` | figures + EDA summary | M4 |
| `src/ml_models.py` | NumPy gradient-descent regression, LR/DT/RF/XGBoost/SVM, tuning, conformal intervals | M5 |
| `src/dl_model.py` | LSTM / Bi-LSTM written from scratch in NumPy (hand-written BPTT + Adam) | M6 |
| `src/predict.py` | reload saved models and forecast (no retraining) | M7 |
| `dashboard/app.py` | the Streamlit dashboard (9 views + upload / retrain / report tools) | M7 |
| `models/` | saved models, scaler, conformal quantiles, model cards | - |
| `reports/` | data-quality report, EDA summary, scoreboards, figures, final report | - |
| `notebooks/` | 01 EDA, 02 cleaning, 03 modelling | - |
| `tests/` | pytest checks | - |
