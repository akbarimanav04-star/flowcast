# FlowCast - Step-by-Step Guide for Beginners

This guide takes you from "I just unzipped the project" to "I can run it, explain it and deploy it". For every step it says
**which app to use**, **what to type or click**, and **what you should see**. Follow the parts in order.

---

## Part A - Install the tools (one time, ~30 min)

| # | Tool | Where to get it | Why you need it |
|---|---|---|---|
| 1 | **Python 3.11** | python.org -> Downloads -> Python 3.11.x. **Windows: tick "Add python.exe to PATH"** on the first installer screen | runs all the code |
| 2 | **VS Code** | code.visualstudio.com | the editor where you read, run and debug everything |
| 3 | VS Code extensions: **Python**, **Jupyter**, **Pylance** (all by Microsoft) | VS Code -> left bar "Extensions" icon -> search -> Install | run `.py` files and the notebooks |
| 4 | **Git** | git-scm.com/downloads | version control (PRD requirement) |
| 5 | **GitHub account** | github.com -> Sign up | hosts your code; needed for deployment |
| 6 | **Streamlit Community Cloud account** | share.streamlit.io -> "Continue with GitHub" | free hosting of the dashboard |

Check in a terminal (VS Code -> menu *Terminal -> New Terminal*):
```
python --version      # should print 3.11.x  (on Mac/Linux you may need: python3 --version)
git --version
```

## Part B - Open the project and create the environment

1. Unzip `flowcast_project.zip` to a folder such as `Documents/flowcast`.
2. **VS Code -> File -> Open Folder...** -> choose the `flowcast` folder (the one that contains `run_pipeline.py`).
3. Open a terminal (*Terminal -> New Terminal*). Create an isolated environment:
   * Windows: `python -m venv .venv` then `.venv\Scripts\activate`
   * Mac/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`

   Your prompt now starts with `(.venv)`. If VS Code asks "use this environment?", click **Yes**.
4. Install the libraries: `pip install -r requirements.txt` (3-5 minutes).
5. Tell VS Code which Python to use: press `Ctrl+Shift+P` (Mac: `Cmd+Shift+P`) -> **Python: Select Interpreter** -> pick the one inside `.venv`.

## Part C - Run the project stage by stage

The whole project is one command (`python run_pipeline.py`), but run it in stages the first time so you can *look* at each result.
All commands are typed in the VS Code terminal with `(.venv)` active.

### Stage 1 (PRD week 1) - Clean the data   `python run_pipeline.py --only clean`   (~30 s)
* **What happens:** `src/ingest.py` loads the three CSVs and checks them; `src/clean.py` removes duplicates, fixes impossible values, fills gaps, normalises weather labels, aligns hourly weather to 30-minute windows and merges everything.
* **Look at:** `reports/data_quality.md` (every defect and how it was fixed), `data/processed/flowcast_clean.parquet` (the clean table), `data/interim/quarantine_traffic.csv` (rows that broke the rules - nothing is silently dropped).
* **Open in VS Code:** `notebooks/02_cleaning.ipynb` -> click **Select Kernel** (top right) -> choose the `.venv` Python -> **Run All**. It walks through each defect with before/after evidence.

### Stage 2 - Explore the data   `python run_pipeline.py --only eda`   (~20 s)
* **Look at:** the PNGs in `reports/figures/` starting with `eda_` and `reports/eda_summary.md`. Open `notebooks/01_eda.ipynb` and run it.
* **Be able to say:** two daily peaks, weekday vs weekend differ, volume and occupancy are strongly correlated, congestion classes are imbalanced (so we use macro-F1), rain/fog raise incident rate.

### Stage 3 (week 2) - Classical machine learning   `python run_pipeline.py --only ml`   (~20-40 min)
* **What happens** (`src/features.py`, `src/ml_models.py`): builds 50 features, splits **by time** (Jan-Mar train, Apr validation, May test), fits scalers on train only, writes a gradient-descent linear regression in NumPy, tunes Linear/Logistic Regression, Decision Tree, Random Forest, XGBoost and SVM with time-series cross-validation, picks champions using the validation set, then trains XGBoost models for +30/+60/+90/+120 minutes with 90 % uncertainty bands.
* **It is slow because** it runs on one CPU core on purpose (reproducible). Let it run; go and read `src/features.py` meanwhile.
* **Look at:** `reports/scoreboard_classical.csv` (open with Excel or the VS Code "Data Wrangler"/CSV viewer), `reports/figures/fi_volume.png` (feature importance), `confusion_congestion.png`, `roc_accident.png`, `gd_loss_curve.png`.

### Stage 4 (week 3) - Deep learning   `python run_pipeline.py --only dl`   (~5-15 min)
* **What happens** (`src/dl_model.py`): an LSTM and a Bi-LSTM, written from scratch with NumPy (forward pass, back-propagation through time, Adam optimiser, dropout, early stopping). It predicts next-window volume and congestion together (multi-task).
* **Look at:** `reports/figures/dl_loss_curves.png` (training vs validation loss; if validation rises while training keeps falling that is over-fitting. Training loss sitting slightly *above* validation is normal here because dropout is active only during training), `reports/scoreboard_dl.csv`, and the head-to-head table in the report.

### Stage 5 - Model cards and the final report   `python run_pipeline.py --only cards` then `--only report`
* Creates `models/model_cards/*.json` (training window, features, hyper-parameters, metrics for each model) and `reports/FlowCast_Final_Report.md`.
* Open the `.md` report in VS Code and press `Ctrl+Shift+V` to see it formatted.

### Stage 6 (week 4) - Dashboard   `streamlit run dashboard/app.py`
A browser tab opens at `http://localhost:8501`. Use the left menu: Live prediction -> Congestion heatmap -> Road comparison -> Historical trends -> Forecast visualisation -> Prediction confidence -> Weather vs traffic -> Model performance -> Feature importance, plus three tools (upload & predict, fast retrain, report export). Stop it with `Ctrl+C` in the terminal.

### Stage 7 - Tests   `pytest -q`
Expect `9 passed`. These prove: no nulls/duplicates after cleaning, no data leakage, the split is chronological, scalers are fitted on train only, and the gradient-descent gradients are correct.

> **Run everything in one go:** `python run_pipeline.py`. Re-running reproduces the same numbers because every random seed is fixed in `config.yaml`.

## Part D - Understand the code (read in this order)

1. `config.yaml` - every setting in one place (paths, split dates, window sizes, DL hyper-parameters).
2. `src/ingest.py` -> `src/clean.py` - how the 8 known data problems are handled:

| Problem in the Data Dictionary | Where it is fixed |
|---|---|
| Missing sensor windows | `clean_traffic`: full 30-min grid re-created, short gaps interpolated, long gaps = road x weekday x slot median |
| Duplicate rows | `clean_traffic` step 2 |
| Impossible values (negative, >200 km/h, >100 %) | `clean_traffic` step 3 (rules + IQR fences) |
| Hourly weather vs 30-min traffic | `merge_all` (both half-hours share the hour's weather) |
| Weather labels `Rain/rainy/RAIN` | `clean_weather` + `WEATHER_MAP` |
| Mixed date formats | `ingest._to_timestamp` (explicit formats) |
| Blank congestion level | `derive_congestion` (volume / (capacity/2) bands) |
| Nested JSON `vehicle_type_dist` | parsed into four share columns |

3. `src/features.py` - lags (t-1, t-2, t-48), rolling mean/std (4 and 8 windows), sin/cos hour and weekday, peak/weekend/holiday/event flags, volume-to-capacity ratio, weather flags. Notice `build_supervised`: features come from time *t*, targets from *t + h* (the leakage guard).
4. `src/ml_models.py` - start with class `LinearRegressionGD` (the maths: cost, gradient, update), then `model_zoo`, then `train_all`.
5. `src/dl_model.py` - class `LSTMForecaster`: `_lstm_forward` and `_lstm_backward` are the LSTM equations written out by hand.
6. `src/predict.py` and `dashboard/app.py` - how saved models are reloaded to make forecasts.

**Maths -> code map (useful for viva):** matrices X and Xw -> `LinearRegressionGD.fit`; gradient descent and MSE cost -> `cost_and_grads`; z-score and min-max scaling -> `HybridScaler` in `src/utils.py`; correlation/covariance -> `src/eda.py`; probability and ROC-AUC -> accident model + `evaluate.risk_metrics`; confusion matrix, precision, recall, F1 -> `evaluate.py`.

## Part E - Put it on GitHub and deploy the dashboard

1. **Create the repo:** github.com -> **New repository** -> name `flowcast` -> Public -> *do not* add README -> Create.
2. **Push from VS Code's terminal** (replace `YOUR-NAME`):
```
git init
git add .
git commit -m "FlowCast: complete pipeline, models and dashboard"
git branch -M main
git remote add origin https://github.com/YOUR-NAME/flowcast.git
git push -u origin main
```
   (First push asks you to sign in to GitHub in the browser.) The dashboard needs `data/processed/`, `models/` and `reports/` - they are **not** git-ignored, so they upload with the code. GitHub refuses any single file above 100 MB; the largest file here (the Random Forest model) is about 40 MB, so it is fine.
3. **Deploy:** share.streamlit.io -> **Create app** -> *Deploy a public app from GitHub* -> pick repo `flowcast`, branch `main`, main file path `dashboard/app.py` -> **Advanced settings -> Python 3.11** -> **Deploy**. First build takes a few minutes; you get a public URL to put on your CV.
4. After later changes: `git add . && git commit -m "update" && git push` - the app redeploys automatically.

## Part F - How to talk about the results honestly

* **Volume forecast** meets its target (MAPE <= 12 %) and gradient boosting is the best classical model.
* **Congestion macro-F1** and **accident ROC-AUC** are below the PRD's stretch targets. This is a *finding*, not a bug: all models plateau at the same score, and the report shows (a) the errors live at the Heavy/Severe threshold and (b) a same-window "oracle" model reaches only about 0.63 AUC for accidents, so 0.75 cannot be achieved from this data. Saying this clearly (with the evidence) is stronger than hiding it.
* **Deep vs classical:** read the verdict printed in section 7 of the report; the comparison uses the identical test window and metrics.
* **Never tune on the test month.** Champions are chosen on the validation month.

## Part G - Troubleshooting

| Problem | Fix |
|---|---|
| `python` not found | reinstall Python and tick *Add to PATH*; on Mac/Linux use `python3` |
| `ModuleNotFoundError` | the venv is not active: run the `activate` command again, then `pip install -r requirements.txt` |
| `streamlit` not recognised | same - activate the venv first |
| Dashboard says "Model outputs not found" | run `python run_pipeline.py` (at least `clean`, `ml`, `dl`) first |
| Notebook asks for a kernel | top right **Select Kernel** -> Python Environments -> `.venv` |
| Pipeline too slow | reduce `n_estimators` of Random Forest in `src/ml_models.py` `model_zoo`, or `--skip dl`; 8 GB RAM is enough |
| Map is blank in the dashboard | it loads OpenStreetMap tiles - check your internet connection |

## Part H - Make it your own

* Change split dates, sequence length, LSTM size or learning rate in `config.yaml` and re-run the stage.
* Use your own traffic file: upload it in the dashboard tool *Upload & predict* (same columns as `traffic_sensor_log.csv`, at least two days of history).
* Ideas from the PRD roadmap: attention/transformer models, graph models over road neighbours, a live weather-forecast feed, drift monitoring.
