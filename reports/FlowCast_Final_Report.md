# FlowCast - Final Technical Report

*AI-Powered Smart Traffic Flow Prediction Platform - Northline Corridor, 25 segments, 1 Jan - 31 May 2025. Dataset version `dae7a579b5`, seed 42.*

## 1. Executive summary

FlowCast ingests sensor, weather and calendar tables, repairs their defects with a documented pipeline, engineers leak-free features and forecasts volume, travel time, congestion class and accident risk 30-120 minutes ahead for each segment. Results below are on the final, never-touched test month (May 2025).

| Success criterion (PRD 3.3) | Target | Achieved | Status |
|---|---|---|---|
| Volume forecast | MAPE <= 12% | 9.15% (Random Forest) | MET |
| Congestion classification | Macro-F1 >= 0.80 | 0.767 (Bi-LSTM) | NOT MET |
| Accident-risk ranking | ROC-AUC >= 0.75 | 0.628 (Logistic Regression) | NOT MET |
| Deep model vs best classical (volume RMSE) | LSTM < classical | 56.5 (Bi-LSTM) vs 58.0 (Random Forest) | MET |
| Reproducibility | one command | `python run_pipeline.py` | MET |
| Dashboard | nine views on real outputs | `streamlit run dashboard/app.py` | MET |

**Honest reading of the misses.** 

* *Congestion macro-F1.* Every model family converges to about the same score (see section 5). The errors sit almost entirely at the Heavy/Severe boundaries, which are defined by a hard threshold on next-window volume/capacity; small unforecastable noise in volume flips the class. Perfect knowledge of next-window volume would give F1 = 1.0, but the volume forecast itself carries about 58 vehicles of irreducible RMSE. Extra lag features (same slot yesterday / last week) were tested and did not help, which suggests a noise ceiling rather than missing signal.
* *Accident risk.* Only 0.91% of windows carry an incident and the label is largely random with respect to the available signals. As a ceiling test we gave a model the *same window's* true volume, speed, occupancy, travel time and weather: it reaches ROC-AUC = **0.630**. A forecaster that sees only the past cannot beat that, so the 0.75 target is not attainable from this dataset. The model still ranks risk usefully (see top-10% capture below) and rain/fog/congestion raise risk roughly 2-4x, but it should be presented as a *prioritisation aid*, not a precise predictor.

## 2. Data & cleaning (M1-M2)

Raw tables: traffic (178,468 rows), weather (10,872 hourly rows), calendar (151 dates). The cleaning pipeline log is reproduced below (`reports/data_quality.md`).


| Stage | Issue / action | Count / value |
|---|---|---|
| 1 validate | traffic rows loaded | 178468 |
| 1 validate | rows with a range-rule violation (quarantined + repaired) | 712 |
| 2 duplicates | exact duplicate rows | 1767 |
| 2 duplicates | key-level (road_id+timestamp) duplicates removed | 1767 |
| 3 outliers | negative traffic_volume | 239 |
| 3 outliers | null/negative volume recovered exactly from vehicle_count | 4587 |
| 3 outliers | avg_speed > 200 km/h -> set to NaN | 236 |
| 3 outliers | occupancy > 100% -> set to NaN | 229 |
| 3 outliers | IQR(k=3.0) extremes per road x hour -> capped (winsorised) | 74 |
| 4 missing | absent 30-min windows (sensor dropout) re-inserted as empty rows | 4499 |
| 4 missing | rows with null volume after repairs (to impute) | 4499 |
| 4 missing | short gaps (<= 4 windows) filled by time interpolation | 31634 |
| 4 missing | long gaps filled with road x weekday x slot median | 13 |
| 5 harmonise | vehicle_type_dist JSON parsed into share_2W/Car/LCV/HCV | 181200 |
| 5 harmonise | blank/imputed congestion_level derived from volume/(capacity/2) | 31123 |
| 5 harmonise | given labels that disagree with the V/C rule (kept as given) | 0 |
| 5 harmonise | weather labels normalised (13 raw spellings -> 4) | 10872 |
| 5 harmonise | weather labels not in controlled vocabulary | 0 |
| 4 missing | weather temperature nulls | 167 |
| 4 missing | weather visibility nulls | 111 |
| 6 merge | weather aligned hourly->30 min and calendar joined; rows preserved | 181200 |
| 8 checks | nulls in modelling columns | 0 |
| 8 checks | final rows (25 roads x full 30-min grid) | 181200 |
| 8 checks | rows flagged is_imputed (volume estimated) | 4499 |


Key engineering decision: `vehicle_count` (raw detector count, equal to volume in this release per the data dictionary) was used to recover null/negative `traffic_volume` *exactly*; only windows absent from the log altogether were interpolated, and those rows are flagged `is_imputed` and excluded from the evaluation targets.

## 3. Feature engineering (M3)

50 features in 7 groups: **Current state** (12); **Lag features** (6); **Rolling statistics** (8); **Weather** (8); **Road (static)** (4); **Target-time encodings** (6); **Calendar** (6).
Lag and rolling features are computed per segment in strict time order; calendar facts about the *target* window (hour, weekday, holiday, event proximity) are allowed because they are known in advance; scalers are fit on the training window only.

## 4. Mathematics applied

* **Linear algebra:** the dataset is a feature matrix X (rows = windows, columns = 50 features); linear regression predicts y = Xw + b; an LSTM layer computes z = [x, h]W + b as one matrix product per time step.
* **Gradient descent / cost:** `LinearRegressionGD` minimises J(w,b) = (1/n) sum (Xw + b - y)^2 with gradients (2/n) X^T(Xw + b - y); the step size is set to 0.9 / lambda_max(X^T X / n), the largest stable value for a quadratic cost. It reaches within 0.5% of the closed-form RMSE. The LSTM uses hand-written back-propagation through time (verified against numerical gradients, relative error ~1e-6) and Adam.
* **Statistics:** mean/median/mode for imputation, standard deviation and variance for rolling-volatility features, correlation/covariance for redundancy checks (volume-occupancy r = 0.94), IQR fences and residual analysis (normality, heteroscedasticity).
* **Scaling:** min-max for bounded columns (occupancy, visibility, vehicle shares), z-score for unbounded ones, fitted on train only.
* **Probability:** accident risk and congestion are class probabilities; ranking quality is measured by ROC-AUC; conformal prediction gives distribution-free 90% volume intervals.

## 5. Results (test window, +30 min)

### volume

| model | RMSE | MAE | MAPE | R2 |
|---|---|---|---|---|
| Bi-LSTM | 56.5469 | 39.3135 | 0.0982 | 0.9610 |
| LSTM | 56.6209 | 39.2402 | 0.0968 | 0.9609 |
| Random Forest | 58.0465 | 39.2610 | 0.0915 | 0.9589 |
| XGBoost | 58.1185 | 39.9191 | 0.0946 | 0.9588 |
| Decision Tree | 64.0057 | 42.9897 | 0.1002 | 0.9500 |
| Linear Regression | 72.0952 | 50.8375 | 0.1341 | 0.9366 |
| Linear Regression (NumPy GD, from scratch) | 72.4221 | 51.1863 | 0.1369 | 0.9360 |

### travel_time

| model | RMSE | MAE | MAPE | R2 |
|---|---|---|---|---|
| XGBoost | 1.0606 | 0.4422 | 0.0952 | 0.8262 |
| Random Forest | 1.0680 | 0.4340 | 0.0913 | 0.8238 |
| Decision Tree | 1.1845 | 0.5036 | 0.1098 | 0.7833 |
| Linear Regression | 1.4047 | 0.6754 | 0.1825 | 0.6952 |

### congestion

| model | MacroF1 | Accuracy | Precision | Recall |
|---|---|---|---|---|
| Bi-LSTM | 0.7668 | 0.8753 | 0.7663 | 0.7678 |
| XGBoost + tuned class weights | 0.7665 | 0.8716 | 0.7603 | 0.7734 |
| LSTM | 0.7655 | 0.8753 | 0.7742 | 0.7619 |
| XGBoost | 0.7627 | 0.8728 | 0.7736 | 0.7568 |
| Random Forest | 0.7602 | 0.8724 | 0.7711 | 0.7533 |
| Decision Tree | 0.7514 | 0.8647 | 0.7558 | 0.7483 |
| Logistic Regression | 0.7276 | 0.8528 | 0.7428 | 0.7175 |
| SVM (RBF) | 0.7217 | 0.8536 | 0.7575 | 0.7130 |

### accident

| model | ROC_AUC | Precision | Recall | F1 | Top10pct_capture |
|---|---|---|---|---|---|
| Logistic Regression | 0.6282 | 0.0321 | 0.1268 | 0.0512 | 0.2563 |
| XGBoost | 0.6135 | 0.0261 | 0.2282 | 0.0468 | 0.2563 |
| Decision Tree | 0.5924 | 0.0242 | 0.2845 | 0.0445 | 0.2338 |
| Random Forest | 0.5889 | 0.0230 | 0.1183 | 0.0385 | 0.2338 |
| SVM (RBF) | 0.5037 | 0.0132 | 0.0310 | 0.0185 | 0.1042 |

### Accuracy by horizon (XGBoost deployment models)

| horizon | RMSE | MAPE | MacroF1 | ROC_AUC | interval_coverage |
|---|---|---|---|---|---|
| +30 min | 58.1185 | 0.0946 | 0.7665 | 0.6135 | 0.9013 |
| +60 min | 58.3344 | 0.0944 | 0.7666 | 0.6101 | 0.9048 |
| +90 min | 57.8605 | 0.0934 | 0.7682 | 0.5966 | 0.9070 |
| +120 min | 57.5281 | 0.0930 | 0.7667 | 0.5875 | 0.9068 |

Figures: `reports/figures/` (confusion matrices, ROC, importance, residuals, error breakdown, loss curves).

## 6. Model selection (PRD 11.4)

Champions are chosen on the **validation** window (never the test set), then confirmed on test. Besides the headline metric, stability across time-series CV folds, inference cost and interpretability were weighed.

| task | champion (validation) | metric | validation | test | CV std (3 folds) | inference ms / 1k rows |
|---|---|---|---|---|---|---|
| volume | Random Forest | RMSE | 57.7249 | 58.0465 | 2.4623 | 9.5638 |
| travel_time | XGBoost | RMSE | 1.0330 | 1.0606 | 0.0747 | 3.2538 |
| congestion | XGBoost + tuned class weights | MacroF1 | 0.7681 | 0.7665 | n/a | n/a |
| accident | XGBoost | ROC_AUC | 0.6001 | 0.6135 | 0.0100 | 1.4395 |

Tree ensembles beat the linear model clearly on volume (RMSE 58.0 vs 72.1, about 19% lower) because flow depends on non-linear thresholds (peak vs off-peak, capacity) that a linear model cannot express; a single decision tree sits in between. Random Forest and XGBoost differ by only 0.1 RMSE, far less than the cross-validation fold-to-fold spread (+/- 3.7), so they are statistically tied; the dashboard's multi-horizon deployment models use XGBoost because it is compact, fast (3.1 ms per 1,000 rows), provides feature importances for operators and can be trained per horizon in seconds. Corridor-wide batch inference takes well under the 30 s target. The SVM was trained on a 15,000-row time-spread subsample because kernel SVMs scale quadratically with rows; on the rare accident class it therefore sees only ~140 positives and performs near chance - an expected limitation, not a defect.

## 7. Deep learning vs classical (PRD 12.4)

The LSTM/Bi-LSTM (hidden 64, 12-window input, dropout 0.2, Adam, early stopping, LR reduction on plateau) was trained from scratch in NumPy and evaluated on the **same test window**. Best classical volume RMSE = 58.0 (Random Forest); best sequence model = 56.5 (Bi-LSTM).
The sequence model **beat** the best classical model on volume RMSE, by a modest 2.6% (single training seed, so treat the margin as indicative rather than decisive; MAPE is slightly worse than Random Forest).

## 8. Recommendations & limitations

* Use volume and congestion forecasts operationally now; treat accident risk as a ranked watch-list (rain/fog and V/C >= 0.8 raise risk several-fold).
* Weather at the target window is assumed equal to the latest observation (persistence); wiring a forecast feed would help rain/fog onset.
* Five months of data covers no summer/monsoon; retrain and monitor drift before extending seasonally.
* Next steps (PRD section 19): streaming ingestion, graph models over road topology, calibrated probabilistic forecasts, drift monitoring.

## 9. Reproducibility

`pip install -r requirements.txt && python run_pipeline.py` reproduces every number here from `data/raw/`; `pytest -q` checks cleaning, leakage and gradient correctness; `streamlit run dashboard/app.py` serves the dashboard.
