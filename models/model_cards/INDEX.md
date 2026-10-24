# Model cards index

Dataset version `dae7a579b5`, seed 42. Full cards: `models/model_cards/*.json`.

| Task | Model | Champion (h=1) | Key test metric |
|---|---|---|---|
| volume | Linear Regression (NumPy GD, from scratch) |  | RMSE = 72.4221 |
| volume | Linear Regression |  | RMSE = 72.0952 |
| volume | Decision Tree |  | RMSE = 64.0057 |
| volume | Random Forest | yes | RMSE = 58.0465 |
| volume | XGBoost |  | RMSE = 58.1185 |
| travel_time | Linear Regression |  | RMSE = 1.4047 |
| travel_time | Decision Tree |  | RMSE = 1.1845 |
| travel_time | Random Forest |  | RMSE = 1.0680 |
| travel_time | XGBoost | yes | RMSE = 1.0606 |
| congestion | Logistic Regression |  | MacroF1 = 0.7276 |
| congestion | Decision Tree |  | MacroF1 = 0.7514 |
| congestion | Random Forest |  | MacroF1 = 0.7602 |
| congestion | XGBoost |  | MacroF1 = 0.7627 |
| congestion | SVM (RBF) |  | MacroF1 = 0.7217 |
| congestion | XGBoost + tuned class weights | yes | MacroF1 = 0.7665 |
| accident | Logistic Regression |  | ROC_AUC = 0.6282 |
| accident | Decision Tree |  | ROC_AUC = 0.5924 |
| accident | Random Forest |  | ROC_AUC = 0.5889 |
| accident | XGBoost | yes | ROC_AUC = 0.6135 |
| accident | SVM (RBF) |  | ROC_AUC = 0.5037 |
| congestion | Bi-LSTM |  | MacroF1 = 0.7668 |
| congestion | LSTM |  | MacroF1 = 0.7655 |
| volume | Bi-LSTM |  | RMSE = 56.5469 |
| volume | LSTM |  | RMSE = 56.6209 |
