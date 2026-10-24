"""M5 - Classical ML engine.

* a NumPy gradient-descent linear regression (the PRD's "build-it-once" rule)
* Linear/Logistic Regression, Decision Tree, Random Forest, XGBoost and SVM
* time-series cross-validated tuning, validation-based model selection, conformal uncertainty
* multi-horizon XGBoost models (30-120 min) for the dashboard
"""
from __future__ import annotations

import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import evaluate as ev
from .features import (FEATURES, TARGETS, add_base_features, build_supervised, fit_scaler, scale, split_by_time)
from .utils import CFG, CONGESTION_ORDER, get_logger, path, set_seed

warnings.filterwarnings("ignore")
log = get_logger("ml")
SEED = CFG["seed"]
TASK_TYPE = {"volume": "reg", "travel_time": "reg", "congestion": "multi", "accident": "binary"}
PRIMARY = {"volume": ("RMSE", False), "travel_time": ("RMSE", False), "congestion": ("MacroF1", True), "accident": ("ROC_AUC", True)}


def _jsonable(o):
    """json.dumps helper: numpy scalars -> plain numbers."""
    return o.item() if hasattr(o, "item") else str(o)


def _parse_params(raw: str) -> dict:
    """Read hyper-parameters back from the scoreboard CSV (values may have been stored as strings)."""
    out = {}
    for k, v in json.loads(raw).items():
        try:
            out[k] = int(v) if float(v) == int(float(v)) and "." not in str(v) else float(v)
        except (TypeError, ValueError):
            out[k] = v
    return out


# ----------------------------------------------------------------------------------------------
# 1. Linear regression from first principles (NumPy only)
# ----------------------------------------------------------------------------------------------
class LinearRegressionGD:
    """y_hat = X w + b, trained by batch gradient descent on the MSE cost.

    cost  J(w,b) = (1/n) * sum (X w + b - y)^2
    grads dJ/dw  = (2/n) * X^T (X w + b - y)      dJ/db = (2/n) * sum (X w + b - y)
    """

    def __init__(self, lr: float | None = None, epochs: int = 1500):
        self.lr, self.epochs = lr, epochs
        self.w: np.ndarray | None = None
        self.b = 0.0
        self.history: list[float] = []

    @staticmethod
    def cost_and_grads(X, y, w, b):
        err = X @ w + b - y                      # matrix-vector product, shape (n,)
        n = len(y)
        return float(np.mean(err ** 2)), (2.0 / n) * (X.T @ err), (2.0 / n) * err.sum()

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.mu, self.sd = y.mean(), y.std()
        yz = (y - self.mu) / self.sd             # z-score the target so one learning rate suits every target
        if self.lr is None:                      # largest stable step: lr < 1 / lambda_max(X^T X / n)
            self.lr = 0.9 / float(np.linalg.eigvalsh(X.T @ X / len(X)).max())
        self.w, self.b, self.history = np.zeros(X.shape[1]), 0.0, []
        for _ in range(self.epochs):
            J, gw, gb = self.cost_and_grads(X, yz, self.w, self.b)
            self.w -= self.lr * gw
            self.b -= self.lr * gb
            self.history.append(J)
        return self

    def predict(self, X):
        return (np.asarray(X, float) @ self.w + self.b) * self.sd + self.mu


# ----------------------------------------------------------------------------------------------
# 2. Model zoo + tuning grids
# ----------------------------------------------------------------------------------------------
def model_zoo(task: str, pos_weight: float = 1.0):
    """Return {name: (estimator, param_grid, needs_scaling)}; grids are small on purpose (single CPU)."""
    typ = TASK_TYPE[task]
    nj = CFG["ml"]["n_jobs"]
    if typ == "reg":
        return {
            "Linear Regression": (LinearRegression(), {}, True),
            "Decision Tree": (DecisionTreeRegressor(random_state=SEED), {"max_depth": [8, 12, 16], "min_samples_leaf": [20, 60]}, False),
            "Random Forest": (RandomForestRegressor(n_estimators=80, max_samples=0.5, max_features=0.5, n_jobs=nj, random_state=SEED),
                              {"min_samples_leaf": [5, 20], "max_depth": [14]}, False),
            "XGBoost": (xgb.XGBRegressor(n_estimators=300, subsample=0.8, colsample_bytree=0.8, n_jobs=nj, random_state=SEED),
                        {"max_depth": [4, 6], "learning_rate": [0.05, 0.1]}, False),
        }
    cw = "balanced" if typ == "binary" else None
    xgb_grid = ({"max_depth": [2, 3, 4], "learning_rate": [0.05], "n_estimators": [150, 300], "min_child_weight": [20]}
                if typ == "binary" else {"max_depth": [4, 6], "learning_rate": [0.05, 0.1]})
    xgb_kw = dict(subsample=0.8, colsample_bytree=0.8, n_jobs=nj, random_state=SEED)
    xgb_est = (xgb.XGBClassifier(**xgb_kw) if typ == "binary" else
               xgb.XGBClassifier(n_estimators=300, objective="multi:softprob", **xgb_kw))
    return {
        "Logistic Regression": (LogisticRegression(max_iter=400, class_weight=cw), {"C": [0.1, 1.0]}, True),
        "Decision Tree": (DecisionTreeClassifier(random_state=SEED, class_weight=cw),
                          {"max_depth": [4, 8, 12], "min_samples_leaf": [50, 150]}, False),
        "Random Forest": (RandomForestClassifier(n_estimators=80, max_samples=0.5, max_features=0.5, n_jobs=nj,
                                                 random_state=SEED, class_weight="balanced_subsample" if typ == "binary" else None),
                          {"min_samples_leaf": [10, 40], "max_depth": [14]}, False),
        "XGBoost": (xgb_est, xgb_grid, False),
        "SVM (RBF)": (SVC(kernel="rbf", C=1.0, gamma="scale", class_weight=cw), {}, True),
    }


def _cv_scoring(task: str) -> str:
    return {"volume": "neg_root_mean_squared_error", "travel_time": "neg_root_mean_squared_error",
            "congestion": "f1_macro", "accident": "roc_auc"}[task]


def _score(task: str, model, X) -> np.ndarray:
    """Continuous output used for the metric: predictions, class labels, or a risk score."""
    typ = TASK_TYPE[task]
    if typ == "reg":
        return model.predict(X)
    if typ == "multi":
        return model.predict(X)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)       # SVM


def _metrics(task: str, y, out, thr=None) -> dict:
    typ = TASK_TYPE[task]
    return (ev.regression_metrics(y, out) if typ == "reg" else
            ev.congestion_metrics(y, out) if typ == "multi" else ev.risk_metrics(y, out, thr))


# ----------------------------------------------------------------------------------------------
# 3. Optional: decision weights for the 4-class problem (tuned on validation only)
# ----------------------------------------------------------------------------------------------
def tune_class_weights(proba: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Coordinate search for per-class probability multipliers that maximise validation macro-F1."""
    w = np.ones(proba.shape[1])
    best = f1_score(y, (proba * w).argmax(1), average="macro")
    for _ in range(3):
        for k in range(len(w)):
            for cand in np.linspace(0.6, 2.2, 17):
                w2 = w.copy(); w2[k] = cand
                s = f1_score(y, (proba * w2).argmax(1), average="macro")
                if s > best + 1e-6:
                    best, w = s, w2
    return w


# ----------------------------------------------------------------------------------------------
# 4. Main training routine (horizon 1 = full comparison)
# ----------------------------------------------------------------------------------------------
def train_all(save: bool = True) -> dict:
    set_seed()
    df = pd.read_parquet(path(CFG["paths"]["processed"], "flowcast_clean.parquet"))
    base = add_base_features(df)
    sup = build_supervised(df, 1, base=base)
    tr, va, te = split_by_time(sup)
    log.info("h=1 rows  train=%d  val=%d  test=%d  features=%d", len(tr), len(va), len(te), len(FEATURES))

    scaler = fit_scaler(tr)                      # statistics from TRAIN only
    Xs = {k: scale(d, scaler) for k, d in zip(("tr", "va", "te"), (tr, va, te))}
    Xr = {"tr": tr[FEATURES], "va": va[FEATURES], "te": te[FEATURES]}
    cv = TimeSeriesSplit(n_splits=CFG["ml"]["cv_splits"])
    rows, fitted, preds = [], {}, {}

    # ---- build-it-once: gradient-descent linear regression vs the closed-form solution
    for task in ("volume",):
        y = tr[TARGETS[task]].values
        gd = LinearRegressionGD().fit(Xs["tr"].values, y)
        for split in ("va", "te"):
            m = ev.regression_metrics({"va": va, "te": te}[split][TARGETS[task]], gd.predict(Xs[split].values))
            rows.append({"task": task, "model": "Linear Regression (NumPy GD, from scratch)", "split": split, **m})
        fitted[(task, "gd")] = gd
        log.info("NumPy GD linear regression: first/last loss %.4f -> %.4f", gd.history[0], gd.history[-1])

    for task in ("volume", "travel_time", "congestion", "accident"):
        ycol = TARGETS[task]
        ytr, yva, yte = tr[ycol].values, va[ycol].values, te[ycol].values
        pos_w = float((ytr == 0).sum() / max((ytr == 1).sum(), 1)) if task == "accident" else 1.0
        for name, (est, grid, needs_scale) in model_zoo(task, pos_w).items():
            X = Xs if needs_scale else Xr
            Xtr, ytr_fit = X["tr"], ytr
            t0 = time.time()
            if name.startswith("SVM"):                           # O(n^2): train on a time-spread subsample
                idx = np.sort(np.random.RandomState(SEED).choice(len(tr), CFG["ml"]["svm_train_rows"], replace=False))
                Xtr, ytr_fit = X["tr"].iloc[idx], ytr[idx]
            if grid:
                gs = GridSearchCV(est, grid, cv=cv, scoring=_cv_scoring(task), n_jobs=1, refit=True)
                gs.fit(Xtr, ytr_fit)
                model, best = gs.best_estimator_, gs.best_params_
                k = gs.best_index_
                cv_mean, cv_std = float(gs.cv_results_["mean_test_score"][k]), float(gs.cv_results_["std_test_score"][k])
                if _cv_scoring(task).startswith("neg_"):
                    cv_mean = -cv_mean
            else:
                model = est.fit(Xtr, ytr_fit); best, cv_mean, cv_std = {}, np.nan, np.nan
            fit_s = time.time() - t0
            t1 = time.time(); out_va = _score(task, model, X["va"]); out_te = _score(task, model, X["te"])
            ms_per_1k = (time.time() - t1) / (len(va) + len(te)) * 1e6
            thr = ev.best_f1_threshold(yva, out_va) if task == "accident" else None
            for split, y, out in (("va", yva, out_va), ("te", yte, out_te)):
                rows.append({"task": task, "model": name, "split": split, **_metrics(task, y, out, thr),
                             "cv_mean": cv_mean, "cv_std": cv_std, "fit_seconds": fit_s,
                             "infer_ms_per_1k": ms_per_1k, "best_params": json.dumps(best, default=_jsonable)})
            fitted[(task, name)] = (model, needs_scale, best)
            preds[(task, name)] = out_te
            log.info("%-11s %-20s val %s=%.4f | test %s=%.4f | %.0fs", task, name, PRIMARY[task][0],
                     rows[-2][PRIMARY[task][0]], PRIMARY[task][0], rows[-1][PRIMARY[task][0]], fit_s)

    # ---- congestion: tuned decision weights on the XGBoost probabilities
    cm, _, _ = fitted[("congestion", "XGBoost")]
    pva, pte = cm.predict_proba(Xr["va"]), cm.predict_proba(Xr["te"])
    cw = tune_class_weights(pva, va["y_cong"].values)
    for split, p, y in (("va", pva, va["y_cong"].values), ("te", pte, te["y_cong"].values)):
        rows.append({"task": "congestion", "model": "XGBoost + tuned class weights", "split": split,
                     **ev.congestion_metrics(y, (p * cw).argmax(1)), "best_params": json.dumps({"class_weights": cw.round(2).tolist()}, default=_jsonable)})
    preds[("congestion", "XGBoost + tuned class weights")] = (pte * cw).argmax(1)

    score = pd.DataFrame(rows)
    score.to_csv(path(CFG["paths"]["reports"], "scoreboard_classical.csv"), index=False)

    # ---- selection: validation score decides (never the test set)
    champions = {}
    for task in TASK_TYPE:
        met, higher = PRIMARY[task]
        v = score[(score.task == task) & (score.split == "va") & ~score.model.str.contains("from scratch")]
        champions[task] = v.sort_values(met, ascending=not higher).iloc[0]["model"]
    log.info("champions (by validation): %s", champions)

    # ---- diagnostics for h=1
    xm = fitted[("volume", "XGBoost")][0]
    ev.plot_importance(pd.Series(xm.feature_importances_, FEATURES), "XGBoost feature importance - volume", "fi_volume.png")
    ev.plot_importance(pd.Series(fitted[("accident", "XGBoost")][0].feature_importances_, FEATURES), "XGBoost feature importance - accident risk", "fi_accident.png")
    ev.plot_importance(pd.Series(fitted[("congestion", "XGBoost")][0].feature_importances_, FEATURES), "XGBoost feature importance - congestion", "fi_congestion.png")
    rf = fitted[("volume", "Random Forest")][0]
    ev.plot_importance(pd.Series(rf.feature_importances_, FEATURES), "Random Forest feature importance - volume", "fi_volume_rf.png")
    pv = preds[("volume", "XGBoost")]
    err_df = pd.DataFrame({"timestamp": te["target_ts"], "weather_condition": te["weather_condition"], "road_id": te["road_id"], "y": te["y_volume"], "p": pv})
    ev.plot_error_breakdown(err_df, "volume_error_breakdown.png"); ev.plot_residuals(err_df.y, err_df.p, "volume_residuals.png")
    ev.plot_pred_vs_actual(err_df, "NL-007", "pred_vs_actual_NL-007.png")
    ev.plot_confusion(te["y_cong"], preds[("congestion", champions["congestion"])] if champions["congestion"] in [m for (_, m) in preds] else preds[("congestion", "XGBoost")],
                      f"Congestion - {champions['congestion']} (test)", "confusion_congestion.png")
    ev.plot_roc(te["y_acc"].values, {n: preds[("accident", n)] for n in ("Logistic Regression", "Random Forest", "XGBoost", "SVM (RBF)")}, "Accident risk ROC (test)", "roc_accident.png")
    import matplotlib.pyplot as plt
    gd = fitted[("volume", "gd")]
    fig, ax = plt.subplots(figsize=(6, 3.6)); ax.plot(gd.history); ax.set(xlabel="epoch", ylabel="MSE (z-scored target)", title="NumPy gradient descent: cost J(w,b)", yscale="log")
    fig.tight_layout(); fig.savefig(ev.FIG("gd_loss_curve.png"), dpi=130); plt.close(fig)

    if save:
        joblib.dump(scaler, path(CFG["paths"]["models"], "scaler.joblib"))
        joblib.dump(gd, path(CFG["paths"]["models"], "linreg_gd_volume.joblib"))
        for task, name in champions.items():
            key = (task, name if (task, name) in fitted else name.split(" +")[0])   # "XGBoost + tuned class weights" -> XGBoost
            if key in fitted:
                joblib.dump(fitted[key][0], path(CFG["paths"]["models"], f"champion_{task}.joblib"))
        json.dump({"champions": champions, "class_weights": cw.tolist()}, open(path(CFG["paths"]["models"], "champions.json"), "w"), indent=2)
    return {"score": score, "champions": champions, "fitted": fitted, "best_params": {t: fitted[(t, "XGBoost")][2] for t in TASK_TYPE},
            "splits": (tr, va, te), "base": base, "class_weights": cw}


# ----------------------------------------------------------------------------------------------
# 5. Multi-horizon deployment models with conformal intervals
# ----------------------------------------------------------------------------------------------
def train_horizons(best_params: dict, base: pd.DataFrame, class_weights, save: bool = True) -> pd.DataFrame:
    """XGBoost per task and horizon (h=1..4) using the tuned h=1 hyper-parameters.
    Regression uncertainty = split-conformal 90% interval from validation residuals (per road)."""
    from .features import add_horizon, date_table
    set_seed()
    meta = {"conformal_q90": {}, "horizons": CFG["features"]["horizons"]}
    all_preds = []
    for h in CFG["features"]["horizons"]:
        sup = build_supervised(None, h, base=base) if False else None
        sup = add_horizon(base, h, with_targets=True)
        sup = sup.dropna(subset=FEATURES + ["y_volume", "y_travel", "y_cong", "y_acc"])
        sup = sup[sup["y_imputed"] == 0].drop(columns="y_imputed").sort_values(["timestamp", "road_id"]).reset_index(drop=True)
        sup["y_cong"], sup["y_acc"] = sup["y_cong"].astype(int), sup["y_acc"].astype(int)
        tr, va, te = split_by_time(sup)
        out = te[["road_id", "target_ts", "weather_condition"]].copy(); out["horizon"] = h
        for task in ("volume", "travel_time", "congestion", "accident"):
            ycol, p = TARGETS[task], dict(best_params[task])
            typ = TASK_TYPE[task]
            common = dict(subsample=0.8, colsample_bytree=0.8, n_jobs=1, random_state=SEED)
            if typ == "reg":
                m = xgb.XGBRegressor(n_estimators=300, **common, **p)
            elif typ == "multi":
                m = xgb.XGBClassifier(n_estimators=300, objective="multi:softprob", **common, **p)
            else:
                m = xgb.XGBClassifier(**common, **p)
            m.fit(tr[FEATURES], tr[ycol])
            if typ == "reg":
                res = (va[ycol] - m.predict(va[FEATURES])).abs()
                q = res.groupby(va["road_id"]).quantile(0.9)
                meta["conformal_q90"][f"{task}_h{h}"] = q.to_dict()
                pred = m.predict(te[FEATURES])
                out[f"{task}_pred"] = pred
                out[f"{task}_lo"] = pred - te["road_id"].map(q).values
                out[f"{task}_hi"] = pred + te["road_id"].map(q).values
                out[f"{task}_true"] = te[ycol].values
            elif typ == "multi":
                pr = m.predict_proba(te[FEATURES]) * np.asarray(class_weights)
                pr = pr / pr.sum(1, keepdims=True)
                out["congestion_pred"], out["congestion_conf"], out["congestion_true"] = pr.argmax(1), pr.max(1), te[ycol].values
            else:
                out["accident_prob"], out["accident_true"] = m.predict_proba(te[FEATURES])[:, 1], te[ycol].values
                meta.setdefault("accident_threshold", {})[f"h{h}"] = ev.best_f1_threshold(va[ycol].values, m.predict_proba(va[FEATURES])[:, 1])
            if save:
                joblib.dump(m, path(CFG["paths"]["models"], f"xgb_{task}_h{h}.joblib"))
        all_preds.append(out)
        log.info("horizon h=%d (+%d min): volume RMSE=%.1f  MacroF1=%.3f  AUC=%.3f", h, 30 * h,
                 ev.regression_metrics(out.volume_true, out.volume_pred)["RMSE"],
                 ev.congestion_metrics(out.congestion_true, out.congestion_pred)["MacroF1"],
                 ev.risk_metrics(out.accident_true.values, out.accident_prob.values)["ROC_AUC"])
    preds = pd.concat(all_preds, ignore_index=True)
    if save:
        preds.to_parquet(path(CFG["paths"]["reports"], "predictions_test_xgb.parquet"), index=False)
        json.dump(meta, open(path(CFG["paths"]["models"], "meta.json"), "w"), indent=2)
    return preds


def retrain_fast() -> dict:
    """Dashboard 'fast retrain': refit the XGBoost +30 min volume model with the tuned parameters (does not overwrite deployed models)."""
    t0 = time.time()
    sc = pd.read_csv(path(CFG["paths"]["reports"], "scoreboard_classical.csv"))
    bp = _parse_params(sc[(sc.task == "volume") & (sc.model == "XGBoost") & (sc.split == "te")].best_params.iloc[0])
    df = pd.read_parquet(path(CFG["paths"]["processed"], "flowcast_clean.parquet"))
    tr, va, te = split_by_time(build_supervised(df, 1))
    set_seed()
    m = xgb.XGBRegressor(n_estimators=300, subsample=0.8, colsample_bytree=0.8, n_jobs=1, random_state=SEED, **bp).fit(tr[FEATURES], tr["y_volume"])
    joblib.dump(m, path(CFG["paths"]["models"], "xgb_volume_h1_retrained.joblib"))
    return {"model": "XGBoost volume +30 min", "hyperparameters": bp, "train_rows": len(tr), "test_rows": len(te),
            "test_metrics": ev.regression_metrics(te["y_volume"], m.predict(te[FEATURES])), "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    res = train_all()
    train_horizons(res["best_params"], res["base"], res["class_weights"])
