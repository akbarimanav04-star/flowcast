"""Inference: reload persisted models (no retraining) and forecast every segment for horizons 30-120 min."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from .features import FEATURES, add_base_features, add_horizon, date_table
from .utils import CFG, CONGESTION_ORDER, path

_CACHE: dict = {}


def load_artifacts() -> dict:
    """Load the XGBoost horizon models, conformal quantiles and decision weights once."""
    if _CACHE:
        return _CACHE
    mdir = CFG["paths"]["models"]
    meta = json.load(open(path(mdir, "meta.json")))
    champs = json.load(open(path(mdir, "champions.json")))
    models = {(t, h): joblib.load(path(mdir, f"xgb_{t}_h{h}.joblib"))
              for t in ("volume", "travel_time", "congestion", "accident") for h in CFG["features"]["horizons"]}
    _CACHE.update(meta=meta, champions=champs, models=models)
    return _CACHE


def forecast(df_clean: pd.DataFrame, origin: pd.Timestamp | None = None, base: pd.DataFrame | None = None) -> pd.DataFrame:
    """Forecast all segments from one origin window. Defaults to the latest window in the data."""
    art = load_artifacts()
    base = add_base_features(df_clean) if base is None else base
    origin = base["timestamp"].max() if origin is None else origin
    o = base[base["timestamp"] == origin]
    if o.empty:
        raise ValueError(f"No data at origin {origin}")
    dates = date_table(df_clean)
    cw = np.asarray(art["champions"]["class_weights"])
    out = []
    for h in CFG["features"]["horizons"]:
        x = add_horizon(o, h, dates=dates, with_targets=False)
        ok = x[FEATURES].notna().all(axis=1)
        if not ok.any():
            raise ValueError("Not enough history before the origin (need >= 48 windows per segment for the t-48 lag).")
        x = x[ok]
        X = x[FEATURES]
        vol = art["models"][("volume", h)].predict(X)
        q = x["road_id"].map(art["meta"]["conformal_q90"][f"volume_h{h}"]).fillna(np.nanmean(list(art["meta"]["conformal_q90"][f"volume_h{h}"].values()))).values
        tt = art["models"][("travel_time", h)].predict(X)
        pr = art["models"][("congestion", h)].predict_proba(X) * cw
        pr = pr / pr.sum(1, keepdims=True)
        acc = art["models"][("accident", h)].predict_proba(X)[:, 1]
        thr = art["meta"]["accident_threshold"][f"h{h}"]
        out.append(pd.DataFrame({
            "road_id": x["road_id"].values, "horizon": h, "minutes_ahead": 30 * h, "target_ts": x["target_ts"].values,
            "volume_pred": vol, "volume_lo": np.maximum(vol - q, 0), "volume_hi": vol + q,
            "travel_time_pred": tt, "congestion": [CONGESTION_ORDER[i] for i in pr.argmax(1)], "congestion_conf": pr.max(1),
            "accident_prob": acc, "risk_band": np.where(acc >= thr, "High", np.where(acc >= thr / 2, "Elevated", "Low"))}))
    res = pd.concat(out, ignore_index=True)
    meta_cols = o[["road_id", "road_name", "latitude", "longitude", "road_capacity"]].drop_duplicates("road_id")
    return res.merge(meta_cols, on="road_id", how="left")


def predict_from_upload(traffic_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate + clean an uploaded traffic file (same schema as traffic_sensor_log.csv) and forecast from its last window."""
    from . import clean
    df = clean.run(traffic_df=traffic_df, save=False)
    return df, forecast(df)
