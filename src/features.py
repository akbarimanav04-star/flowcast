"""M3 - Feature engineering, supervised-table construction and time-based splitting.

Terminology
-----------
* **origin window t**  - the last window whose data we know when we make a forecast.
* **horizon h**        - how many 30-min windows ahead we predict (h=1 -> +30 min ... h=4 -> +120 min).
* Every input feature is computed from windows <= t (lags / rolling stats use only the past, per road,
  in strict time order). Calendar facts about the *target* window (hour, holiday...) are known in advance
  and therefore allowed. Targets are shifted -h per road.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import CFG, CONGESTION_TO_INT, HybridScaler

F = CFG["features"]

# ---- feature catalogue (single source of truth for models + dashboard) ----------------------
CURRENT = ["traffic_volume", "avg_speed", "occupancy", "travel_time", "signal_timing", "vc_ratio", "headroom",
           "share_2W", "share_Car", "share_LCV", "share_HCV", "accident_now"]
LAGS = [f"{c}_lag{l}" for c in ("traffic_volume", "avg_speed") for l in F["lags"]]
ROLL = [f"{c}_roll{stat}{w}" for c in ("traffic_volume", "avg_speed") for stat in ("mean", "std")
        for w in F["rolling_windows"]]
WEATHER = ["temperature", "rainfall", "visibility", "rain_flag", "low_vis_flag", "temp_band", "is_fog", "is_rain"]
STATIC = ["road_capacity", "latitude", "longitude", "road_idx"]
TARGET_TIME = ["t_hour_sin", "t_hour_cos", "t_dow_sin", "t_dow_cos", "t_weekend", "t_peak"]
CALENDAR = ["t_holiday", "t_event", "t_roadwork", "t_holiday_x_peak", "t_days_to_event", "horizon"]
FEATURES = CURRENT + LAGS + ROLL + WEATHER + STATIC + TARGET_TIME + CALENDAR

GROUPS = {"Current state": CURRENT, "Lag features": LAGS, "Rolling statistics": ROLL, "Weather": WEATHER,
          "Road (static)": STATIC, "Target-time encodings": TARGET_TIME, "Calendar": CALENDAR}
MINMAX_COLS = ["occupancy", "visibility", "share_2W", "share_Car", "share_LCV", "share_HCV"]
BINARY_OR_CYCLIC = {"rain_flag", "low_vis_flag", "is_fog", "is_rain", "t_weekend", "t_peak", "t_holiday",
                    "t_event", "t_roadwork", "t_holiday_x_peak", "t_hour_sin", "t_hour_cos", "t_dow_sin",
                    "t_dow_cos", "temp_band", "accident_now"}
ZSCORE_COLS = [c for c in FEATURES if c not in MINMAX_COLS and c not in BINARY_OR_CYCLIC]

TARGETS = {"volume": "y_volume", "travel_time": "y_travel", "congestion": "y_cong", "accident": "y_acc"}


def date_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar date with corridor-wide flags + distance (days) to the nearest event."""
    d = df.groupby("date")[["public_holiday", "event_flag", "roadwork_flag"]].max()
    ev = d.index[d.event_flag == 1]
    if len(ev):
        d["days_to_event"] = [min(7, int(np.abs((ev - x).days).min())) for x in d.index]
    else:
        d["days_to_event"] = 7
    return d


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Origin-window features (lags, rolling stats, ratios, weather flags). Uses only past/current data."""
    df = df.sort_values(["road_id", "timestamp"]).reset_index(drop=True).copy()
    g = df.groupby("road_id", sort=False)
    df["road_idx"] = df["road_id"].str.extract(r"(\d+)").astype(int)
    df["accident_now"] = (df["accident_count"] > 0).astype(int)
    half_cap = df["road_capacity"] / 2.0
    df["vc_ratio"] = df["traffic_volume"] / half_cap
    df["headroom"] = half_cap - df["traffic_volume"]

    for col in ("traffic_volume", "avg_speed"):
        for l in F["lags"]:
            df[f"{col}_lag{l}"] = g[col].shift(l)
        for w in F["rolling_windows"]:
            roll = g[col].rolling(w, min_periods=w)
            df[f"{col}_rollmean{w}"] = roll.mean().reset_index(level=0, drop=True)
            df[f"{col}_rollstd{w}"] = roll.std().reset_index(level=0, drop=True)

    df["rain_flag"] = (df["rainfall"] >= F["rain_mm"]).astype(int)
    df["low_vis_flag"] = (df["visibility"] < F["low_visibility_m"]).astype(int)
    df["temp_band"] = pd.cut(df["temperature"], [-np.inf, 12, 22, np.inf], labels=[0, 1, 2]).astype(int)
    df["is_fog"] = (df["weather_condition"] == "Fog").astype(int)
    df["is_rain"] = (df["weather_condition"] == "Rain").astype(int)
    return df


def add_horizon(df: pd.DataFrame, h: int, dates: pd.DataFrame | None = None, with_targets: bool = True) -> pd.DataFrame:
    """Attach target-window encodings (+ targets shifted -h) for forecasting h windows ahead."""
    dates = date_table(df) if dates is None else dates
    out = df.copy()
    tts = out["timestamp"] + pd.Timedelta(minutes=30 * h)
    hour = tts.dt.hour + tts.dt.minute / 60.0
    dow = tts.dt.dayofweek
    out["t_hour_sin"], out["t_hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    out["t_dow_sin"], out["t_dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    out["t_weekend"] = (dow >= 5).astype(int)
    out["t_peak"] = tts.dt.hour.isin(F["peak_hours"]).astype(int)
    tdate = tts.dt.tz_localize(None).dt.normalize()
    out["t_holiday"] = tdate.map(dates["public_holiday"]).fillna(0).astype(int)
    out["t_event"] = tdate.map(dates["event_flag"]).fillna(0).astype(int)
    out["t_roadwork"] = tdate.map(dates["roadwork_flag"]).fillna(0).astype(int)
    out["t_days_to_event"] = tdate.map(dates["days_to_event"]).fillna(7).astype(int)
    out["t_holiday_x_peak"] = out["t_holiday"] * out["t_peak"]
    out["horizon"] = h
    out["target_ts"] = tts
    if with_targets:
        g = out.groupby("road_id", sort=False)
        out["y_volume"] = g["traffic_volume"].shift(-h)
        out["y_travel"] = g["travel_time"].shift(-h)
        out["y_cong"] = g["congestion_level"].shift(-h).map(CONGESTION_TO_INT)
        out["y_acc"] = (g["accident_count"].shift(-h) > 0).astype(int).where(g["accident_count"].shift(-h).notna())
        out["y_imputed"] = g["is_imputed"].shift(-h)
    return out


def build_supervised(df_clean: pd.DataFrame, h: int = 1, drop_imputed_targets: bool = True,
                     base: pd.DataFrame | None = None) -> pd.DataFrame:
    """Full supervised table for horizon h: features at t, targets at t+h. Rows without a full history or
    without a target (and, optionally, whose target window was only estimated) are removed."""
    base = add_base_features(df_clean) if base is None else base
    sup = add_horizon(base, h)
    sup = sup.dropna(subset=FEATURES + ["y_volume", "y_travel", "y_cong", "y_acc"])
    if drop_imputed_targets:
        sup = sup[sup["y_imputed"] == 0]
    sup = sup.drop(columns="y_imputed")
    sup["y_cong"] = sup["y_cong"].astype(int)
    sup["y_acc"] = sup["y_acc"].astype(int)
    # sort chronologically (then by road) so that TimeSeriesSplit folds are genuinely time-ordered
    return sup.sort_values(["timestamp", "road_id"]).reset_index(drop=True)


def split_by_time(sup: pd.DataFrame):
    """Chronological train / validation / test split on the origin date (never random)."""
    d = sup["date"]
    tr = sup[d <= pd.Timestamp(CFG["split"]["train_end"])]
    va = sup[(d > pd.Timestamp(CFG["split"]["train_end"])) & (d <= pd.Timestamp(CFG["split"]["val_end"]))]
    te = sup[d > pd.Timestamp(CFG["split"]["val_end"])]
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


def fit_scaler(train: pd.DataFrame) -> HybridScaler:
    """Learn scaling statistics on the TRAIN split only."""
    return HybridScaler(MINMAX_COLS, ZSCORE_COLS).fit(train[FEATURES])


def scale(df: pd.DataFrame, scaler: HybridScaler) -> pd.DataFrame:
    return scaler.transform(df[FEATURES])
