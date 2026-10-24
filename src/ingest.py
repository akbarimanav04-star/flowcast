"""M1 - Ingestion & validation.

Loads the three raw tables, enforces schema/dtypes, parses the two different
date formats into one tz-aware timestamp and flags every rule violation.
Nothing is silently dropped: violating rows are written to a quarantine file
with the reason, even when a later cleaning stage repairs them.
"""
from __future__ import annotations

import pandas as pd

from .utils import CFG, get_logger, path

log = get_logger("ingest")
TZ = "Asia/Kolkata"   # the corridor GPS coordinates are in Delhi, India

SCHEMA = {
    "traffic": ["road_id", "road_name", "latitude", "longitude", "weather_station_id", "date", "time",
                "traffic_volume", "vehicle_count", "vehicle_type_dist", "avg_speed", "occupancy",
                "congestion_level", "travel_time", "accident_count", "signal_timing", "road_capacity"],
    "weather": ["station_id", "date", "time", "weather_condition", "temperature", "rainfall", "visibility"],
    "calendar": ["date", "public_holiday", "holiday_name", "event_flag", "event_name", "roadwork_flag"],
}
NUMERIC = {
    "traffic": ["latitude", "longitude", "traffic_volume", "vehicle_count", "avg_speed", "occupancy",
                "travel_time", "accident_count", "signal_timing", "road_capacity"],
    "weather": ["temperature", "rainfall", "visibility"],
    "calendar": ["public_holiday", "event_flag", "roadwork_flag"],
}


def _check_schema(df: pd.DataFrame, name: str) -> None:
    missing = [c for c in SCHEMA[name] if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] missing required columns: {missing}")


def _to_timestamp(date: pd.Series, time: pd.Series, date_format: str) -> pd.Series:
    """Parse an explicit date format + HH:MM into one tz-aware timestamp."""
    ts = pd.to_datetime(date.str.strip() + " " + time.str.strip(), format=f"{date_format} %H:%M", errors="coerce")
    return ts.dt.tz_localize(TZ)


def validate_traffic(df: pd.DataFrame) -> pd.DataFrame:
    """Parse + range-check the traffic table. Adds ``timestamp`` and ``qc_reason`` columns."""
    _check_schema(df, "traffic")
    df = df.copy()
    for c in NUMERIC["traffic"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["timestamp"] = _to_timestamp(df["date"], df["time"], "%Y-%m-%d")
    v = CFG["validation"]
    reasons = pd.Series("", index=df.index)

    def flag(mask: pd.Series, reason: str) -> None:
        nonlocal reasons
        reasons = reasons.where(~mask, reasons + reason + ";")

    flag(df["timestamp"].isna(), "bad_timestamp")
    flag(df["traffic_volume"] < v["min_volume"], "negative_volume")
    flag(df["avg_speed"] > v["max_speed_kmh"], "impossible_speed")
    flag(df["occupancy"] > v["max_occupancy_pct"], "impossible_occupancy")
    df["qc_reason"] = reasons
    return df


def validate_weather(df: pd.DataFrame) -> pd.DataFrame:
    _check_schema(df, "weather")
    df = df.copy()
    for c in NUMERIC["weather"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["timestamp"] = _to_timestamp(df["date"], df["time"], "%d/%m/%Y")   # weather uses DD/MM/YYYY
    df["qc_reason"] = ""
    df.loc[df["timestamp"].isna(), "qc_reason"] = "bad_timestamp;"
    return df


def validate_calendar(df: pd.DataFrame) -> pd.DataFrame:
    _check_schema(df, "calendar")
    df = df.copy()
    for c in NUMERIC["calendar"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce").dt.normalize()
    df["qc_reason"] = ""
    return df


def load_raw(raw_dir: str | None = None, traffic_df: pd.DataFrame | None = None):
    """Load + validate everything. ``traffic_df`` lets the dashboard inject an uploaded file."""
    raw = path(raw_dir or CFG["paths"]["raw"], "x").parent
    files = CFG["files"]
    calendar = validate_calendar(pd.read_csv(raw / files["calendar"]))
    weather = validate_weather(pd.read_csv(raw / files["weather"]))
    traffic_raw = traffic_df if traffic_df is not None else pd.read_csv(raw / files["traffic"])
    traffic = validate_traffic(traffic_raw)
    log.info("loaded calendar=%d weather=%d traffic=%d", len(calendar), len(weather), len(traffic))

    bad = traffic[traffic["qc_reason"] != ""]
    q = path(CFG["paths"]["interim"], "quarantine_traffic.csv" if traffic_df is None else "quarantine_upload.csv")
    bad.drop(columns=["timestamp"]).to_csv(q, index=False)
    log.info("quarantine: %d traffic rows violate range rules (saved -> %s)", len(bad), q.name)
    return traffic, weather, calendar
