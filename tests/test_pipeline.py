"""Run with:  pytest -q      (requires the pipeline to have been run once: python run_pipeline.py)"""
import numpy as np
import pandas as pd
import pytest

from src import clean
from src.features import FEATURES, add_base_features, build_supervised, fit_scaler, scale, split_by_time
from src.ml_models import LinearRegressionGD
from src.utils import CFG, path


@pytest.fixture(scope="module")
def df():
    return pd.read_parquet(path(CFG["paths"]["processed"], "flowcast_clean.parquet"))


def test_clean_has_no_nulls_or_duplicates(df):
    assert df[["traffic_volume", "avg_speed", "occupancy", "congestion_level", "visibility"]].isna().sum().sum() == 0
    assert not df.duplicated(["road_id", "timestamp"]).any()
    assert df.avg_speed.max() <= 200 and df.occupancy.max() <= 100 and df.traffic_volume.min() >= 0


def test_full_grid(df):
    assert len(df) == df.road_id.nunique() * df.timestamp.nunique()


def test_weather_labels_controlled(df):
    assert set(df.weather_condition) <= {"Clear", "Cloudy", "Rain", "Fog"}


def test_congestion_rule():
    v = pd.Series([100, 700, 950, 1200]); cap = pd.Series([2000] * 4)   # half capacity = 1000
    assert clean.derive_congestion(v, cap).tolist() == ["Free-flow", "Moderate", "Heavy", "Severe"]


def test_no_leakage_lag_and_target(df):
    sup = build_supervised(df, 1)
    r = sup[sup.road_id == "NL-007"]
    vol = df[df.road_id == "NL-007"].set_index("timestamp").traffic_volume          # look values up by time
    assert np.allclose(r.traffic_volume_lag1.values, vol.loc[r.timestamp - pd.Timedelta(minutes=30)].values)  # lag = past
    assert np.allclose(r.y_volume.values, vol.loc[r.timestamp + pd.Timedelta(minutes=30)].values)             # target = next
    assert (r.target_ts > r.timestamp).all()


def test_split_is_chronological(df):
    tr, va, te = split_by_time(build_supervised(df, 1))
    assert tr.timestamp.max() < va.timestamp.min() and va.timestamp.max() < te.timestamp.min()


def test_scaler_uses_train_only(df):
    tr, va, te = split_by_time(build_supervised(df, 1))
    sc = fit_scaler(tr)
    assert abs(scale(tr, sc)["traffic_volume"].mean()) < 1e-6          # train is exactly standardised ...
    assert abs(scale(te, sc)["traffic_volume"].mean()) > 1e-6          # ... test is not (stats not re-fitted)


def test_gradient_descent_gradients():
    rs = np.random.RandomState(0); X, y, w = rs.randn(200, 4), rs.randn(200), rs.randn(4)
    J, gw, gb = LinearRegressionGD.cost_and_grads(X, y, w, 0.3)
    w2 = w.copy(); w2[1] += 1e-6
    num = (LinearRegressionGD.cost_and_grads(X, y, w2, 0.3)[0] - J) / 1e-6
    assert abs(num - gw[1]) < 1e-4


def test_feature_count():
    assert len(FEATURES) == len(set(FEATURES)) == 50
