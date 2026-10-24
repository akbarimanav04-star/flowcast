"""Shared helpers: config loading, logging, seeding, paths, scaling."""
from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    """Read config.yaml from the project root."""
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load_config()


def path(*parts: str) -> Path:
    """Build an absolute path inside the project and create parent dirs."""
    p = ROOT.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(name)-9s | %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def set_seed(seed: int | None = None) -> None:
    """Seed every source of randomness so reruns reproduce the same metrics."""
    seed = CFG["seed"] if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)
    except ImportError:  # torch is only needed for the deep-learning stage
        pass


CONGESTION_ORDER = ["Free-flow", "Moderate", "Heavy", "Severe"]
CONGESTION_TO_INT = {c: i for i, c in enumerate(CONGESTION_ORDER)}
SEVERITY_COLORS = {"Free-flow": "#2e9e5b", "Moderate": "#f2c230", "Heavy": "#f08a24", "Severe": "#d6322b"}


class HybridScaler:
    """Min-max for bounded columns, z-score for the rest (PRD section 10.1).

    Parameters are learned with ``fit`` on the TRAINING split only, then reused
    by ``transform`` for validation, test and live data -> no leakage.
    """

    def __init__(self, minmax_cols: list[str], zscore_cols: list[str]):
        self.minmax_cols, self.zscore_cols = list(minmax_cols), list(zscore_cols)
        self.params: dict[str, tuple[float, float]] = {}

    def fit(self, df):
        for c in self.minmax_cols:
            self.params[c] = (float(df[c].min()), float(df[c].max()))
        for c in self.zscore_cols:
            self.params[c] = (float(df[c].mean()), float(df[c].std(ddof=0)) or 1.0)
        return self

    def transform(self, df):
        out = df.copy()
        for c in self.minmax_cols:
            lo, hi = self.params[c]
            out[c] = (out[c] - lo) / ((hi - lo) or 1.0)
        for c in self.zscore_cols:
            mu, sd = self.params[c]
            out[c] = (out[c] - mu) / sd
        return out

    @property
    def columns(self) -> list[str]:
        return self.minmax_cols + self.zscore_cols
