"""Metrics, scoreboard helpers and diagnostic plots shared by the ML and DL engines."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_absolute_percentage_error, mean_squared_error, precision_score,
                             r2_score, recall_score, roc_auc_score, roc_curve)

from .utils import CFG, CONGESTION_ORDER, path

sns.set_theme(style="whitegrid", context="notebook")
FIG = lambda name: path(CFG["paths"]["figures"], name)   # noqa: E731


# ------------------------------------------------------------------ metrics
def regression_metrics(y, p) -> dict:
    """RMSE is the primary metric; MAE, MAPE and R2 support it."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"RMSE": float(np.sqrt(mean_squared_error(y, p))), "MAE": float(mean_absolute_error(y, p)),
            "MAPE": float(mean_absolute_percentage_error(y, p)), "R2": float(r2_score(y, p))}


def congestion_metrics(y, pred) -> dict:
    """Macro-F1 is the headline metric across the four classes."""
    return {"MacroF1": float(f1_score(y, pred, average="macro")), "Accuracy": float(accuracy_score(y, pred)),
            "Precision": float(precision_score(y, pred, average="macro", zero_division=0)),
            "Recall": float(recall_score(y, pred, average="macro", zero_division=0))}


def best_f1_threshold(y, score) -> float:
    """Operating threshold that maximises F1 (chosen on the VALIDATION split only)."""
    qs = np.unique(np.quantile(score, np.linspace(0.5, 0.999, 120)))
    f = [f1_score(y, score >= q, zero_division=0) for q in qs]
    return float(qs[int(np.argmax(f))])


def risk_metrics(y, score, threshold: float | None = None) -> dict:
    """ROC-AUC is the headline metric; precision/recall/F1 are reported at an operating threshold."""
    y, score = np.asarray(y), np.asarray(score, float)
    thr = best_f1_threshold(y, score) if threshold is None else threshold
    pred = score >= thr
    top = score >= np.quantile(score, 0.9)
    return {"ROC_AUC": float(roc_auc_score(y, score)), "Precision": float(precision_score(y, pred, zero_division=0)),
            "Recall": float(recall_score(y, pred, zero_division=0)), "F1": float(f1_score(y, pred, zero_division=0)),
            "Top10pct_capture": float(y[top].sum() / max(y.sum(), 1)), "threshold": thr}


# ------------------------------------------------------------------ plots
def plot_confusion(y, pred, title: str, fname: str) -> None:
    cm = confusion_matrix(y, pred, labels=range(4))
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cmn, annot=cm, fmt="d", cmap="Blues", xticklabels=CONGESTION_ORDER, yticklabels=CONGESTION_ORDER,
                cbar_kws={"label": "row-normalised"}, ax=ax)
    ax.set(xlabel="Predicted", ylabel="Actual", title=title)
    fig.tight_layout(); fig.savefig(FIG(fname), dpi=130); plt.close(fig)


def plot_roc(y, scores: dict, title: str, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, s):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title=title); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG(fname), dpi=130); plt.close(fig)


def plot_error_breakdown(df: pd.DataFrame, fname: str) -> None:
    """df needs: timestamp (target), weather_condition, y, p. Shows RMSE by hour / weekday / weather."""
    d = df.assign(err2=(df.y - df.p) ** 2, hour=df.timestamp.dt.hour, dow=df.timestamp.dt.dayofweek)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, col, lab in zip(axes, ["hour", "dow", "weather_condition"], ["Hour of day", "Day of week (0=Mon)", "Weather"]):
        r = d.groupby(col).err2.mean() ** .5
        ax.bar(r.index.astype(str), r.values, color="#3b6ea5"); ax.set(xlabel=lab, ylabel="RMSE (vehicles)")
    fig.suptitle("Volume forecast error breakdown (test window)"); fig.tight_layout()
    fig.savefig(FIG(fname), dpi=130); plt.close(fig)


def plot_residuals(y, p, fname: str) -> None:
    r = np.asarray(y) - np.asarray(p)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    sns.histplot(r, bins=60, kde=True, ax=axes[0], color="#3b6ea5"); axes[0].set(title="Residual distribution", xlabel="actual - predicted")
    axes[1].scatter(p, r, s=2, alpha=.15, color="#3b6ea5"); axes[1].axhline(0, color="k", lw=1)
    axes[1].set(title="Residuals vs prediction (heteroscedasticity check)", xlabel="predicted", ylabel="residual")
    fig.tight_layout(); fig.savefig(FIG(fname), dpi=130); plt.close(fig)


def plot_pred_vs_actual(df: pd.DataFrame, road_id: str, fname: str, days: int = 4) -> None:
    d = df[df.road_id == road_id].sort_values("timestamp").head(48 * days)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(d.timestamp, d.y, label="actual", color="k", lw=1.2)
    ax.plot(d.timestamp, d.p, label="predicted", color="#d6322b", lw=1.2)
    ax.set(title=f"{road_id}: predicted vs actual volume (first {days} test days)", ylabel="vehicles / 30 min"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG(fname), dpi=130); plt.close(fig)


def plot_importance(imp: pd.Series, title: str, fname: str, top: int = 20) -> None:
    s = imp.sort_values(ascending=True).tail(top)
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(s) + 1.2))
    ax.barh(s.index, s.values, color="#3b6ea5"); ax.set(title=title, xlabel="importance")
    fig.tight_layout(); fig.savefig(FIG(fname), dpi=130); plt.close(fig)
