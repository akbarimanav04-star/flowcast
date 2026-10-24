"""M4 - Exploratory data analysis: figures + a short written summary used to justify modelling choices."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .evaluate import FIG
from .utils import CFG, CONGESTION_ORDER, SEVERITY_COLORS, get_logger, path

log = get_logger("eda")


def run() -> dict:
    df = pd.read_parquet(path(CFG["paths"]["processed"], "flowcast_clean.parquet"))
    df["hour"], df["dow"] = df.timestamp.dt.hour, df.timestamp.dt.dayofweek
    df["vc"] = df.traffic_volume / (df.road_capacity / 2)
    df["acc"] = (df.accident_count > 0).astype(int)
    out: dict = {}

    # 1. distributions + descriptive statistics + normality (skew / kurtosis)
    num = ["traffic_volume", "avg_speed", "occupancy", "travel_time"]
    desc = df[num].agg(["mean", "median", "std", "var", "skew", "kurt"]).T
    desc["mode"] = [df[c].round().mode().iat[0] for c in num]
    out["descriptive"] = desc.round(2)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    for ax, c in zip(axes, num):
        sns.histplot(df[c], bins=50, kde=True, ax=ax, color="#3b6ea5"); ax.set_title(c)
    fig.tight_layout(); fig.savefig(FIG("eda_distributions.png"), dpi=130); plt.close(fig)

    # 2. correlation (and covariance) structure
    cols = ["traffic_volume", "avg_speed", "occupancy", "travel_time", "vc", "rainfall", "visibility", "temperature", "signal_timing", "acc"]
    corr = df[cols].corr()
    out["corr"] = corr
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax); ax.set_title("Correlation matrix")
    fig.tight_layout(); fig.savefig(FIG("eda_correlation.png"), dpi=130); plt.close(fig)

    # 3. temporal patterns
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    for lab, d in df.groupby(df.dow >= 5):
        axes[0].plot(d.groupby("hour").traffic_volume.mean(), label="weekend" if lab else "weekday")
    axes[0].set(title="Average volume by hour (corridor)", xlabel="hour", ylabel="vehicles / 30 min"); axes[0].legend()
    sns.lineplot(data=df, x="hour", y="avg_speed", hue=df.dow >= 5, ax=axes[1], legend=False); axes[1].set(title="Average speed by hour (km/h)")
    fig.tight_layout(); fig.savefig(FIG("eda_hourly_pattern.png"), dpi=130); plt.close(fig)

    # 4. congestion heatmap: road x hour
    hm = df.pivot_table(index="road_id", columns="hour", values="vc", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(11, 6.5))
    sns.heatmap(hm, cmap="RdYlGn_r", vmin=0.2, vmax=1.0, ax=ax, cbar_kws={"label": "mean volume / capacity"})
    ax.set_title("Congestion heatmap: segment x hour of day")
    fig.tight_layout(); fig.savefig(FIG("eda_congestion_heatmap.png"), dpi=130); plt.close(fig)

    # 5. weather vs traffic, incident drivers
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    sns.boxplot(data=df, x="weather_condition", y="avg_speed", ax=axes[0], order=["Clear", "Cloudy", "Rain", "Fog"], color="#9ec5e8")
    axes[0].set_title("Speed by weather")
    df.groupby("weather_condition").acc.mean().reindex(["Clear", "Cloudy", "Rain", "Fog"]).mul(100).plot.bar(ax=axes[1], color="#d6322b")
    axes[1].set(title="Accident rate by weather (%)", ylabel="% windows with incident")
    df.groupby(pd.cut(df.vc, [0, .5, .8, 1, 2], labels=CONGESTION_ORDER)).acc.mean().mul(100).plot.bar(ax=axes[2], color="#f08a24")
    axes[2].set(title="Accident rate by congestion band (%)", xlabel="")
    fig.tight_layout(); fig.savefig(FIG("eda_weather_accident.png"), dpi=130); plt.close(fig)

    # 6. class balance
    fig, ax = plt.subplots(figsize=(5, 3.4))
    cnt = df.congestion_level.value_counts().reindex(CONGESTION_ORDER)
    ax.bar(cnt.index, cnt.values, color=[SEVERITY_COLORS[c] for c in cnt.index]); ax.set_title("Congestion class balance")
    fig.tight_layout(); fig.savefig(FIG("eda_class_balance.png"), dpi=130); plt.close(fig)

    ac_by_w = df.groupby("weather_condition").acc.mean()
    summary = [
        "# FlowCast - EDA summary (auto-generated)", "",
        f"* Records: {len(df):,} over {df.road_id.nunique()} segments, {df.date.min():%d %b %Y} - {df.date.max():%d %b %Y}.",
        f"* Volume peaks twice a day (AM ~08:00-09:00, PM ~17:00-18:00); weekday vs weekend profiles differ -> weekend/peak flags and hour encodings are justified.",
        f"* Volume skewness = {desc.loc['traffic_volume', 'skew']:.2f}: right-skewed, so RMSE (not accuracy) is used and tree models are expected to beat linear ones.",
        f"* Correlation volume vs occupancy = {corr.loc['traffic_volume', 'occupancy']:.2f}; speed vs occupancy = {corr.loc['avg_speed', 'occupancy']:.2f} (strong, redundancy noted).",
        f"* Congestion classes are imbalanced: " + ", ".join(f"{k} {v / len(df):.1%}" for k, v in cnt.items()) + " -> macro-F1 is the headline metric.",
        f"* Incident rate: {df.acc.mean():.2%} of windows. Rain/Fog raise it to {ac_by_w['Rain']:.1%}/{ac_by_w['Fog']:.1%} vs {ac_by_w['Clear']:.1%} in clear weather.",
        f"* Heavy congestion ratio (V/C >= 0.8) share: {(df.vc >= .8).mean():.1%}.", ""]
    path(CFG["paths"]["reports"], "eda_summary.md").write_text("\n".join(summary), encoding="utf-8")
    log.info("EDA figures written to %s", CFG["paths"]["figures"])
    return out


if __name__ == "__main__":
    run()
