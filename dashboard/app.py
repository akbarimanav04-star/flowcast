import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import features as ft
from src import predict
from src.utils import CONGESTION_ORDER, SEVERITY_COLORS

st.set_page_config(page_title="FlowCast", page_icon="🚦", layout="wide")
REP, MOD, FIG = ROOT / "reports", ROOT / "models", ROOT / "reports" / "figures"
CMAP = SEVERITY_COLORS
HORIZON_LABEL = {1: "+30 min", 2: "+60 min", 3: "+90 min", 4: "+120 min"}


# ----------------------------------------------------------------------------- data access
@st.cache_data(show_spinner=False)
def load_clean() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data/processed/flowcast_clean.parquet")
    df["ts"] = df.timestamp.dt.tz_localize(None)
    df["vc"] = df.traffic_volume / (df.road_capacity / 2)
    return df


@st.cache_data(show_spinner=False)
def load_preds() -> pd.DataFrame:
    p = pd.read_parquet(REP / "predictions_test_xgb.parquet")
    p["ts"] = p.target_ts.dt.tz_localize(None)
    return p


@st.cache_data(show_spinner=False)
def load_dl() -> pd.DataFrame | None:
    f = REP / "predictions_test_dl.parquet"
    if not f.exists():
        return None
    d = pd.read_parquet(f); d["ts"] = d.target_ts.dt.tz_localize(None)
    return d


@st.cache_data(show_spinner=False)
def load_scores():
    sc = pd.read_csv(REP / "scoreboard_classical.csv")
    dl = pd.read_csv(REP / "scoreboard_dl.csv") if (REP / "scoreboard_dl.csv").exists() else pd.DataFrame()
    return sc, dl


@st.cache_data(show_spinner=False)
def cached_forecast(origin_iso: str) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data/processed/flowcast_clean.parquet")
    return predict.forecast(df, pd.Timestamp(origin_iso))


def sev_fig_layout(fig, h=420):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2))
    return fig


need = [ROOT / "data/processed/flowcast_clean.parquet", REP / "predictions_test_xgb.parquet", REP / "scoreboard_classical.csv", MOD / "meta.json"]
if not all(p.exists() for p in need):
    st.error("Model outputs not found. Run the pipeline first:  `python run_pipeline.py`")
    st.stop()

df, preds = load_clean(), load_preds()
roads = df[["road_id", "road_name"]].drop_duplicates().sort_values("road_id")
name_of = dict(zip(roads.road_id, roads.road_name))

# ----------------------------------------------------------------------------- navigation (flat: <= 3 clicks)
st.sidebar.title("🚦 FlowCast")
st.sidebar.caption("Northline Corridor · 25 segments · 30-min windows")
VIEWS = ["Live prediction", "Congestion heatmap", "Road comparison", "Historical trends", "Forecast visualisation",
         "Prediction confidence", "Weather vs traffic", "Model performance", "Feature importance",
         "Tools: upload & predict", "Tools: retrain", "Tools: report export"]
view = st.sidebar.radio("View", VIEWS, label_visibility="collapsed")
st.sidebar.markdown("**Severity colours**  \n" + "  \n".join(f"<span style='color:{c}'>■</span> {k}" for k, c in CMAP.items()), unsafe_allow_html=True)


# ============================================================================= 1. LIVE PREDICTION
if view == "Live prediction":
    st.header("Live prediction")
    st.caption("Forecast for every segment from a chosen origin window (default: latest window in the data).")
    last = df.ts.max()
    c1, c2, c3 = st.columns(3)
    d0 = c1.date_input("Origin date", value=last.date(), min_value=(df.ts.min() + pd.Timedelta(days=2)).date(), max_value=last.date())
    slots = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]
    t0 = c2.selectbox("Origin window start", slots, index=slots.index(f"{last.hour:02d}:{last.minute:02d}") if d0 == last.date() else 34)
    hz = c3.selectbox("Horizon", list(HORIZON_LABEL), format_func=HORIZON_LABEL.get)
    origin = pd.Timestamp(f"{d0} {t0}", tz="Asia/Kolkata")
    try:
        fc = cached_forecast(origin.isoformat())
    except Exception as e:
        st.warning(f"Cannot forecast from this origin: {e}"); st.stop()
    seg = st.multiselect("Filter segments (prediction interface)", list(roads.road_id), format_func=lambda r: f"{r} {name_of[r]}")
    f = fc[fc.horizon == hz].copy()
    if seg:
        f = f[f.road_id.isin(seg)]
    tgt = f.target_ts.iloc[0]
    st.subheader(f"Forecast for {pd.Timestamp(tgt).tz_localize(None):%a %d %b %Y %H:%M} ({HORIZON_LABEL[hz]})")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Corridor volume (veh/30 min)", f"{f.volume_pred.sum():,.0f}")
    m2.metric("Heavy / Severe segments", int(f.congestion.isin(["Heavy", "Severe"]).sum()))
    m3.metric("High-risk segments", int((f.risk_band == "High").sum()))
    m4.metric("Avg predicted travel time", f"{f.travel_time_pred.mean():.1f} min")
    actual = df[df.timestamp == tgt][["road_id", "traffic_volume", "congestion_level"]].rename(columns={"traffic_volume": "actual_volume", "congestion_level": "actual_congestion"})
    f = f.merge(actual, on="road_id", how="left")
    mp = px.scatter_map(f, lat="latitude", lon="longitude", color="congestion", color_discrete_map=CMAP, category_orders={"congestion": CONGESTION_ORDER},
                        size="volume_pred", size_max=26, zoom=11, hover_name="road_name",
                        hover_data={"volume_pred": ":.0f", "accident_prob": ":.3f", "congestion_conf": ":.2f", "latitude": False, "longitude": False}, map_style="open-street-map")
    st.plotly_chart(sev_fig_layout(mp, 460), width="stretch")
    show = f.assign(confidence=(f.congestion_conf * 100).round(0).astype(int).astype(str) + "%",
                    volume=f.volume_pred.round(0).astype(int), interval=f.volume_lo.round(0).astype(int).astype(str) + " – " + f.volume_hi.round(0).astype(int).astype(str),
                    travel_min=f.travel_time_pred.round(2), accident_risk=(f.accident_prob * 100).round(2).astype(str) + "%")
    cols = ["road_id", "road_name", "volume", "interval", "congestion", "confidence", "travel_min", "accident_risk", "risk_band"]
    if f.actual_volume.notna().any():
        cols += ["actual_volume", "actual_congestion"]
    st.dataframe(show.sort_values("accident_prob", ascending=False)[cols].style.apply(lambda s: [f"background-color:{CMAP.get(v, '')}33" if s.name in ("congestion", "actual_congestion") else "" for v in s], axis=0), width="stretch", hide_index=True)
    st.caption("Incident Response: the table is ranked by accident-risk probability. Risk band thresholds come from the validation split.")
    st.download_button("Download this forecast (CSV)", fc.to_csv(index=False).encode(), "flowcast_forecast.csv")


# ============================================================================= 2. HEATMAP
elif view == "Congestion heatmap":
    st.header("Congestion heatmap")
    mode = st.radio("Show", ["Selected date", "Typical weekday", "Typical weekend"], horizontal=True)
    if mode == "Selected date":
        day = st.date_input("Date", value=df.ts.max().date(), min_value=df.ts.min().date(), max_value=df.ts.max().date())
        d = df[df.ts.dt.date == day]
        z = d.pivot_table(index="road_id", columns=d.ts.dt.strftime("%H:%M"), values="vc")
    else:
        wk = df.ts.dt.dayofweek >= 5
        d = df[wk] if mode == "Typical weekend" else df[~wk]
        z = d.pivot_table(index="road_id", columns=d.ts.dt.strftime("%H:%M"), values="vc")
    codes = np.select([z < .5, z < .8, z < 1.0], [0, 1, 2], default=3).astype(float)
    codes[z.isna().values] = np.nan
    scale = [[0, CMAP["Free-flow"]], [.25, CMAP["Free-flow"]], [.25, CMAP["Moderate"]], [.5, CMAP["Moderate"]], [.5, CMAP["Heavy"]], [.75, CMAP["Heavy"]], [.75, CMAP["Severe"]], [1, CMAP["Severe"]]]
    fig = go.Figure(go.Heatmap(z=codes, x=list(z.columns), y=[f"{r} {name_of[r]}" for r in z.index], colorscale=scale, zmin=0, zmax=3, showscale=False,
                               customdata=z.values.round(2), hovertemplate="%{y}<br>%{x}<br>V/C %{customdata}<extra></extra>"))
    fig.update_layout(height=640, margin=dict(l=10, r=10, t=30, b=10), xaxis=dict(dtick=4), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")
    st.caption("Colour = congestion band from volume / (capacity/2): Free-flow < 0.5 · Moderate < 0.8 · Heavy < 1.0 · Severe ≥ 1.0")


# ============================================================================= 3. ROAD COMPARISON
elif view == "Road comparison":
    st.header("Road comparison")
    g = df.groupby("road_id")
    comp = pd.DataFrame({"avg_volume": g.traffic_volume.mean(), "avg_speed": g.avg_speed.mean(), "avg_travel_min": g.travel_time.mean(),
                         "buffer_index": (g.travel_time.quantile(.95) - g.travel_time.mean()) / g.travel_time.mean(),
                         "heavy_or_severe_%": g.apply(lambda x: x.congestion_level.isin(["Heavy", "Severe"]).mean() * 100),
                         "incident_rate_%": g.apply(lambda x: (x.accident_count > 0).mean() * 100)}).round(2)
    comp.insert(0, "road_name", [name_of[r] for r in comp.index])
    pick = st.multiselect("Segments", list(comp.index), default=list(comp.sort_values("avg_volume", ascending=False).index[:8]), format_func=lambda r: f"{r} {name_of[r]}")
    c = comp.loc[pick]
    cols = st.columns(2)
    for i, (m, t) in enumerate([("avg_volume", "Average volume (veh/30 min)"), ("avg_speed", "Average speed (km/h)"), ("buffer_index", "Travel-time reliability (buffer index, lower = better)"), ("heavy_or_severe_%", "Share of windows Heavy/Severe (%)")]):
        cols[i % 2].plotly_chart(sev_fig_layout(px.bar(c.reset_index(), x="road_id", y=m, title=t, hover_data=["road_name"], color_discrete_sequence=["#3b6ea5"]), 320), width="stretch")
    st.dataframe(c, width="stretch")


# ============================================================================= 4. HISTORICAL TRENDS
elif view == "Historical trends":
    st.header("Historical trends")
    c1, c2, c3 = st.columns([2, 2, 1])
    sel = c1.multiselect("Segments", list(roads.road_id), default=["NL-007"], format_func=lambda r: f"{r} {name_of[r]}")
    rng = c2.date_input("Date range", value=(df.ts.max().date() - pd.Timedelta(days=6), df.ts.max().date()), min_value=df.ts.min().date(), max_value=df.ts.max().date())
    res = c3.selectbox("Resolution", ["30min", "1h", "1D"])
    if len(rng) == 2 and sel:
        d = df[df.road_id.isin(sel) & (df.ts.dt.date >= rng[0]) & (df.ts.dt.date <= rng[1])]
        r = d.set_index("ts").groupby("road_id").resample(res).agg(volume=("traffic_volume", "mean"), speed=("avg_speed", "mean")).reset_index()
        st.plotly_chart(sev_fig_layout(px.line(r, x="ts", y="volume", color="road_id", title="Volume over time")), width="stretch")
        st.plotly_chart(sev_fig_layout(px.line(r, x="ts", y="speed", color="road_id", title="Average speed over time (km/h)"), 320), width="stretch")
        hp = d.assign(hour=d.ts.dt.hour, kind=np.where(d.ts.dt.dayofweek >= 5, "weekend", "weekday")).groupby(["hour", "kind"]).traffic_volume.mean().reset_index()
        st.plotly_chart(sev_fig_layout(px.line(hp, x="hour", y="traffic_volume", color="kind", title="Peak-hour pattern (selected segments & range)"), 320), width="stretch")


# ============================================================================= 5. FORECAST VISUALISATION
elif view == "Forecast visualisation":
    st.header("Forecast visualisation")
    st.caption("Predicted-vs-actual across the unseen test window (May 2025).")
    c1, c2, c3 = st.columns(3)
    rid = c1.selectbox("Segment", list(roads.road_id), index=6, format_func=lambda r: f"{r} {name_of[r]}")
    days = sorted(preds.ts.dt.date.unique())
    day = c2.selectbox("Day", days, index=0)
    hset = c3.multiselect("Horizons", list(HORIZON_LABEL), default=[1, 4], format_func=HORIZON_LABEL.get)
    p = preds[(preds.road_id == rid) & (preds.ts.dt.date == day)]
    fig = go.Figure()
    a = p[p.horizon == 1].sort_values("ts")
    fig.add_trace(go.Scatter(x=a.ts, y=a.volume_true, name="actual", line=dict(color="black", width=2)))
    for h in hset:
        q = p[p.horizon == h].sort_values("ts"); fig.add_trace(go.Scatter(x=q.ts, y=q.volume_pred, name=f"XGBoost {HORIZON_LABEL[h]}", mode="lines"))
    dl = load_dl()
    if dl is not None:
        q = dl[(dl.road_id == rid) & (dl.ts.dt.date == day)].sort_values("ts")
        fig.add_trace(go.Scatter(x=q.ts, y=q["LSTM_volume_pred"], name="LSTM +30 min", line=dict(dash="dot")))
    st.plotly_chart(sev_fig_layout(fig.update_layout(yaxis_title="vehicles / 30 min", title=f"{rid} {name_of[rid]} - {day}")), width="stretch")
    err = preds.assign(ae=(preds.volume_true - preds.volume_pred).abs(), ape=(preds.volume_true - preds.volume_pred).abs() / preds.volume_true).groupby("horizon").agg(MAE=("ae", "mean"), MAPE=("ape", "mean"))
    err["RMSE"] = preds.assign(e2=(preds.volume_true - preds.volume_pred) ** 2).groupby("horizon").e2.mean() ** .5
    err.index = [HORIZON_LABEL[i] for i in err.index]
    st.subheader("Accuracy decay with horizon (all segments, test window)")
    st.dataframe(err.round(3), width="stretch")


# ============================================================================= 6. CONFIDENCE
elif view == "Prediction confidence":
    st.header("Prediction confidence")
    st.caption("Volume bands are split-conformal 90% intervals (per segment, calibrated on the validation window). Congestion confidence = top class probability.")
    c1, c2, c3 = st.columns(3)
    rid = c1.selectbox("Segment", list(roads.road_id), index=6, format_func=lambda r: f"{r} {name_of[r]}")
    day = c2.selectbox("Day", sorted(preds.ts.dt.date.unique()), index=7)
    h = c3.selectbox("Horizon", list(HORIZON_LABEL), format_func=HORIZON_LABEL.get)
    p = preds[(preds.road_id == rid) & (preds.ts.dt.date == day) & (preds.horizon == h)].sort_values("ts")
    fig = go.Figure([go.Scatter(x=p.ts, y=p.volume_hi, line=dict(width=0), showlegend=False),
                     go.Scatter(x=p.ts, y=p.volume_lo, fill="tonexty", fillcolor="rgba(59,110,165,.25)", line=dict(width=0), name="90% interval"),
                     go.Scatter(x=p.ts, y=p.volume_pred, name="predicted", line=dict(color="#3b6ea5")),
                     go.Scatter(x=p.ts, y=p.volume_true, name="actual", line=dict(color="black"))])
    st.plotly_chart(sev_fig_layout(fig.update_layout(yaxis_title="vehicles / 30 min")), width="stretch")
    cov = preds.assign(inside=(preds.volume_true >= preds.volume_lo) & (preds.volume_true <= preds.volume_hi), width=preds.volume_hi - preds.volume_lo).groupby("horizon").agg(coverage=("inside", "mean"), mean_width=("width", "mean"))
    cov.index = [HORIZON_LABEL[i] for i in cov.index]
    c1, c2 = st.columns(2)
    c1.subheader("Empirical interval coverage (target 90%)"); c1.dataframe(cov.round(3), width="stretch")
    pc = preds[preds.horizon == h].assign(bin=lambda x: pd.cut(x.congestion_conf, [0, .5, .7, .9, 1.0]), ok=lambda x: x.congestion_pred == x.congestion_true)
    cal = pc.groupby("bin", observed=True).agg(accuracy=("ok", "mean"), n=("ok", "size")).reset_index(); cal["bin"] = cal["bin"].astype(str)
    c2.subheader("Congestion confidence calibration"); c2.plotly_chart(sev_fig_layout(px.bar(cal, x="bin", y="accuracy", text="n", color_discrete_sequence=["#3b6ea5"]), 280), width="stretch")


# ============================================================================= 7. WEATHER VS TRAFFIC
elif view == "Weather vs traffic":
    st.header("Weather vs traffic")
    order = ["Clear", "Cloudy", "Rain", "Fog"]
    c1, c2 = st.columns(2)
    c1.plotly_chart(sev_fig_layout(px.box(df.sample(40000, random_state=1), x="weather_condition", y="avg_speed", category_orders={"weather_condition": order}, title="Speed by weather"), 340), width="stretch")
    t = df.assign(peak=df.ts.dt.hour.isin(ft.F["peak_hours"])).groupby(["weather_condition", "peak"]).traffic_volume.mean().reset_index()
    c2.plotly_chart(sev_fig_layout(px.bar(t, x="weather_condition", y="traffic_volume", color="peak", barmode="group", category_orders={"weather_condition": order}, title="Average volume by weather (peak vs off-peak)"), 340), width="stretch")
    c1, c2 = st.columns(2)
    rate = df.assign(inc=df.accident_count > 0).groupby("weather_condition").inc.mean().mul(100).reindex(order).reset_index()
    c1.plotly_chart(sev_fig_layout(px.bar(rate, x="weather_condition", y="inc", title="Incident rate by weather (% of windows)", color_discrete_sequence=["#d6322b"]), 320), width="stretch")
    p1 = preds[preds.horizon == 1]
    e = p1.assign(e2=(p1.volume_true - p1.volume_pred) ** 2).groupby("weather_condition").e2.mean().pow(.5).reindex(order).reset_index()
    c2.plotly_chart(sev_fig_layout(px.bar(e, x="weather_condition", y="e2", title="Model RMSE by weather (test, +30 min)", color_discrete_sequence=["#3b6ea5"]), 320), width="stretch")
    rr = df[df.rainfall > 0].sample(min(8000, int((df.rainfall > 0).sum())), random_state=1)
    fig = px.scatter(rr, x="rainfall", y="avg_speed", opacity=.25, title="Rainfall (mm/h) vs speed when raining")
    k, b0 = np.polyfit(rr.rainfall, rr.avg_speed, 1)                      # least-squares trend line (NumPy)
    xs = np.array([rr.rainfall.min(), rr.rainfall.max()])
    fig.add_trace(go.Scatter(x=xs, y=k * xs + b0, name=f"trend ({k:.2f} km/h per mm)", line=dict(color="black")))
    st.plotly_chart(sev_fig_layout(fig, 340), width="stretch")


# ============================================================================= 8. MODEL PERFORMANCE
elif view == "Model performance":
    st.header("Model performance")
    sc, dl = load_scores()
    allsc = pd.concat([sc, dl], ignore_index=True)
    te = allsc[allsc.split == "te"]
    st.subheader("Success criteria (PRD §3.3) on the unseen test window")
    vol, cong = te[te.task == "volume"], te[(te.task == "congestion") & te.MacroF1.notna()]
    acc = te[te.task == "accident"]
    best_vol = vol.loc[vol.MAPE.idxmin()]
    classical_best_rmse = vol[~vol.model.str.contains("LSTM")].RMSE.min()
    dl_vol = vol[vol.model.str.contains("LSTM")]
    rows = [("Volume MAPE ≤ 12%", f"{best_vol.MAPE * 100:.1f}% ({best_vol.model})", best_vol.MAPE <= .12),
            ("Congestion macro-F1 ≥ 0.80", f"{cong.MacroF1.max():.3f} ({cong.loc[cong.MacroF1.idxmax()].model})", cong.MacroF1.max() >= .8),
            ("Accident ROC-AUC ≥ 0.75", f"{acc.ROC_AUC.max():.3f} ({acc.loc[acc.ROC_AUC.idxmax()].model})", acc.ROC_AUC.max() >= .75)]
    if len(dl_vol):
        rows.append(("Sequence model beats best classical on volume RMSE", f"best LSTM {dl_vol.RMSE.min():.1f} vs classical {classical_best_rmse:.1f}", dl_vol.RMSE.min() < classical_best_rmse))
    st.table(pd.DataFrame([{"Criterion": a, "Achieved": b, "Met": "✅" if c else "❌ not met"} for a, b, c in rows]))
    st.caption("Accident risk: see the written report - a same-window 'oracle' model reaches only ~0.63 AUC, so the label is dominated by randomness in this dataset.")
    task = st.selectbox("Scoreboard for task", ["volume", "travel_time", "congestion", "accident"])
    t = allsc[(allsc.task == task) & (allsc.split == "te")].dropna(axis=1, how="all").drop(columns=["task", "split", "best_params"], errors="ignore").set_index("model")
    st.dataframe(t.round(4), width="stretch")
    c1, c2 = st.columns(2)
    for col, img in ((c1, "confusion_congestion.png"), (c2, "roc_accident.png")):
        if (FIG / img).exists():
            col.image(str(FIG / img))
    c1, c2 = st.columns(2)
    for col, img in ((c1, "dl_loss_curves.png"), (c2, "gd_loss_curve.png")):
        if (FIG / img).exists():
            col.image(str(FIG / img))
    if (FIG / "volume_error_breakdown.png").exists():
        st.image(str(FIG / "volume_error_breakdown.png"))


# ============================================================================= 9. FEATURE IMPORTANCE
elif view == "Feature importance":
    st.header("Feature importance")
    import joblib
    c1, c2 = st.columns(2)
    task = c1.selectbox("Target", ["volume", "travel_time", "congestion", "accident"])
    h = c2.selectbox("Horizon", list(HORIZON_LABEL), format_func=HORIZON_LABEL.get)
    m = joblib.load(MOD / f"xgb_{task}_h{h}.joblib")
    imp = pd.Series(m.feature_importances_, ft.FEATURES).sort_values(ascending=False)
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(sev_fig_layout(px.bar(imp.head(20)[::-1], orientation="h", title=f"Top drivers - {task} {HORIZON_LABEL[h]} (XGBoost gain)", color_discrete_sequence=["#3b6ea5"]), 560), width="stretch")
    grp = pd.Series({g: imp[cols].sum() for g, cols in ft.GROUPS.items()}).sort_values()
    c2.plotly_chart(sev_fig_layout(px.bar(grp, orientation="h", title="By feature group", color_discrete_sequence=["#f08a24"]), 560), width="stretch")


# ============================================================================= TOOL: UPLOAD & PREDICT
elif view == "Tools: upload & predict":
    st.header("Upload new data & predict")
    st.write("Upload a traffic file with the same columns as `traffic_sensor_log.csv` (needs at least 2 days of history per segment, "
             "dates inside Jan–May 2025 so weather/calendar can be joined). It is validated, cleaned and forecast from its last window.")
    up = st.file_uploader("traffic CSV", type="csv")
    if up is not None:
        raw = pd.read_csv(up)
        st.write(f"{len(raw):,} rows uploaded")
        try:
            with st.spinner("Validating, cleaning, forecasting…"):
                cleaned, fc = predict.predict_from_upload(raw)
            st.success(f"Cleaned to {len(cleaned):,} rows. Forecast origin: {cleaned.timestamp.max():%Y-%m-%d %H:%M}")
            h = st.selectbox("Horizon", list(HORIZON_LABEL), format_func=HORIZON_LABEL.get)
            st.dataframe(fc[fc.horizon == h][["road_id", "road_name", "volume_pred", "volume_lo", "volume_hi", "congestion", "congestion_conf", "travel_time_pred", "accident_prob", "risk_band"]].round(3), hide_index=True, width="stretch")
            st.download_button("Download forecast CSV", fc.to_csv(index=False).encode(), "uploaded_forecast.csv")
        except Exception as e:
            st.error(f"Could not process the file: {e}")
    q = ROOT / "data/interim/quarantine_traffic.csv"
    if q.exists():
        with st.expander("Quarantined rows from the last cleaning run (rows breaking range rules, nothing silently dropped)"):
            st.dataframe(pd.read_csv(q).head(200), width="stretch")


# ============================================================================= TOOL: RETRAIN
elif view == "Tools: retrain":
    st.header("Model-training interface")
    st.write("**Fast retrain** refits the XGBoost volume model (+30 min) with the tuned hyper-parameters on the training window and reports test metrics (~15 s). "
             "The deployed models are not overwritten. For the full tuning of every model run `python run_pipeline.py` in a terminal.")
    if st.button("Run fast retrain"):
        from src import ml_models
        with st.spinner("Training…"):
            res = ml_models.retrain_fast()
        st.json(res)


# ============================================================================= TOOL: REPORT EXPORT
elif view == "Tools: report export":
    st.header("Reports & insights")
    rng = st.date_input("Date range", value=(df.ts.min().date(), df.ts.max().date()), min_value=df.ts.min().date(), max_value=df.ts.max().date())
    if len(rng) == 2:
        d = df[(df.ts.dt.date >= rng[0]) & (df.ts.dt.date <= rng[1])]
        by = d.groupby("road_id")
        top_c = by.apply(lambda x: x.congestion_level.isin(["Heavy", "Severe"]).mean()).sort_values(ascending=False).head(5)
        top_i = by.apply(lambda x: (x.accident_count > 0).sum()).sort_values(ascending=False).head(5)
        pk = d.groupby(d.ts.dt.hour).traffic_volume.mean()
        wi = d.groupby("weather_condition").agg(speed=("avg_speed", "mean"), incident=("accident_count", lambda s: (s > 0).mean() * 100))
        md = [f"# FlowCast insights: {rng[0]} to {rng[1]}", "", f"* Windows analysed: {len(d):,} across {d.road_id.nunique()} segments.",
              f"* Corridor average volume {d.traffic_volume.mean():.0f} veh/30 min; average speed {d.avg_speed.mean():.1f} km/h.",
              f"* Peak hour: {pk.idxmax():02d}:00 (avg {pk.max():.0f} veh per segment-window); quietest: {pk.idxmin():02d}:00.",
              f"* Share of windows Heavy or Severe: {d.congestion_level.isin(['Heavy', 'Severe']).mean():.1%}.", "",
              "## Most congested segments (share of Heavy/Severe windows)"] + [f"* {r} {name_of[r]}: {v:.1%}" for r, v in top_c.items()] + \
             ["", "## Segments with most reported incidents"] + [f"* {r} {name_of[r]}: {int(v)}" for r, v in top_i.items()] + \
             ["", "## Weather impact", "", "| Weather | Avg speed (km/h) | Incident rate (%) |", "|---|---|---|"] + [f"| {w} | {r.speed:.1f} | {r.incident:.2f} |" for w, r in wi.iterrows()]
        st.markdown("\n".join(md))
        st.download_button("Download report (Markdown)", "\n".join(md).encode(), "flowcast_insights.md")
