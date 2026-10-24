"""M6 - Deep-learning engine: an LSTM (and Bi-LSTM) forecaster written from scratch in NumPy.

Everything is hand-written: the forward pass, back-propagation through time (BPTT), the Adam optimiser,
dropout, gradient clipping, early stopping and learning-rate reduction on plateau. No pre-trained weights.

Architecture (multi-task)
    last N windows of dynamic features --> LSTM (x1 or bidirectional) --> final hidden state h_N
    [h_N , context features of the target window] --> Dense(ReLU) --> two heads:
        volume (regression)  and  congestion (4-class softmax)
Loss = MSE(volume, z-scored) + 0.5 * cross-entropy(congestion)
"""
from __future__ import annotations

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from . import evaluate as ev
from .features import (FEATURES, add_base_features, build_supervised, fit_scaler, split_by_time)
from .utils import CFG, get_logger, path, set_seed

log = get_logger("dl")
DL = CFG["dl"]
DYN_SCALED = ["traffic_volume", "avg_speed", "occupancy", "travel_time", "signal_timing", "vc_ratio", "rainfall", "visibility", "temperature"]
CTX_COLS = ["t_hour_sin", "t_hour_cos", "t_dow_sin", "t_dow_cos", "t_weekend", "t_peak", "t_holiday", "t_event", "t_roadwork",
            "t_holiday_x_peak", "t_days_to_event", "road_capacity", "latitude", "longitude", "road_idx",
            "share_2W", "share_Car", "share_LCV", "share_HCV", "traffic_volume_lag48", "avg_speed_lag48"]
F32 = np.float32


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# --------------------------------------------------------------------------------------------- data
def _scale_cols(df: pd.DataFrame, scaler, cols) -> pd.DataFrame:
    """Apply the TRAIN-fitted scaler parameters to selected columns (min-max or z-score as fitted)."""
    out = df[cols].copy().astype(float)
    for c in cols:
        if c in scaler.params:
            a, b = scaler.params[c]
            out[c] = (out[c] - a) / ((b - a) or 1.0) if c in scaler.minmax_cols else (out[c] - a) / b
    if "road_idx" in cols:
        out["road_idx"] = df["road_idx"] / 25.0
    return out


def make_arrays(base: pd.DataFrame, origins: pd.DataFrame, scaler, seq_len: int):
    """Return (sequences [n, N, D], context [n, C]) for each origin row (road_id, timestamp)."""
    ts = base["timestamp"]
    dyn = _scale_cols(base, scaler, DYN_SCALED)
    hour = ts.dt.hour + ts.dt.minute / 60
    dyn["hsin"], dyn["hcos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    dyn["acc"] = base["accident_now"].values
    dyn = dyn.astype(F32)
    t0 = ts.min()
    pos_of = lambda s: ((s - t0) / pd.Timedelta(minutes=30)).round().astype(int).values   # noqa: E731
    seqs, key = [], {}
    offset = 0
    for rid, idx in base.groupby("road_id", sort=False).indices.items():
        A = dyn.iloc[idx].values
        w = sliding_window_view(A, seq_len, axis=0).transpose(0, 2, 1)      # (T-N+1, N, D)
        seqs.append(w)
        p = pos_of(ts.iloc[idx])
        for k, pp in enumerate(p[seq_len - 1:]):
            key[(rid, pp)] = offset + k
        offset += len(w)
    big = np.concatenate(seqs, axis=0)
    rows = np.array([key[(r, p)] for r, p in zip(origins["road_id"], pos_of(origins["timestamp"]))])
    ctx = _scale_cols(origins, scaler, CTX_COLS).values.astype(F32)
    return big[rows].astype(F32), ctx


# --------------------------------------------------------------------------------------------- model
class LSTMForecaster:
    """Single-layer (optionally bidirectional) LSTM + dense head, trained with hand-written BPTT."""

    def __init__(self, d_in: int, d_ctx: int, hidden: int = 32, dense: int = 32, bidirectional: bool = False,
                 dropout: float = 0.2, seed: int = 42):
        rs = np.random.RandomState(seed)
        self.H, self.D, self.bi, self.p = hidden, d_in, bidirectional, dropout
        self.cfg = dict(d_in=d_in, d_ctx=d_ctx, hidden=hidden, dense=dense, bidirectional=bidirectional, dropout=dropout)
        glorot = lambda a, b: (rs.randn(a, b) * np.sqrt(2.0 / (a + b))).astype(F32)   # noqa: E731
        self.P = {"W": glorot(d_in + hidden, 4 * hidden), "b": np.zeros(4 * hidden, F32)}
        self.P["b"][hidden:2 * hidden] = 1.0                       # forget-gate bias = 1 (standard trick)
        if bidirectional:
            self.P["Wr"], self.P["br"] = glorot(d_in + hidden, 4 * hidden), self.P["b"].copy()
        n_h = hidden * (2 if bidirectional else 1) + d_ctx
        self.P.update({"W1": glorot(n_h, dense), "b1": np.zeros(dense, F32),
                       "Wv": glorot(dense, 1), "bv": np.zeros(1, F32), "Wc": glorot(dense, 4), "bc": np.zeros(4, F32)})
        self.m = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.t = 0

    # ---- LSTM over a (B, N, D) batch; returns last hidden state and the cache for BPTT
    def _lstm_forward(self, X, W, b):
        B, N, _ = X.shape; H = self.H
        h, c = np.zeros((B, H), F32), np.zeros((B, H), F32)
        cache = []
        for t in range(N):
            xh = np.concatenate([X[:, t], h], axis=1)
            z = xh @ W + b                                          # (B, 4H)  <- the matrix product
            i, f, g, o = _sigmoid(z[:, :H]), _sigmoid(z[:, H:2 * H]), np.tanh(z[:, 2 * H:3 * H]), _sigmoid(z[:, 3 * H:])
            c_new = f * c + i * g
            tc = np.tanh(c_new)
            h_new = o * tc
            cache.append((xh, i, f, g, o, c, tc))
            h, c = h_new, c_new
        return h, cache

    def _lstm_backward(self, dh_last, cache, W):
        H = self.H
        dW, db = np.zeros_like(W), np.zeros(4 * H, F32)
        dh, dc = dh_last, np.zeros_like(dh_last)
        for xh, i, f, g, o, c_prev, tc in reversed(cache):
            do = dh * tc
            dc = dc + dh * o * (1 - tc ** 2)
            dz = np.concatenate([dc * g * i * (1 - i), dc * c_prev * f * (1 - f), dc * i * (1 - g ** 2), do * o * (1 - o)], axis=1)
            dW += xh.T @ dz
            db += dz.sum(0)
            dh = (dz @ W.T)[:, self.D:]
            dc = dc * f
        return dW, db

    def forward(self, X, ctx, train: bool, rs=None):
        P = self.P
        h_f, cache_f = self._lstm_forward(X, P["W"], P["b"])
        parts, caches = [h_f], {"f": cache_f}
        if self.bi:
            h_r, cache_r = self._lstm_forward(X[:, ::-1], P["Wr"], P["br"])
            parts.append(h_r); caches["r"] = cache_r
        hcat = np.concatenate(parts, axis=1)
        mask0 = mask1 = None
        if train and self.p > 0:
            mask0 = (rs.rand(*hcat.shape) > self.p).astype(F32) / (1 - self.p)
            hcat = hcat * mask0
        u = np.concatenate([hcat, ctx], axis=1)
        a1 = u @ P["W1"] + P["b1"]
        r1 = np.maximum(a1, 0)
        if train and self.p > 0:
            mask1 = (rs.rand(*r1.shape) > self.p).astype(F32) / (1 - self.p)
            r1 = r1 * mask1
        vol = (r1 @ P["Wv"] + P["bv"])[:, 0]
        logits = r1 @ P["Wc"] + P["bc"]
        return vol, logits, (caches, hcat.shape[1], mask0, mask1, u, a1, r1)

    def loss_and_grads(self, X, ctx, y_vol, y_cls, lam=0.5, rs=None):
        P, B = self.P, len(X)
        vol, logits, (caches, nh, mask0, mask1, u, a1, r1) = self.forward(X, ctx, True, rs)
        e = vol - y_vol
        ex = np.exp(logits - logits.max(1, keepdims=True)); prob = ex / ex.sum(1, keepdims=True)
        ce = -np.log(prob[np.arange(B), y_cls] + 1e-9).mean()
        loss = float((e ** 2).mean() + lam * ce)
        dvol = (2 * e / B)[:, None].astype(F32)
        dlog = prob.copy(); dlog[np.arange(B), y_cls] -= 1; dlog = (lam / B * dlog).astype(F32)
        G = {"Wv": r1.T @ dvol, "bv": dvol.sum(0), "Wc": r1.T @ dlog, "bc": dlog.sum(0)}
        dr1 = dvol @ P["Wv"].T + dlog @ P["Wc"].T
        if mask1 is not None:
            dr1 = dr1 * mask1
        da1 = dr1 * (a1 > 0)
        G["W1"], G["b1"] = u.T @ da1, da1.sum(0)
        du = da1 @ P["W1"].T
        dh = du[:, :nh]
        if mask0 is not None:
            dh = dh * mask0
        G["W"], G["b"] = self._lstm_backward(dh[:, :self.H], caches["f"], P["W"])
        if self.bi:
            G["Wr"], G["br"] = self._lstm_backward(dh[:, self.H:], caches["r"], P["Wr"])
        return loss, G

    def adam_step(self, G, lr, clip=5.0, wd=1e-5, b1=0.9, b2=0.999, eps=1e-8):
        norm = np.sqrt(sum(float((g ** 2).sum()) for g in G.values()))
        scale = min(1.0, clip / (norm + 1e-9))                       # global gradient-norm clipping
        self.t += 1
        for k, g in G.items():
            g = g * scale + wd * self.P[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh, vh = self.m[k] / (1 - b1 ** self.t), self.v[k] / (1 - b2 ** self.t)
            self.P[k] -= (lr * mh / (np.sqrt(vh) + eps)).astype(F32)

    def predict(self, X, ctx, batch=4096):
        vols, probs = [], []
        for s in range(0, len(X), batch):
            v, lg, _ = self.forward(X[s:s + batch], ctx[s:s + batch], False)
            ex = np.exp(lg - lg.max(1, keepdims=True))
            vols.append(v); probs.append(ex / ex.sum(1, keepdims=True))
        return np.concatenate(vols), np.concatenate(probs)

    def save(self, fname: str) -> None:
        np.savez(path(CFG["paths"]["models"], fname), **self.P, cfg=json.dumps(self.cfg))

    @classmethod
    def load(cls, fname: str) -> "LSTMForecaster":
        z = np.load(path(CFG["paths"]["models"], fname), allow_pickle=False)
        m = cls(**json.loads(str(z["cfg"])))
        m.P = {k: z[k].astype(F32) for k in m.P}
        return m


# --------------------------------------------------------------------------------------------- training
def train_model(name, Xtr, Ctr, ytr_v, ytr_c, Xva, Cva, yva_v, yva_c, bidirectional=False):
    set_seed()
    rs = np.random.RandomState(CFG["seed"])
    model = LSTMForecaster(Xtr.shape[2], Ctr.shape[1], DL["hidden_size"], 32, bidirectional, DL["dropout"], CFG["seed"])
    lr, best, best_P, bad, plateau = DL["lr"], np.inf, None, 0, 0
    hist = {"train": [], "val": [], "lr": []}
    n, bs = len(Xtr), DL["batch_size"]
    for ep in range(1, DL["max_epochs"] + 1):
        t0 = time.time()
        order = rs.permutation(n)
        tl = []
        for s in range(0, n, bs):
            ix = order[s:s + bs]
            loss, G = model.loss_and_grads(Xtr[ix], Ctr[ix], ytr_v[ix], ytr_c[ix], rs=rs)
            model.adam_step(G, lr)
            tl.append(loss)
        v, prob = model.predict(Xva, Cva)
        ce = -np.log(prob[np.arange(len(yva_c)), yva_c] + 1e-9).mean()
        vl = float(((v - yva_v) ** 2).mean() + 0.5 * ce)
        hist["train"].append(float(np.mean(tl))); hist["val"].append(vl); hist["lr"].append(lr)
        log.info("[%s] epoch %2d  train %.4f  val %.4f  lr %.4f  (%.0fs)", name, ep, hist["train"][-1], vl, lr, time.time() - t0)
        if vl < best - 1e-4:
            best, bad, plateau = vl, 0, 0
            best_P = {k: x.copy() for k, x in model.P.items()}        # restore-best-weights
        else:
            bad += 1; plateau += 1
            if plateau >= 2:
                lr, plateau = lr * 0.5, 0                              # reduce LR on plateau
            if bad >= DL["patience"]:
                log.info("[%s] early stopping at epoch %d", name, ep); break
    model.P = best_P
    return model, hist


def plot_curves(hist_by_name: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, h in hist_by_name.items():
        ax.plot(h["train"], label=f"{name} train"); ax.plot(h["val"], "--", label=f"{name} validation")
    ax.set(xlabel="epoch", ylabel="loss (MSE + 0.5*CE)", title="LSTM training vs validation loss"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(ev.FIG("dl_loss_curves.png"), dpi=130); plt.close(fig)


def run(quick: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(path(CFG["paths"]["processed"], "flowcast_clean.parquet"))
    base = add_base_features(df)
    sup = build_supervised(df, 1, base=base)
    tr, va, te = split_by_time(sup)
    scaler = fit_scaler(tr)                                            # TRAIN statistics only
    N = DL["seq_len"]
    if quick:
        tr, va, te = tr.iloc[::20], va.iloc[::20], te.iloc[::20]
    arrays = {k: make_arrays(base, d, scaler, N) for k, d in zip(("tr", "va", "te"), (tr, va, te))}
    sd = tr["y_volume"].std(); mu = tr["y_volume"].mean()
    yv = {k: ((d["y_volume"].values - mu) / sd).astype(F32) for k, d in zip(("tr", "va", "te"), (tr, va, te))}
    yc = {k: d["y_cong"].values for k, d in zip(("tr", "va", "te"), (tr, va, te))}
    log.info("sequences: train %s  val %s  test %s", arrays["tr"][0].shape, arrays["va"][0].shape, arrays["te"][0].shape)

    rows, hists, preds = [], {}, {}
    for name, bi in (("LSTM", False), ("Bi-LSTM", True)):
        model, hist = train_model(name, *arrays["tr"], yv["tr"], yc["tr"], *arrays["va"], yv["va"], yc["va"], bidirectional=bi)
        hists[name] = hist
        for split, d in (("va", va), ("te", te)):
            v, prob = model.predict(*arrays[split])
            pv = v * sd + mu
            rows.append({"task": "volume", "model": name, "split": split, **ev.regression_metrics(d["y_volume"], pv)})
            rows.append({"task": "congestion", "model": name, "split": split, **ev.congestion_metrics(d["y_cong"], prob.argmax(1))})
            if split == "te":
                preds[name] = (pv, prob)
        model.save(f"{name.lower().replace('-', '')}_h1.npz")
        log.info("[%s] test volume RMSE %.2f | congestion macro-F1 %.3f", name,
                 [r for r in rows if r["model"] == name and r["split"] == "te" and r["task"] == "volume"][0]["RMSE"],
                 [r for r in rows if r["model"] == name and r["split"] == "te" and r["task"] == "congestion"][0]["MacroF1"])
    json.dump({"mu": float(mu), "sd": float(sd)}, open(path(CFG["paths"]["models"], "dl_target_scale.json"), "w"))
    plot_curves(hists)
    out = te[["road_id", "target_ts"]].copy()
    for name, (pv, prob) in preds.items():
        out[f"{name}_volume_pred"] = pv; out[f"{name}_congestion_pred"] = prob.argmax(1)
    out["volume_true"], out["congestion_true"] = te["y_volume"].values, te["y_cong"].values
    if not quick:
        out.to_parquet(path(CFG["paths"]["reports"], "predictions_test_dl.parquet"), index=False)
        pd.DataFrame(rows).to_csv(path(CFG["paths"]["reports"], "scoreboard_dl.csv"), index=False)
        json.dump(hists, open(path(CFG["paths"]["reports"], "dl_history.json"), "w"))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run()
