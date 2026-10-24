"""Model governance (PRD 13.4): one JSON card per deployed model + a combined markdown index."""
from __future__ import annotations

import json
from datetime import date

import pandas as pd

from .features import FEATURES
from .utils import CFG, path

TASKS = ["volume", "travel_time", "congestion", "accident"]


def write_cards() -> None:
    sc = pd.read_csv(path(CFG["paths"]["reports"], "scoreboard_classical.csv"))
    try:
        dl = pd.read_csv(path(CFG["paths"]["reports"], "scoreboard_dl.csv"))
    except FileNotFoundError:
        dl = pd.DataFrame()
    version = path(CFG["paths"]["processed"], "VERSION.txt").read_text().split()[1]
    champs = json.load(open(path(CFG["paths"]["models"], "champions.json")))["champions"]
    cards = []
    common = {"dataset_version": version, "seed": CFG["seed"], "created": str(date.today()), "n_features": len(FEATURES), "features": FEATURES,
              "training_window": f"2025-01-01 .. {CFG['split']['train_end']}", "validation_window": f"2025-04-01 .. {CFG['split']['val_end']}",
              "test_window": f"2025-05-01 .. 2025-05-31", "scaler": "fit on training window only"}
    for task in TASKS:
        for model in sc[sc.task == task].model.unique():
            r = sc[(sc.task == task) & (sc.model == model)]
            te, va = r[r.split == "te"].iloc[0].dropna().to_dict(), r[r.split == "va"].iloc[0].dropna().to_dict()
            keep = lambda d: {k: v for k, v in d.items() if k not in ("task", "model", "split", "best_params")}   # noqa: E731
            cards.append({**common, "task": task, "model": model, "selected_champion_h1": champs.get(task) == model,
                          "hyperparameters": json.loads(te.get("best_params", "{}")), "validation_metrics": keep(va), "test_metrics": keep(te)})
    for _, grp in dl.groupby(["task", "model"]):
        te = grp[grp.split == "te"].iloc[0].dropna().to_dict(); va = grp[grp.split == "va"].iloc[0].dropna().to_dict()
        keep = lambda d: {k: v for k, v in d.items() if k not in ("task", "model", "split")}   # noqa: E731
        cards.append({**common, "task": te["task"], "model": te["model"], "selected_champion_h1": False,
                      "hyperparameters": {k: v for k, v in CFG["dl"].items()}, "validation_metrics": keep(va), "test_metrics": keep(te)})
    for c in cards:
        fn = f"{c['task']}__{c['model'].lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus').replace('/', '_')}.json"
        json.dump(c, open(path(CFG["paths"]["models"], "model_cards", fn), "w"), indent=2)
    lines = ["# Model cards index", "", f"Dataset version `{version}`, seed {CFG['seed']}. Full cards: `models/model_cards/*.json`.", "",
             "| Task | Model | Champion (h=1) | Key test metric |", "|---|---|---|---|"]
    for c in cards:
        k = {"volume": "RMSE", "travel_time": "RMSE", "congestion": "MacroF1", "accident": "ROC_AUC"}[c["task"]]
        lines.append(f"| {c['task']} | {c['model']} | {'yes' if c['selected_champion_h1'] else ''} | {k} = {c['test_metrics'].get(k, float('nan')):.4f} |")
    path(CFG["paths"]["models"], "model_cards", "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
