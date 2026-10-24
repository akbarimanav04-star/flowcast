"""One command reproduces everything from the raw CSVs:   python run_pipeline.py

Stages: clean -> eda -> ml (classical + horizons) -> dl (LSTM / Bi-LSTM) -> cards -> report
Use  --skip dl  (etc.) to leave a stage out, or --only ml to run a single stage.
"""
import argparse
import time

from src import clean, dl_model, eda, ml_models, model_cards, report
from src.utils import get_logger, set_seed

log = get_logger("pipeline")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="*", default=[], choices=["clean", "eda", "ml", "dl", "cards", "report"])
    ap.add_argument("--only", choices=["clean", "eda", "ml", "dl", "cards", "report"])
    a = ap.parse_args()
    todo = [a.only] if a.only else [s for s in ["clean", "eda", "ml", "dl", "cards", "report"] if s not in a.skip]
    set_seed()
    t0 = time.time()
    if "clean" in todo:
        clean.run()
    if "eda" in todo:
        eda.run()
    if "ml" in todo:
        res = ml_models.train_all()
        ml_models.train_horizons(res["best_params"], res["base"], res["class_weights"])
    if "dl" in todo:
        dl_model.run()
    if "cards" in todo:
        model_cards.write_cards()
    if "report" in todo:
        report.write_report()
    log.info("pipeline finished in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
