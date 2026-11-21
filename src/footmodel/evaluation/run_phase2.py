"""Score the Dixon–Coles model against the baselines and the market.

    python -m footmodel.evaluation.run_phase2 --raw data/raw --out data/results

Reads the settings written by tune_dc (data/results/dc_params.json). The tuning
only saw seasons up to 2018/19, so seasons from 2020/21 on are the final test.
"""

from __future__ import annotations

# Lets this file also be run directly (for example with VS Code's "Run Python File").
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    __package__ = "footmodel.evaluation"

import argparse
import json
from pathlib import Path

import pandas as pd

from ..data.loader import load_all
from ..models import BaseRate, EloLogit, MarketModel, add_elo
from ..models.dixon_coles import DixonColes
from .report import paired_logloss_diff, table_1x2, table_binary, table_by_season, table_goals
from .walkforward import walk_forward

FINAL_FIRST_SEASON = 2020


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/results")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    params_file = out / "dc_params.json"
    params = json.loads(params_file.read_text()) if params_file.exists() else {"xi": 0.001, "ridge": 8.0}

    df = add_elo(load_all(args.raw))
    models = [
        BaseRate(3),
        EloLogit(),
        DixonColes(xi=params["xi"], ridge=params["ridge"], name="dixon_coles"),
        MarketModel("market_avg_open", "avg_open", ou=("avg_ou_open_over", "avg_ou_open_under")),
        MarketModel("market_ps_close", "ps_close", ou=("ps_ou_close_over", "ps_ou_close_under")),
    ]
    preds = walk_forward(df, models, first_test_start=2014)
    preds.to_csv(out / "phase2_forecasts.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    final = preds[preds["season_start"] >= FINAL_FIRST_SEASON]
    early = preds[preds["season_start"] < FINAL_FIRST_SEASON]
    print(f"Dixon–Coles settings: xi={params['xi']}, ridge={params['ridge']}, refit every 14 days.")
    print(f"Selection seasons 2014/15-2018/19: {early['match_id'].nunique()} matches (used for tuning, so not a clean test).")
    print(f"FINAL seasons 2020/21-2025/26: {final['match_id'].nunique()} matches (never used for tuning).\n")

    def show(title, tbl, name):
        print(title)
        print(tbl.to_string(), "\n")
        tbl.to_csv(out / f"{name}.csv")

    show("1X2, final seasons, matches with Pinnacle closing odds",
         table_1x2(final, ["base_rate_last3", "elo_logit", "dixon_coles", "market_avg_open", "market_ps_close"]),
         "phase2_1x2_final_pinnacle")
    show("1X2, final seasons, all matches (market average opening odds)",
         table_1x2(final, ["base_rate_last3", "elo_logit", "dixon_coles", "market_avg_open"]),
         "phase2_1x2_final_all")

    bys = table_by_season(preds, ["elo_logit", "dixon_coles", "market_avg_open", "market_ps_close"])
    show("1X2 log loss by season (each model on its own available matches)", bys, "phase2_by_season")

    print("Paired 1X2 log-loss difference, final seasons (negative = first model better), 95% bootstrap interval")
    for a, b in [("dixon_coles", "elo_logit"), ("dixon_coles", "market_avg_open"),
                 ("dixon_coles", "market_ps_close"), ("elo_logit", "market_avg_open")]:
        d = paired_logloss_diff(final, a, b)
        verdict = "cannot separate" if d["lo"] <= 0 <= d["hi"] else ("first better" if d["mean"] < 0 else "second better")
        print(f"  {a} minus {b}: {d['mean']:+.4f}  [{d['lo']:+.4f}, {d['hi']:+.4f}]  n={d['n']}  -> {verdict}")
    print()

    show("Over 2.5 goals, final seasons",
         table_binary(final, ["base_rate_last3", "dixon_coles", "market_avg_open"], "p_o25", "over_2_5"),
         "phase2_over25_final")
    show("Over 1.5 goals, final seasons (no bookmaker odds in the data)",
         table_binary(final, ["base_rate_last3", "dixon_coles"], "p_o15", "over_1_5"), "phase2_over15_final")
    show("BTTS, final seasons (no bookmaker odds in the data)",
         table_binary(final, ["base_rate_last3", "dixon_coles"], "p_btts", "btts"), "phase2_btts_final")
    show("Expected goals and correct score, final seasons",
         table_goals(final, ["base_rate_last3", "dixon_coles"]), "phase2_goals_final")


if __name__ == "__main__":
    main()
    from .dashboard import build_from_argv
    build_from_argv()
