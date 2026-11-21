"""Score the baselines with walk-forward evaluation.

    python -m footmodel.evaluation.run_baselines --raw data/raw --out data/results

Trains on earlier seasons only and forecasts each season from 2014/15 on. Writes
the forecasts and score tables to the output folder and prints a report.
"""

from __future__ import annotations

# Lets this file also be run directly (for example with VS Code's "Run Python File").
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    __package__ = "footmodel.evaluation"

import argparse
from pathlib import Path

import pandas as pd

from ..data.loader import load_all
from ..models import BaseRate, EloLogit, MarketModel, add_elo
from .report import paired_logloss_diff, table_1x2, table_binary, table_by_season
from .walkforward import walk_forward

FIRST_TEST_START = 2014


def build_models() -> list:
    return [
        BaseRate(None),
        BaseRate(3),
        EloLogit(),
        MarketModel("market_ps_close", "ps_close", ou=("ps_ou_close_over", "ps_ou_close_under")),
        MarketModel("market_ps_open", "ps_open", ou=("ps_ou_open_over", "ps_ou_open_under")),
        MarketModel("market_avg_open", "avg_open", ou=("avg_ou_open_over", "avg_ou_open_under")),
    ]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/results")
    args = ap.parse_args(argv)

    df = add_elo(load_all(args.raw))
    preds = walk_forward(df, build_models(), first_test_start=FIRST_TEST_START)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out / "baseline_forecasts.csv", index=False)

    test_seasons = sorted(preds["season"].unique())
    print(f"Walk-forward: {len(test_seasons)} test seasons ({test_seasons[0]} to {test_seasons[-1]}), "
          f"{preds['match_id'].nunique()} matches. Lower is better for every score except accuracy.\n")

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")

    # A: everything on the matches where Pinnacle closing odds exist
    a_models = ["base_rate_all", "base_rate_last3", "elo_logit", "market_avg_open", "market_ps_open", "market_ps_close"]
    t_a = table_1x2(preds, a_models)
    print("1X2, matches with Pinnacle closing odds (all models on identical matches)")
    print(t_a.to_string(), "\n")
    t_a.to_csv(out / "table_1x2_with_pinnacle_close.csv")

    # B: no Pinnacle close, so the full sample including 2025/26
    b_models = ["base_rate_all", "base_rate_last3", "elo_logit", "market_avg_open"]
    t_b = table_1x2(preds, b_models)
    print("1X2, all test matches (market average opening odds as the market)")
    print(t_b.to_string(), "\n")
    t_b.to_csv(out / "table_1x2_all_matches.csv")

    by_season = table_by_season(preds, ["base_rate_last3", "elo_logit", "market_avg_open", "market_ps_close"])
    print("1X2 log loss by season (each model on its own available matches)")
    print(by_season.to_string(), "\n")
    by_season.to_csv(out / "table_by_season.csv")

    print("Paired difference in 1X2 log loss (negative = first model better), 95% bootstrap interval")
    for a, b in [("elo_logit", "market_ps_close"), ("elo_logit", "market_avg_open"),
                 ("market_ps_close", "market_avg_open"), ("base_rate_last3", "elo_logit")]:
        d = paired_logloss_diff(preds, a, b)
        verdict = "cannot separate" if d["lo"] <= 0 <= d["hi"] else ("first better" if d["mean"] < 0 else "second better")
        print(f"  {a} minus {b}: {d['mean']:+.4f}  [{d['lo']:+.4f}, {d['hi']:+.4f}]  n={d['n']}  -> {verdict}")
    print()

    print("Over 2.5 goals, matches with market odds")
    t_o25 = table_binary(preds, ["base_rate_all", "base_rate_last3", "market_avg_open"], "p_o25", "over_2_5")
    print(t_o25.to_string(), "\n")
    t_o25.to_csv(out / "table_over25.csv")
    t_o25p = table_binary(preds, ["base_rate_all", "market_avg_open", "market_ps_close"], "p_o25", "over_2_5")
    print("Over 2.5 goals, matches with Pinnacle closing odds")
    print(t_o25p.to_string(), "\n")

    print("Goals markets with no bookmaker odds in the data: base rates only")
    for p_col, y_col, label in [("p_o15", "over_1_5", "Over 1.5"), ("p_btts", "btts", "BTTS")]:
        t = table_binary(preds, ["base_rate_all", "base_rate_last3"], p_col, y_col)
        print(label)
        print(t.to_string(), "\n")
        t.to_csv(out / f"table_{y_col}.csv")


if __name__ == "__main__":
    main()
    from .dashboard import build_from_argv
    build_from_argv()
