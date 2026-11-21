"""Tune and test the Poisson-with-features model.

    python -m footmodel.evaluation.run_poisson --raw data/raw --out data/results

Stage 1 picks the form half-life and the regularisation on seasons 2014/15 to
2018/19 only. Stage 2 scores the chosen model on 2020/21 onward, which the tuning
never saw, against Elo, Dixon–Coles and the market, with two ablations that show
what the shot features add.
"""

from __future__ import annotations

# Lets this file also be run directly (for example with VS Code's "Run Python File").
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    __package__ = "footmodel.evaluation"

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from ..data.loader import load_all
from ..features import add_form_features
from ..models import BaseRate, EloLogit, MarketModel, add_elo
from ..models.dixon_coles import DixonColes
from ..models.poisson_features import FEATURES, PoissonFeatures
from . import metrics as M
from .report import paired_logloss_diff, table_1x2, table_binary, table_by_season, table_goals
from .walkforward import walk_forward

HALFLIVES = [10.0, 20.0, 40.0]
ALPHAS = [0.1, 0.3, 1.0, 3.0, 10.0]
SELECTION_LAST_SEASON = 2018
FINAL_FIRST_SEASON = 2020
ELO_ONLY = ["elo_diff", "new_home", "new_away", "h_rest", "a_rest"]
NO_SHOTS = ELO_ONLY + ["h_gf", "h_ga", "a_gf", "a_ga"]


def tune(df_elo: pd.DataFrame, out: Path) -> dict:
    rows = []
    for hl in HALFLIVES:
        d = add_form_features(df_elo, halflife=hl)
        models = [PoissonFeatures(alpha=a, name=f"pf_hl{hl}_a{a}") for a in ALPHAS]
        preds = walk_forward(d, models, first_test_start=2014, last_test_start=SELECTION_LAST_SEASON)
        for m in models:
            g = preds[preds["model"] == m.name]
            ll_1x2 = M.logloss(g[["p_h", "p_d", "p_a"]].to_numpy(), g["result_code"].to_numpy(dtype=int))
            ll_o25 = M.binary_logloss(g["p_o25"], g["over_2_5"])
            ll_btts = M.binary_logloss(g["p_btts"], g["btts"])
            rows.append({"halflife": hl, "alpha": m.alpha, "ll_1x2": ll_1x2, "ll_over25": ll_o25,
                         "ll_btts": ll_btts, "selection_score": ll_1x2 + ll_o25, "n": len(g)})
    grid = pd.DataFrame(rows).sort_values("selection_score").reset_index(drop=True)
    print(f"Stage 1: selection seasons 2014/15-{SELECTION_LAST_SEASON}/{str(SELECTION_LAST_SEASON + 1)[-2:]}, "
          f"{int(grid['n'].iloc[0])} matches. Lower is better.\n")
    print(grid.round(4).to_string(index=False), "\n")
    grid.to_csv(out / "poisson_tuning_grid.csv", index=False)
    best = {"halflife": float(grid.loc[0, "halflife"]), "alpha": float(grid.loc[0, "alpha"])}
    (out / "poisson_params.json").write_text(json.dumps(best) + "\n")
    print(f"Chosen: half-life {best['halflife']} matches, alpha {best['alpha']}\n")
    return best


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/results")
    ap.add_argument("--skip-tuning", action="store_true", help="reuse poisson_params.json")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df_elo = add_elo(load_all(args.raw))
    pfile = out / "poisson_params.json"
    best = json.loads(pfile.read_text()) if (args.skip_tuning and pfile.exists()) else tune(df_elo, out)

    dc_file = out / "dc_params.json"
    dc = json.loads(dc_file.read_text()) if dc_file.exists() else {"xi": 0.001, "ridge": 8.0}

    d = add_form_features(df_elo, halflife=best["halflife"])
    models = [
        BaseRate(3),
        EloLogit(),
        DixonColes(xi=dc["xi"], ridge=dc["ridge"], name="dixon_coles"),
        PoissonFeatures(alpha=best["alpha"], features=ELO_ONLY, name="poisson_elo_only"),
        PoissonFeatures(alpha=best["alpha"], features=NO_SHOTS, name="poisson_no_shots"),
        PoissonFeatures(alpha=best["alpha"], features=FEATURES, name="poisson_features"),
        MarketModel("market_avg_open", "avg_open", ou=("avg_ou_open_over", "avg_ou_open_under")),
        MarketModel("market_ps_close", "ps_close", ou=("ps_ou_close_over", "ps_ou_close_under")),
    ]
    preds = walk_forward(d, models, first_test_start=2014)
    preds.to_csv(out / "poisson_forecasts.csv", index=False)
    final = preds[preds["season_start"] >= FINAL_FIRST_SEASON]

    pd.set_option("display.width", 240)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(f"Stage 2: FINAL seasons 2020/21-2025/26, {final['match_id'].nunique()} matches "
          f"(not used for tuning). Poisson settings: half-life {best['halflife']}, alpha {best['alpha']}.\n")

    def show(title, tbl, name):
        print(title)
        print(tbl.to_string(), "\n")
        tbl.to_csv(out / f"{name}.csv")

    names = ["elo_logit", "dixon_coles", "poisson_elo_only", "poisson_no_shots", "poisson_features"]
    show("1X2, final seasons, all matches", table_1x2(final, ["base_rate_last3"] + names + ["market_avg_open"]),
         "poisson_1x2_final_all")
    show("1X2, final seasons, matches with Pinnacle closing odds",
         table_1x2(final, ["elo_logit", "dixon_coles", "poisson_features", "market_avg_open", "market_ps_close"]),
         "poisson_1x2_final_pinnacle")
    show("1X2 log loss by season", table_by_season(preds, ["elo_logit", "dixon_coles", "poisson_features",
                                                           "market_avg_open", "market_ps_close"]), "poisson_by_season")

    print("Paired 1X2 log-loss difference, final seasons (negative = first model better), 95% bootstrap interval")
    for a, b in [("poisson_features", "elo_logit"), ("poisson_features", "dixon_coles"),
                 ("poisson_features", "poisson_no_shots"), ("poisson_features", "market_avg_open"),
                 ("poisson_features", "market_ps_close")]:
        r = paired_logloss_diff(final, a, b)
        verdict = "cannot separate" if r["lo"] <= 0 <= r["hi"] else ("first better" if r["mean"] < 0 else "second better")
        print(f"  {a} minus {b}: {r['mean']:+.4f}  [{r['lo']:+.4f}, {r['hi']:+.4f}]  n={r['n']}  -> {verdict}")
    print()

    goal_models = ["base_rate_last3", "dixon_coles", "poisson_features"]
    show("Over 2.5 goals, final seasons", table_binary(final, goal_models + ["market_avg_open"], "p_o25", "over_2_5"),
         "poisson_over25_final")
    show("Over 1.5 goals, final seasons", table_binary(final, goal_models, "p_o15", "over_1_5"), "poisson_over15_final")
    show("BTTS, final seasons", table_binary(final, goal_models, "p_btts", "btts"), "poisson_btts_final")
    show("Expected goals and correct score, final seasons", table_goals(final, goal_models), "poisson_goals_final")


if __name__ == "__main__":
    main()
    from .dashboard import build_from_argv
    build_from_argv()
