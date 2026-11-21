"""Ensemble, calibration and market pooling, scored walk-forward on 2020/21 onward.

    python -m footmodel.evaluation.run_ensemble --forecasts data/results/poisson_forecasts.csv --out data/results

Reads the out-of-sample forecasts written by run_poisson. For each test season the
ensembles are fitted on the forecasts of earlier seasons only.
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

import numpy as np
import pandas as pd

from ..ensemble.stack import fit_pool_weight, geometric_mean
from ..ensemble.walkforward import GOAL_MODELS, MARKET, ONE_X_TWO_MODELS, run_ensemble, to_long
from . import metrics as M
from .reliability_plot import draw
from .report import paired_binary_logloss_diff, paired_logloss_diff, table_1x2, table_binary

FIRST_TEST = 2020


def build_inputs(f: pd.DataFrame):
    """Align every model's forecasts to one row per match, in date order."""
    ref = f[f["model"] == "base_rate_last3"].sort_values(["date", "match_id"]).reset_index(drop=True)
    meta = ref[["match_id", "season", "season_start", "date", "result_code", "home_goals", "away_goals",
                "total_goals", "over_1_5", "over_2_5", "btts"]].copy()
    A = {}
    for m in ONE_X_TWO_MODELS + [MARKET]:
        g = f[f["model"] == m].set_index("match_id").reindex(meta["match_id"])
        A[m] = {"p1x2": g[["p_h", "p_d", "p_a"]].to_numpy(dtype=float)}
        A[m].update({"o15": g["p_o15"].to_numpy(dtype=float), "o25": g["p_o25"].to_numpy(dtype=float),
                     "btts": g["p_btts"].to_numpy(dtype=float), "exp_h": g["exp_h"].to_numpy(dtype=float),
                     "exp_a": g["exp_a"].to_numpy(dtype=float)})
    return meta, A


def bootstrap_weight(market, model, y, kind, n_boot=300, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    ws = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        ws.append(fit_pool_weight(market[i], model[i], y[i], kind))
    return float(np.percentile(ws, 2.5)), float(np.percentile(ws, 97.5))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forecasts", default="data/results/poisson_forecasts.csv")
    ap.add_argument("--out", default="data/results")
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    f = pd.read_csv(args.forecasts, parse_dates=["date"])
    meta, A = build_inputs(f)
    out, weights = run_ensemble(meta, A, first_test_start=FIRST_TEST)
    long = to_long(meta, out)
    allp = pd.concat([f, long], ignore_index=True)
    final = allp[allp["season_start"] >= FIRST_TEST]
    long.to_csv(out_dir / "ensemble_forecasts.csv", index=False)

    pd.set_option("display.width", 240)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    n_final = final[final["model"] == "ens_stack3"]["match_id"].nunique()
    print(f"Walk-forward ensembles on seasons 2020/21 onward: {n_final} matches. Lower is better except accuracy.\n")

    def show(title, tbl, name):
        print(title)
        print(tbl.to_string(), "\n")
        tbl.to_csv(out_dir / f"{name}.csv")

    m1 = ["base_rate_last3", "elo_logit", "dixon_coles", "poisson_features", "ens_avg3", "ens_avg3_cal", "ens_stack3",
          "ens_pool_avg3", "market_avg_open"]
    show("1X2", table_1x2(final, m1), "ensemble_1x2")

    print("Paired 1X2 log-loss difference (negative = first better), 95% bootstrap interval")
    for a, b in [("ens_avg3", "elo_logit"), ("ens_avg3", "dixon_coles"), ("ens_avg3", "market_avg_open"),
                 ("ens_stack3", "elo_logit"), ("ens_avg3_cal", "elo_logit"), ("ens_avg3_cal", "ens_stack3"),
                 ("ens_pool_avg3", "market_avg_open"), ("ens_avg3_cal", "market_avg_open"), ("ens_stack3", "market_avg_open")]:
        r = paired_logloss_diff(final, a, b)
        v = "cannot separate" if r["lo"] <= 0 <= r["hi"] else ("first better" if r["mean"] < 0 else "second better")
        print(f"  {a} minus {b}: {r['mean']:+.4f}  [{r['lo']:+.4f}, {r['hi']:+.4f}]  -> {v}")
    print()

    # how much weight does the market-pooling put on our models?
    te = (meta["season_start"] >= FIRST_TEST).to_numpy()
    y = meta["result_code"].to_numpy()[te].astype(int)
    mk = A[MARKET]["p1x2"][te]
    avg = geometric_mean([A[m]["p1x2"][te] for m in ONE_X_TWO_MODELS])
    w_all = fit_pool_weight(mk, avg, y, "1x2")
    lo, hi = bootstrap_weight(mk, avg, y, "1x2")
    print("Weight on our models when blending with the market (0 = trust the market fully, 1 = trust our models fully)")
    print("  1X2, walk-forward weight per season: " + ", ".join(f"{s}/{str(s + 1)[-2:]}: {w:.2f}" for s, w in weights["1x2"].items()))
    print(f"  1X2, all final seasons together: {w_all:.2f}  (95% interval {lo:.2f} to {hi:.2f})")
    o25y = meta["over_2_5"].to_numpy(dtype=float)[te]
    mko = A[MARKET]["o25"][te]
    gavg = np.mean([A[m]["o25"][te] for m in GOAL_MODELS], axis=0)
    w_o = fit_pool_weight(mko, gavg, o25y, "binary")
    lo_o, hi_o = bootstrap_weight(mko, gavg, o25y, "binary")
    print("  Over 2.5, walk-forward weight per season: " + ", ".join(f"{s}/{str(s + 1)[-2:]}: {w:.2f}" for s, w in weights["o25"].items()))
    print(f"  Over 2.5, all final seasons together: {w_o:.2f}  (95% interval {lo_o:.2f} to {hi_o:.2f})\n")

    g_models = ["base_rate_last3", "dixon_coles", "poisson_features", "ens_goals_avg", "ens_goals_cal"]
    show("Over 2.5 goals", table_binary(final, g_models + ["ens_o25_pool", "market_avg_open"], "p_o25", "over_2_5"),
         "ensemble_over25")
    for a, b in [("ens_goals_cal", "dixon_coles"), ("ens_goals_cal", "market_avg_open"), ("ens_o25_pool", "market_avg_open")]:
        r = paired_binary_logloss_diff(final, a, b, "p_o25", "over_2_5")
        v = "cannot separate" if r["lo"] <= 0 <= r["hi"] else ("first better" if r["mean"] < 0 else "second better")
        print(f"  Over 2.5: {a} minus {b}: {r['mean']:+.4f}  [{r['lo']:+.4f}, {r['hi']:+.4f}]  -> {v}")
    print()
    show("Over 1.5 goals", table_binary(final, g_models, "p_o15", "over_1_5"), "ensemble_over15")
    show("BTTS", table_binary(final, g_models, "p_btts", "btts"), "ensemble_btts")

    xg = []
    for name in ["dixon_coles", "poisson_features", "ens_goals_avg", "ens_goals_cal"]:
        g = final[final["model"] == name]
        xg.append({"model": name, "mean_exp_home": g["exp_h"].mean(), "actual_home": g["home_goals"].mean(),
                   "mean_exp_away": g["exp_a"].mean(), "actual_away": g["away_goals"].mean(),
                   "rmse_home": float(np.sqrt(((g["exp_h"] - g["home_goals"]) ** 2).mean())),
                   "rmse_away": float(np.sqrt(((g["exp_a"] - g["away_goals"]) ** 2).mean()))})
    show("Expected goals", pd.DataFrame(xg).set_index("model"), "ensemble_expected_goals")

    # robustness: does the regularisation strength matter?
    print("Robustness: 1X2 and Over 2.5 log loss for different stacker regularisation (C)")
    for C in (0.1, 1.0, 10.0):
        o, _ = run_ensemble(meta, A, first_test_start=FIRST_TEST, C=C)
        t = np.isfinite(o["stack3"][:, 0])
        yy = meta["result_code"].to_numpy()[t].astype(int)
        print(f"  C={C:<4} stack3 {M.logloss(o['stack3'][t], yy):.4f}  avg3_cal {M.logloss(o['avg3_cal'][t], yy):.4f}  "
              f"o25_cal {M.binary_logloss(o['o25_cal'][t], meta['over_2_5'].to_numpy(dtype=float)[t]):.4f}")
    print()

    # calibration with intervals
    print("Calibration error (ECE) with 95% bootstrap interval, final seasons")
    rows = []
    checks = [
        ("Draw", "p_d", lambda g: (g["result_code"] == 1).astype(float), 5, ["dixon_coles", "ens_stack3", "market_avg_open"]),
        ("Home win", "p_h", lambda g: (g["result_code"] == 0).astype(float), 5, ["dixon_coles", "ens_stack3", "market_avg_open"]),
        ("Away win", "p_a", lambda g: (g["result_code"] == 2).astype(float), 5, ["dixon_coles", "ens_stack3", "market_avg_open"]),
        ("Over 2.5", "p_o25", lambda g: g["over_2_5"], 10, ["dixon_coles", "ens_goals_cal", "market_avg_open"]),
        ("Over 1.5", "p_o15", lambda g: g["over_1_5"], 10, ["dixon_coles", "ens_goals_cal"]),
        ("BTTS", "p_btts", lambda g: g["btts"], 10, ["dixon_coles", "ens_goals_cal"]),
    ]
    for label, col, target, bins, models in checks:
        for m in models:
            g = final[final["model"] == m]
            r = M.ece_interval(g[col].to_numpy(dtype=float), target(g).to_numpy(dtype=float), bins=bins, n_boot=500)
            rows.append({"market": label, "model": m, **r})
    ece_tbl = pd.DataFrame(rows)
    print(ece_tbl.to_string(index=False), "\n")
    ece_tbl.to_csv(out_dir / "ensemble_calibration_ece.csv", index=False)

    # reliability diagram
    def series(label_e, name_e, label_s, name_s, col, target, label_m=None, name_m=None):
        def get(name):
            g = final[final["model"] == name]
            return g[col].to_numpy(dtype=float), target(g).to_numpy(dtype=float)
        s = {"ensemble": (label_e, *get(name_e)), "single": (label_s, *get(name_s))}
        if name_m:
            s["market"] = (label_m, *get(name_m))
        return s

    ens1, ens2 = "Ensemble (stacked)", "Ensemble (stacked)"
    panels = [
        {"title": "Home win", "series": series(ens1, "ens_stack3", "Dixon–Coles alone", "dixon_coles", "p_h", lambda g: (g["result_code"] == 0).astype(float), "Bookmaker market", "market_avg_open")},
        {"title": "Draw", "series": series(ens1, "ens_stack3", "Dixon–Coles alone", "dixon_coles", "p_d", lambda g: (g["result_code"] == 1).astype(float), "Bookmaker market", "market_avg_open")},
        {"title": "Away win", "series": series(ens1, "ens_stack3", "Dixon–Coles alone", "dixon_coles", "p_a", lambda g: (g["result_code"] == 2).astype(float), "Bookmaker market", "market_avg_open")},
        {"title": "Over 2.5 goals", "series": series(ens2, "ens_goals_cal", "Dixon–Coles alone", "dixon_coles", "p_o25", lambda g: g["over_2_5"], "Bookmaker market", "market_avg_open")},
        {"title": "Over 1.5 goals", "series": series(ens2, "ens_goals_cal", "Dixon–Coles alone", "dixon_coles", "p_o15", lambda g: g["over_1_5"])},
        {"title": "Both teams to score", "series": series(ens2, "ens_goals_cal", "Dixon–Coles alone", "dixon_coles", "p_btts", lambda g: g["btts"])},
    ]
    draw(panels, str(out_dir / "reliability.png"), str(out_dir / "reliability_points.csv"),
         title="Forecast probabilities against how often they came true, 2020/21 to 2025/26")
    print(f"Reliability diagram: {out_dir / 'reliability.png'}")


if __name__ == "__main__":
    main()
    from .dashboard import build_from_argv
    build_from_argv()
