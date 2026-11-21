"""Does any model add information beyond the market?

    python -m footmodel.evaluation.residual_test --forecasts data/results/poisson_forecasts.csv

Walk-forward stack: for each season from 2020/21 on, fit a multinomial logistic
regression on the earlier seasons' out-of-sample forecasts and score it on the
season. Three stacks are compared:
  market_recal  market probabilities only (recalibrated), the bar to clear
  models_only   Elo, Dixon–Coles and Poisson-with-features log-odds, no market
  market_plus   market log-odds plus the three models' log-odds
If market_plus does not beat market_recal by more than noise, the models carry
no usable information the market does not already have.
"""

from __future__ import annotations

# Lets this file also be run directly (for example with VS Code's "Run Python File").
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    __package__ = "footmodel.evaluation"

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import metrics as M

MODELS = ["elo_logit", "dixon_coles", "poisson_features"]
MARKET = "market_avg_open"
FIRST_STACK_TEST = 2020
C = 1.0


def logodds(w: pd.DataFrame, model: str) -> np.ndarray:
    p = w[model][["p_h", "p_d", "p_a"]].to_numpy(dtype=float)
    p = np.clip(p, 1e-6, 1.0)
    return np.column_stack([np.log(p[:, 0] / p[:, 1]), np.log(p[:, 2] / p[:, 1])])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forecasts", default="data/results/poisson_forecasts.csv")
    args = ap.parse_args(argv)

    f = pd.read_csv(args.forecasts)
    keep = f[f["model"].isin(MODELS + [MARKET])]
    wide = keep.pivot(index="match_id", columns="model", values=["p_h", "p_d", "p_a"])
    wide = wide.swaplevel(axis=1).sort_index(axis=1)
    meta = f.drop_duplicates("match_id").set_index("match_id")[["season_start", "result_code", "date"]]
    ids = wide.index
    season = meta.loc[ids, "season_start"].to_numpy()
    y = meta.loc[ids, "result_code"].to_numpy(dtype=int)

    def feats(names: list[str]) -> np.ndarray:
        return np.hstack([logodds(wide, n) for n in names])

    sets = {"market_recal": feats([MARKET]), "models_only": feats(MODELS), "market_plus": feats([MARKET] + MODELS)}
    market_raw = wide[MARKET][["p_h", "p_d", "p_a"]].to_numpy(dtype=float)

    rows, per_match = [], {k: [] for k in list(sets) + ["market_raw"]}
    order = []
    for s in sorted(set(season[season >= FIRST_STACK_TEST])):
        tr, te = season < s, season == s
        for name, X in sets.items():
            lr = LogisticRegression(C=C, max_iter=2000).fit(X[tr], y[tr])
            P = lr.predict_proba(X[te])
            per_match[name].append(M.row_logloss(P, y[te]))
        per_match["market_raw"].append(M.row_logloss(market_raw[te], y[te]))
        order.append(s)

    losses = {k: np.concatenate(v) for k, v in per_match.items()}
    n = len(losses["market_raw"])
    print(f"Walk-forward stacks, seasons {order[0]}/{str(order[0] + 1)[-2:]} on, {n} matches. 1X2 log loss (lower is better):")
    for k in ["market_raw", "market_recal", "models_only", "market_plus"]:
        print(f"  {k:13s} {losses[k].mean():.4f}")
    print("\nPaired differences (negative = first is better), 95% bootstrap interval:")
    for a, b in [("market_plus", "market_recal"), ("market_plus", "market_raw"), ("models_only", "market_raw")]:
        r = M.paired_bootstrap(losses[a] - losses[b])
        verdict = "cannot separate" if r["lo"] <= 0 <= r["hi"] else ("first better" if r["mean"] < 0 else "second better")
        print(f"  {a} minus {b}: {r['mean']:+.4f}  [{r['lo']:+.4f}, {r['hi']:+.4f}]  -> {verdict}")


if __name__ == "__main__":
    main()
