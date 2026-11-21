"""Walk-forward ensembles: for each test season, fit on the forecasts of earlier seasons only.

Inputs are the out-of-sample forecasts of the base models (arrays aligned to the
rows of `meta`, NaN where a model made no forecast), not the raw features, so the
ensemble layer can never see a test-season result when it is fitted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .stack import (BinaryStacker, OutcomeStacker, fit_pool_weight, geometric_mean, logit_mean,
                    pool_1x2, pool_binary)

ONE_X_TWO_MODELS = ["elo_logit", "dixon_coles", "poisson_features"]
GOAL_MODELS = ["dixon_coles", "poisson_features"]
MARKET = "market_avg_open"


def _finite_rows(*arrays) -> np.ndarray:
    ok = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        a = np.asarray(a, dtype=float)
        ok &= np.isfinite(a if a.ndim == 1 else a.sum(axis=1))
    return ok


def run_ensemble(meta: pd.DataFrame, A: dict, first_test_start: int = 2020, C: float = 1.0):
    """Return (forecasts, weights).

    forecasts maps a name to an array over the rows of meta (NaN outside the test seasons):
      1X2, shape (n, 3): avg3, avg3_cal, stack3, pool_avg3
      yes/no markets, shape (n,): o25_avg, o25_cal, o25_pool, o15_avg, o15_cal, btts_avg, btts_cal
      goals, shape (n,): exp_h_avg, exp_a_avg, exp_h_cal, exp_a_cal
    weights maps "1x2" and "o25" to {season_start: pooling weight on the model}.
    """
    n = len(meta)
    seasons = meta["season_start"].to_numpy()
    y = meta["result_code"].to_numpy(dtype=float)
    o15, o25, btts = (meta[c].to_numpy(dtype=float) for c in ("over_1_5", "over_2_5", "btts"))
    hg, ag = meta["home_goals"].to_numpy(dtype=float), meta["away_goals"].to_numpy(dtype=float)

    out = {k: np.full((n, 3), np.nan) for k in ("avg3", "avg3_cal", "stack3", "pool_avg3")}
    for k in ("o25_avg", "o25_cal", "o25_pool", "o15_avg", "o15_cal", "btts_avg", "btts_cal",
              "exp_h_avg", "exp_a_avg", "exp_h_cal", "exp_a_cal"):
        out[k] = np.full(n, np.nan)
    weights = {"1x2": {}, "o25": {}}

    P = {m: np.asarray(A[m]["p1x2"], dtype=float) for m in ONE_X_TWO_MODELS + [MARKET]}
    G = {m: {k: np.asarray(A[m][k], dtype=float) for k in ("o15", "o25", "btts", "exp_h", "exp_a")}
         for m in GOAL_MODELS}
    mkt_o25 = np.asarray(A[MARKET]["o25"], dtype=float)

    for s in sorted(set(seasons[seasons >= first_test_start])):
        te = seasons == s
        tr_base = seasons < s

        # ----- 1X2 -----
        ok = tr_base & _finite_rows(y, *[P[m] for m in ONE_X_TWO_MODELS + [MARKET]])
        trm = list(ONE_X_TWO_MODELS)
        yt = y[ok].astype(int)
        avg_tr = geometric_mean([P[m][ok] for m in trm])
        avg_te = geometric_mean([P[m][te] for m in trm])
        out["avg3"][te] = avg_te
        out["avg3_cal"][te] = OutcomeStacker(C).fit([avg_tr], yt).predict([avg_te])
        out["stack3"][te] = OutcomeStacker(C).fit([P[m][ok] for m in trm], yt).predict([P[m][te] for m in trm])
        w = fit_pool_weight(P[MARKET][ok], avg_tr, yt, "1x2")
        weights["1x2"][int(s)] = w
        out["pool_avg3"][te] = pool_1x2(P[MARKET][te], avg_te, w)

        # ----- yes/no goals markets -----
        for key, target, tag in (("o25", o25, "o25"), ("o15", o15, "o15"), ("btts", btts, "btts")):
            okg = tr_base & _finite_rows(target, *[G[m][key] for m in GOAL_MODELS])
            avg_tr_g = logit_mean([G[m][key][okg] for m in GOAL_MODELS])
            avg_te_g = logit_mean([G[m][key][te] for m in GOAL_MODELS])
            out[f"{tag}_avg"][te] = avg_te_g
            out[f"{tag}_cal"][te] = BinaryStacker(C).fit([G[m][key][okg] for m in GOAL_MODELS], target[okg]).predict(
                [G[m][key][te] for m in GOAL_MODELS])
            if key == "o25":
                okp = okg & _finite_rows(mkt_o25)
                wp = fit_pool_weight(mkt_o25[okp], logit_mean([G[m][key][okp] for m in GOAL_MODELS]), target[okp], "binary")
                weights["o25"][int(s)] = wp
                out["o25_pool"][te] = pool_binary(mkt_o25[te], avg_te_g, wp)

        # ----- expected goals: average, then scale so the training mean matches the actual mean -----
        for side, actual in (("h", hg), ("a", ag)):
            key = f"exp_{side}"
            okx = tr_base & _finite_rows(actual, *[G[m][key] for m in GOAL_MODELS])
            avg_tr_x = np.mean([G[m][key][okx] for m in GOAL_MODELS], axis=0)
            avg_te_x = np.mean([G[m][key][te] for m in GOAL_MODELS], axis=0)
            scale = actual[okx].mean() / avg_tr_x.mean()
            out[f"exp_{side}_avg"][te] = avg_te_x
            out[f"exp_{side}_cal"][te] = avg_te_x * scale

    return out, weights


def to_long(meta: pd.DataFrame, out: dict) -> pd.DataFrame:
    """Rows in the same long layout as the walk-forward forecasts, so the report tables can score them."""
    te = np.isfinite(out["avg3"][:, 0])
    base = meta.loc[te, ["match_id", "season", "season_start", "date", "result_code", "home_goals", "away_goals",
                         "total_goals", "over_1_5", "over_2_5", "btts"]].reset_index(drop=True)
    frames = []

    def add(name, p1x2=None, o15=None, o25=None, btts=None, exp_h=None, exp_a=None):
        f = base.copy()
        f["model"] = name
        f["p_h"], f["p_d"], f["p_a"] = (p1x2[te].T if p1x2 is not None else (np.nan, np.nan, np.nan))
        f["p_o15"] = o15[te] if o15 is not None else np.nan
        f["p_o25"] = o25[te] if o25 is not None else np.nan
        f["p_btts"] = btts[te] if btts is not None else np.nan
        f["exp_h"] = exp_h[te] if exp_h is not None else np.nan
        f["exp_a"] = exp_a[te] if exp_a is not None else np.nan
        f["p_score"] = np.nan
        frames.append(f)

    for name in ("avg3", "avg3_cal", "stack3", "pool_avg3"):
        add("ens_" + name, p1x2=out[name])
    add("ens_goals_avg", o15=out["o15_avg"], o25=out["o25_avg"], btts=out["btts_avg"],
        exp_h=out["exp_h_avg"], exp_a=out["exp_a_avg"])
    add("ens_goals_cal", o15=out["o15_cal"], o25=out["o25_cal"], btts=out["btts_cal"],
        exp_h=out["exp_h_cal"], exp_a=out["exp_a_cal"])
    add("ens_o25_pool", o25=out["o25_pool"])
    return pd.concat(frames, ignore_index=True)
