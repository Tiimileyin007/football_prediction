"""Walk-forward evaluation: train on earlier seasons, forecast the next one, roll on.

A model is any object with a `name`, `fit(train)`, `predict_1x2(test)` returning
an (n, 3) array in the order home, draw, away, and `predict_goals(test)` returning
a dict with any of the keys over_1_5, over_2_5, btts. NaN means "no forecast".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGETS = ["result_code", "home_goals", "away_goals", "total_goals", "over_1_5", "over_2_5", "btts"]
GOAL_COLUMNS = {"over_1_5": "p_o15", "over_2_5": "p_o25", "btts": "p_btts",
                "exp_home": "exp_h", "exp_away": "exp_a"}


def walk_forward(df: pd.DataFrame, models: list, first_test_start: int = 2014,
                 last_test_start: int | None = None) -> pd.DataFrame:
    """Forecast every season from `first_test_start` on, training only on earlier seasons.

    `last_test_start` stops the run early, so tuning can be confined to early seasons.
    Returns one row per (match, model) with forecasts and the actual targets.
    """
    frames = []
    for season in sorted(df["season_start"].unique()):
        if season < first_test_start or (last_test_start is not None and season > last_test_start):
            continue
        train = df[df["season_start"] < season]
        test = df[df["season_start"] == season]
        if train.empty or test.empty:
            continue
        for model in models:
            model.fit(train)
            p = np.asarray(model.predict_1x2(test), dtype=float)
            goals = model.predict_goals(test) or {}
            f = pd.DataFrame({
                "match_id": test["match_id"].to_numpy(),
                "season": test["season"].to_numpy(),
                "season_start": test["season_start"].to_numpy(),
                "date": test["date"].to_numpy(),
                "model": model.name,
                "p_h": p[:, 0], "p_d": p[:, 1], "p_a": p[:, 2],
            })
            for key, col in GOAL_COLUMNS.items():
                f[col] = np.asarray(goals[key], dtype=float) if key in goals else np.nan
            for t in TARGETS:
                f[t] = test[t].astype(float).to_numpy()
            f["p_score"] = _actual_score_probability(goals.get("score_probs"), test)
            frames.append(f)
    return pd.concat(frames, ignore_index=True)


def _actual_score_probability(score_probs, test: pd.DataFrame) -> np.ndarray:
    """Probability the model gave to the scoreline that actually happened (NaN if no scoreline forecast)."""
    if score_probs is None:
        return np.full(len(test), np.nan)
    S = np.asarray(score_probs, dtype=float)
    hg = test["home_goals"].to_numpy(dtype=int)
    ag = test["away_goals"].to_numpy(dtype=int)
    G = S.shape[1]
    inside = (hg < G) & (ag < G)
    out = np.full(len(test), 1e-12)
    out[inside] = S[np.flatnonzero(inside), hg[inside], ag[inside]]
    return out


def common_matches(preds: pd.DataFrame, models: list[str], column: str = "p_h") -> pd.DataFrame:
    """Keep only matches where every listed model has a forecast, so scores are comparable."""
    sub = preds[preds["model"].isin(models)]
    wide = sub.pivot(index="match_id", columns="model", values=column)
    ok = wide.index[wide[models].notna().all(axis=1)]
    return sub[sub["match_id"].isin(ok)]
