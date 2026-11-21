"""Turn walk-forward forecasts into score tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as M
from .walkforward import common_matches

P_COLS = ["p_h", "p_d", "p_a"]


def _probs(g: pd.DataFrame) -> np.ndarray:
    return g[P_COLS].to_numpy(dtype=float)


def table_1x2(preds: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """1X2 scores per model on the matches where all listed models have a forecast."""
    sub = common_matches(preds, models, "p_h")
    rows = []
    for name in models:
        g = sub[sub["model"] == name]
        P, y = _probs(g), g["result_code"].to_numpy(dtype=int)
        draw = (y == 1).astype(float)
        rows.append({
            "model": name, "n": len(g),
            "log_loss": M.logloss(P, y), "brier": M.brier(P, y), "rps": M.rps(P, y),
            "accuracy": M.accuracy(P, y), "balanced_acc": M.balanced_accuracy(P, y),
            "mean_p_draw": P[:, 1].mean(), "actual_draw": draw.mean(), "draw_ece": M.ece(P[:, 1], draw),
        })
    return pd.DataFrame(rows).set_index("model")


def table_by_season(preds: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """1X2 log loss per season, each model on all of its own forecast rows."""
    rows = {}
    for name in models:
        g = preds[(preds["model"] == name) & preds["p_h"].notna()]
        rows[name] = g.groupby("season").apply(
            lambda s: M.logloss(_probs(s), s["result_code"].to_numpy(dtype=int)), include_groups=False)
    return pd.DataFrame(rows)


def table_binary(preds: pd.DataFrame, models: list[str], p_col: str, y_col: str) -> pd.DataFrame:
    """Scores for one yes/no market (for example over 2.5 goals) on the common matches."""
    sub = common_matches(preds, models, p_col)
    rows = []
    for name in models:
        g = sub[sub["model"] == name]
        p, o = g[p_col].to_numpy(dtype=float), g[y_col].to_numpy(dtype=float)
        rows.append({"model": name, "n": len(g), "log_loss": M.binary_logloss(p, o),
                     "brier": M.binary_brier(p, o), "ece": M.ece(p, o),
                     "mean_forecast": p.mean(), "actual_rate": o.mean()})
    return pd.DataFrame(rows).set_index("model")


def table_goals(preds: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Expected-goals error and correct-score log loss, on matches where every model has a scoreline forecast."""
    sub = common_matches(preds, models, "p_score")
    rows = []
    for name in models:
        g = sub[sub["model"] == name]
        rows.append({
            "model": name, "n": len(g),
            "mean_exp_home": g["exp_h"].mean(), "actual_home": g["home_goals"].mean(),
            "mean_exp_away": g["exp_a"].mean(), "actual_away": g["away_goals"].mean(),
            "rmse_home": float(np.sqrt(((g["exp_h"] - g["home_goals"]) ** 2).mean())),
            "rmse_away": float(np.sqrt(((g["exp_a"] - g["away_goals"]) ** 2).mean())),
            "correct_score_ll": float(-np.log(np.clip(g["p_score"], 1e-12, 1)).mean()),
        })
    return pd.DataFrame(rows).set_index("model")


def paired_binary_logloss_diff(preds: pd.DataFrame, a: str, b: str, p_col: str, y_col: str, **kw) -> dict:
    """Mean log-loss difference a minus b for one yes/no market, with a bootstrap interval."""
    sub = common_matches(preds, [a, b], p_col).sort_values(["match_id", "model"])
    ga, gb = sub[sub["model"] == a], sub[sub["model"] == b]
    la = M.binary_row_logloss(ga[p_col].to_numpy(dtype=float), ga[y_col].to_numpy(dtype=float))
    lb = M.binary_row_logloss(gb[p_col].to_numpy(dtype=float), gb[y_col].to_numpy(dtype=float))
    return M.paired_bootstrap(la - lb, **kw)


def paired_logloss_diff(preds: pd.DataFrame, a: str, b: str, **kw) -> dict:
    """Mean log-loss difference a minus b on common matches, with a bootstrap interval."""
    sub = common_matches(preds, [a, b], "p_h").sort_values(["match_id", "model"])
    la = M.row_logloss(_probs(sub[sub["model"] == a]), sub[sub["model"] == a]["result_code"].to_numpy(dtype=int))
    lb = M.row_logloss(_probs(sub[sub["model"] == b]), sub[sub["model"] == b]["result_code"].to_numpy(dtype=int))
    return M.paired_bootstrap(la - lb, **kw)
