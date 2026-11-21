"""Simple models every later model has to beat: base rates and Elo.

Constants chosen here (Elo K factor, home advantage, regression between seasons,
promoted-team start rating) are fixed defaults, not tuned on the test seasons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class BaseRate:
    """Predict the same probabilities for every match: the historical rates.

    window_seasons=None uses every season in the training data; a number uses
    only that many most recent seasons, which tracks drift in draw and goal rates.
    """

    def __init__(self, window_seasons: int | None = None, name: str | None = None):
        self.window = window_seasons
        self.name = name or ("base_rate_all" if window_seasons is None else f"base_rate_last{window_seasons}")
        self.p1x2 = None
        self.goals = None

    def fit(self, train: pd.DataFrame):
        if self.window is not None:
            keep = sorted(train["season_start"].unique())[-self.window:]
            train = train[train["season_start"].isin(keep)]
        self.p1x2 = np.array([(train["result_code"] == k).mean() for k in (0, 1, 2)])
        self.goals = {
            "over_1_5": float(train["over_1_5"].astype(float).mean()),
            "over_2_5": float(train["over_2_5"].astype(float).mean()),
            "btts": float(train["btts"].astype(float).mean()),
            "exp_home": float(train["home_goals"].mean()),
            "exp_away": float(train["away_goals"].mean()),
        }
        # scoreline frequencies (0 to 10 goals each side), lightly smoothed so no scoreline has zero probability
        G = 11
        hg = np.minimum(train["home_goals"].to_numpy(dtype=int), G - 1)
        ag = np.minimum(train["away_goals"].to_numpy(dtype=int), G - 1)
        counts = np.full((G, G), 0.05)
        np.add.at(counts, (hg, ag), 1.0)
        self.score_matrix = counts / counts.sum()
        return self

    def predict_1x2(self, test: pd.DataFrame) -> np.ndarray:
        return np.tile(self.p1x2, (len(test), 1))

    def predict_goals(self, test: pd.DataFrame) -> dict:
        out = {k: np.full(len(test), v) for k, v in self.goals.items()}
        out["score_probs"] = np.tile(self.score_matrix, (len(test), 1, 1))
        return out


def add_elo(
    df: pd.DataFrame,
    k: float = 20.0,
    home_adv: float = 65.0,
    season_regress: float = 0.25,
    init: float = 1500.0,
    promoted_init: float = 1450.0,
) -> pd.DataFrame:
    """Add pre-match Elo ratings (elo_home, elo_away, elo_diff) to a match table.

    Ratings are recorded before the match is played and updated afterwards, so
    each row only uses results from earlier matches. The table must be sorted by
    date. At each new season, ratings move `season_regress` of the way back to
    `init`; a team seen for the first time after the first season starts at
    `promoted_init`.
    """
    if not df["date"].is_monotonic_increasing:
        raise ValueError("matches must be sorted by date")
    ratings: dict[str, float] = {}
    first_season = df["season_start"].min()
    current = None
    elo_h, elo_a = [], []
    for season, home, away, hg, ag in zip(
        df["season_start"], df["home_team"], df["away_team"], df["home_goals"], df["away_goals"]
    ):
        if season != current:
            if current is not None:
                for t in ratings:
                    ratings[t] += season_regress * (init - ratings[t])
            current = season
        start = init if season == first_season else promoted_init
        rh = ratings.setdefault(home, start)
        ra = ratings.setdefault(away, start)
        elo_h.append(rh)
        elo_a.append(ra)

        expected = 1.0 / (1.0 + 10.0 ** (-(rh - ra + home_adv) / 400.0))
        score = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        gd = abs(hg - ag)
        mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11.0 + gd) / 8.0)
        delta = k * mult * (score - expected)
        ratings[home] = rh + delta
        ratings[away] = ra - delta

    out = df.copy()
    out["elo_home"] = elo_h
    out["elo_away"] = elo_a
    out["elo_diff"] = out["elo_home"] - out["elo_away"]
    return out


class EloLogit:
    """1X2 probabilities from the pre-match Elo difference through a multinomial logistic model.

    Needs the elo_diff column from add_elo(). Goals markets are not modelled.
    """

    name = "elo_logit"

    def __init__(self, c: float = 1.0):
        self.c = c
        self.lr = None

    def fit(self, train: pd.DataFrame):
        x = train[["elo_diff"]].to_numpy(dtype=float) / 100.0
        self.lr = LogisticRegression(C=self.c, max_iter=1000).fit(x, train["result_code"].astype(int))
        return self

    def predict_1x2(self, test: pd.DataFrame) -> np.ndarray:
        return self.lr.predict_proba(test[["elo_diff"]].to_numpy(dtype=float) / 100.0)

    def predict_goals(self, test: pd.DataFrame) -> dict:
        return {}
