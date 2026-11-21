"""Pre-match form features built only from earlier matches.

For every match, each team gets exponentially weighted averages (over its own
previous matches, home and away together) of shots for and against, shots on
target for and against, and goals for and against, plus days of rest and a flag
for a team that was not in the league the season before. The values are recorded
before the match is played and updated afterwards, so a row never sees its own
result or any later one. A team with no history starts at the league average
seen so far.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STATS = ["sf", "sa", "tf", "ta", "gf", "ga"]  # shots, shots on target, goals: for and against
# Generic starting values for the very first match only (typical Premier League per-team averages).
START = {"s": 12.0, "t": 4.0, "g": 1.4}

FORM_COLUMNS = (
    [f"h_{s}" for s in STATS] + [f"a_{s}" for s in STATS] + ["h_rest", "a_rest", "new_home", "new_away"]
)


def add_form_features(df: pd.DataFrame, halflife: float = 8.0, rest_cap: float = 14.0) -> pd.DataFrame:
    """Add the form columns to a match table sorted by date.

    halflife is in matches: a result from `halflife` matches ago counts half as much as the latest.
    """
    if not df["date"].is_monotonic_increasing:
        raise ValueError("matches must be sorted by date")
    alpha = 1.0 - 0.5 ** (1.0 / halflife)

    seasons = sorted(df["season_start"].unique())
    teams_by_season = {s: set(df.loc[df["season_start"] == s, "home_team"]) | set(df.loc[df["season_start"] == s, "away_team"])
                       for s in seasons}
    previous = {s: (teams_by_season[seasons[i - 1]] if i > 0 else set()) for i, s in enumerate(seasons)}
    first_season = seasons[0]

    lg_sum = {"s": 0.0, "t": 0.0, "g": 0.0}
    lg_n = 0
    state: dict[str, dict] = {}
    out = {c: np.empty(len(df)) for c in FORM_COLUMNS}

    def league_average() -> dict:
        if lg_n == 0:
            return dict(START)
        return {k: lg_sum[k] / lg_n for k in lg_sum}

    def fresh(avg: dict) -> dict:
        return {"sf": avg["s"], "sa": avg["s"], "tf": avg["t"], "ta": avg["t"], "gf": avg["g"], "ga": avg["g"], "last": None}

    cols = ["date", "season_start", "home_team", "away_team", "home_goals", "away_goals",
            "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target"]
    for i, (date, season, home, away, hg, ag, hs, as_, hst, ast) in enumerate(zip(*(df[c] for c in cols))):
        avg = league_average()
        sh = state.setdefault(home, fresh(avg))
        sa_ = state.setdefault(away, fresh(avg))

        for prefix, st, team in (("h", sh, home), ("a", sa_, away)):
            for s in STATS:
                out[f"{prefix}_{s}"][i] = st[s]
            out[f"{prefix}_rest"][i] = rest_cap if st["last"] is None else min((date - st["last"]).days, rest_cap)
            is_new = (season != first_season) and (team not in previous[season])
            out["new_home" if prefix == "h" else "new_away"][i] = float(is_new)

        # update after the match
        for st, sf_, sa_s, tf_, ta_, gf_, ga_ in (
            (sh, hs, as_, hst, ast, hg, ag),
            (sa_, as_, hs, ast, hst, ag, hg),
        ):
            for key, value in (("sf", sf_), ("sa", sa_s), ("tf", tf_), ("ta", ta_), ("gf", gf_), ("ga", ga_)):
                if np.isfinite(value):
                    st[key] += alpha * (value - st[key])
            st["last"] = date
        if np.isfinite(hs) and np.isfinite(as_) and np.isfinite(hst) and np.isfinite(ast):
            lg_sum["s"] += (hs + as_) / 2.0
            lg_sum["t"] += (hst + ast) / 2.0
            lg_sum["g"] += (hg + ag) / 2.0
            lg_n += 1

    res = df.copy()
    for c in FORM_COLUMNS:
        res[c] = out[c]
    return res
