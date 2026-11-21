"""Live forecasts for upcoming fixtures: Dixon–Coles fitted on all finished matches,
then blended with the sharp book's fair probabilities (weight w on the model)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.dixon_coles import COLS, fit_dc, markets, rates, score_matrices
from ..models.market import demargin
from . import store
from .edge import pool

SHARP = "pinnacle"


def sharp_fair(con, key, market, before=None, which="latest"):
    prices, t = store.quote_set(con, key, SHARP, market, which, before)
    if prices is None:
        return None, None
    sels = store.SELECTIONS[market]
    p = demargin(np.array([prices[s] for s in sels]), "power")
    return dict(zip(sels, p)), t


def run_forecasts(con, matches: pd.DataFrame, w: float = 0.2, xi: float = 0.001, ridge: float = 8.0, now: str | None = None) -> int:
    now = now or store.now_utc()
    fx = pd.DataFrame(con.execute("SELECT * FROM fixtures WHERE kickoff>?", (now,)).fetchall(), columns=["match_key", "kickoff", "home", "away"])
    if fx.empty:
        return 0
    hist = matches.copy()
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist[hist["date"] < pd.Timestamp(now)].dropna(subset=["home_goals", "away_goals"])[COLS]
    params = fit_dc(hist, pd.Timestamp(now), xi, ridge)
    known = set(hist["home_team"]) | set(hist["away_team"])
    n = 0
    for _, r in fx.iterrows():
        if r["home"] not in known or r["away"] not in known:
            continue                                  # new team: no model forecast, the market-only fair price still works
        lam, mu = rates(params, pd.Series([r["home"]]), pd.Series([r["away"]]))
        mk = markets(score_matrices(lam, mu, params.rho, 10))
        model = {"1x2": dict(H=mk["p_h"][0], D=mk["p_d"][0], A=mk["p_a"][0]), "ou25": dict(O=mk["over_2_5"][0], U=1 - mk["over_2_5"][0])}
        for market, pm in model.items():
            sels = store.SELECTIONS[market]
            for sel in sels:
                con.execute("INSERT INTO forecasts VALUES (?,?,?,?,?,?)", (r["match_key"], now, "dixon_coles", market, sel, float(pm[sel])))
            fair, _ = sharp_fair(con, r["match_key"], market)
            if fair is not None:
                final = pool([fair[s] for s in sels], [pm[s] for s in sels], w)
                for sel, p in zip(sels, final):
                    con.execute("INSERT INTO forecasts VALUES (?,?,?,?,?,?)", (r["match_key"], now, f"final_w{w}", market, sel, float(p)))
        n += 1
    con.commit()
    return n


def latest_forecast(con, key, market, model_prefix):
    row = con.execute("SELECT MAX(made_at) FROM forecasts WHERE match_key=? AND market=? AND model LIKE ?", (key, market, model_prefix + "%")).fetchone()
    if not row or row[0] is None:
        return None
    rows = con.execute("SELECT selection, prob FROM forecasts WHERE match_key=? AND market=? AND model LIKE ? AND made_at=?", (key, market, model_prefix + "%", row[0])).fetchall()
    return {r["selection"]: r["prob"] for r in rows}
