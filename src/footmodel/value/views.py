"""Tables for the app: plain pandas, no Streamlit, so they can be tested."""

from __future__ import annotations

import pandas as pd

from . import store
from .forecast import latest_forecast, sharp_fair


def fixtures_table(con, book: str = "sportybet", w: float = 0.2, now: str | None = None) -> pd.DataFrame:
    """One row per upcoming fixture, market and selection: our probability, the sharp market's, the book's price and the edge."""
    now = now or store.now_utc()
    rows = []
    for fx in con.execute("SELECT * FROM fixtures WHERE kickoff>? ORDER BY kickoff", (now,)).fetchall():
        for market, sels in store.SELECTIONS.items():
            model = latest_forecast(con, fx["match_key"], market, "dixon_coles") or {}
            final = latest_forecast(con, fx["match_key"], market, f"final_w{w}") or {}
            sharp, _ = sharp_fair(con, fx["match_key"], market)
            prices, taken = store.quote_set(con, fx["match_key"], book, market, "latest")
            for s in sels:
                fair = final.get(s) or (sharp or {}).get(s)
                odds = (prices or {}).get(s)
                rows.append({
                    "kickoff": fx["kickoff"], "match": f"{fx['home']} v {fx['away']}", "match_key": fx["match_key"],
                    "market": market, "selection": s,
                    "model_p": model.get(s), "market_fair_p": (sharp or {}).get(s), "our_p": fair,
                    "book_odds": odds, "odds_taken_at": taken,
                    "edge": (fair * odds - 1) if (fair and odds) else None})
    return pd.DataFrame(rows)


def bets_table(con) -> pd.DataFrame:
    return pd.DataFrame([dict(r) for r in con.execute(
        "SELECT b.id, f.kickoff, f.home || ' v ' || f.away AS match, b.market, b.selection, b.book, b.odds, b.stake, b.edge, "
        "b.p_final, b.status, b.profit FROM bets b JOIN fixtures f USING(match_key) ORDER BY f.kickoff DESC, b.id DESC")])


def counts(con) -> dict:
    q = lambda sql: con.execute(sql).fetchone()[0]
    return {"fixtures": q("SELECT COUNT(*) FROM fixtures WHERE kickoff>'" + store.now_utc() + "'"), "quotes": q("SELECT COUNT(*) FROM quotes"),
            "bets": q("SELECT COUNT(*) FROM bets"), "settled": q("SELECT COUNT(*) FROM bets WHERE status!='open'")}
