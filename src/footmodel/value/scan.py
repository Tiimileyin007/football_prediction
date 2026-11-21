"""Find value: selections whose price beats our fair probability."""

from __future__ import annotations

import numpy as np

from . import store
from .edge import candidates, stake
from .forecast import latest_forecast, sharp_fair


def scan(con, book: str = "sportybet", w: float = 0.2, min_edge: float = 0.03, bankroll: float = 1000.0,
         kelly_mult: float = 0.25, cap: float = 0.02, now: str | None = None, max_odds: float = 12.0) -> list[dict]:
    now = now or store.now_utc()
    out = []
    for fx in con.execute("SELECT * FROM fixtures WHERE kickoff>? ORDER BY kickoff", (now,)).fetchall():
        for market, sels in store.SELECTIONS.items():
            prices, taken = store.quote_set(con, fx["match_key"], book, market, "latest")
            if prices is None:
                continue
            fair = latest_forecast(con, fx["match_key"], market, f"final_w{w}")
            source = "model+market"
            if fair is None:
                fair, _ = sharp_fair(con, fx["match_key"], market)
                source = "market only"
            if fair is None:
                continue
            sharp, _ = sharp_fair(con, fx["match_key"], market)
            model = latest_forecast(con, fx["match_key"], market, "dixon_coles") or {}
            for c in candidates(fair, prices, min_edge, max_odds):
                c.update(match_key=fx["match_key"], kickoff=fx["kickoff"], market=market, book=book, quoted_at=taken, source=source,
                         p_sharp=(sharp or {}).get(c["selection"]), p_model=model.get(c["selection"]),
                         stake=stake(c["p_fair"], c["odds"], bankroll, kelly_mult, cap))
                out.append(c)
    return sorted(out, key=lambda r: -r["edge"])


def record_bet(con, cand: dict, stake_amount: float | None = None, placed_at: str | None = None) -> int:
    cur = con.execute(
        "INSERT INTO bets(match_key,book,market,selection,odds,stake,placed_at,p_model,p_sharp,p_final,edge) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (cand["match_key"], cand["book"], cand["market"], cand["selection"], cand["odds"],
         stake_amount if stake_amount is not None else cand["stake"], placed_at or store.now_utc(),
         cand.get("p_model"), cand.get("p_sharp"), cand["p_fair"], cand["edge"]))
    con.commit()
    return cur.lastrowid
