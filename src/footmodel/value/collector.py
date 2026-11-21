"""Take one odds snapshot and store it. Schedule this often (see README): the closing
price is simply the last snapshot taken before kick-off, so a 30-minute gap means a
closing price up to 30 minutes stale. The tracker reports how stale each close is."""

from __future__ import annotations

from . import store
from .sources import Quote


def snapshot(con, quotes: list[Quote], taken_at: str | None = None) -> dict:
    t = taken_at or store.now_utc()
    keys = set()
    for q in quotes:
        key = store.upsert_fixture(con, q.kickoff, q.home, q.away)
        keys.add(key)
        store.add_quote(con, key, q.book, q.market, q.selection, q.odds, t)
    con.commit()
    return {"taken_at": t, "fixtures": len(keys), "quotes": len(quotes)}
