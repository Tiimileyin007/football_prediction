"""SQLite log: fixtures, timestamped odds quotes, forecasts, bets and results.

Every quote is kept with its timestamp. "Opening" is the earliest quote and
"closing" the last quote taken before kick-off, so closing-line value comes from
real snapshots. Times are naive UTC ISO strings.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS fixtures(match_key TEXT PRIMARY KEY, kickoff TEXT, home TEXT, away TEXT);
CREATE TABLE IF NOT EXISTS quotes(id INTEGER PRIMARY KEY, match_key TEXT, book TEXT, market TEXT,
    selection TEXT, odds REAL, taken_at TEXT);
CREATE INDEX IF NOT EXISTS q_idx ON quotes(match_key, market, book, taken_at);
CREATE TABLE IF NOT EXISTS forecasts(match_key TEXT, made_at TEXT, model TEXT, market TEXT,
    selection TEXT, prob REAL);
CREATE TABLE IF NOT EXISTS bets(id INTEGER PRIMARY KEY, match_key TEXT, book TEXT, market TEXT,
    selection TEXT, odds REAL, stake REAL, placed_at TEXT, p_model REAL, p_sharp REAL, p_final REAL,
    edge REAL, status TEXT DEFAULT 'open', profit REAL);
CREATE TABLE IF NOT EXISTS results(match_key TEXT PRIMARY KEY, home_goals INTEGER, away_goals INTEGER);
"""

SELECTIONS = {"1x2": ("H", "D", "A"), "ou25": ("O", "U")}


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


def connect(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def match_key(kickoff: str, home: str, away: str) -> str:
    return f"{kickoff[:10]}|{home}|{away}"


def upsert_fixture(con, kickoff: str, home: str, away: str) -> str:
    key = match_key(kickoff, home, away)
    con.execute("INSERT OR REPLACE INTO fixtures VALUES (?,?,?,?)", (key, kickoff, home, away))
    return key


def add_quote(con, key, book, market, selection, odds, taken_at=None):
    con.execute("INSERT INTO quotes(match_key,book,market,selection,odds,taken_at) VALUES (?,?,?,?,?,?)",
                (key, book, market, selection, float(odds), taken_at or now_utc()))


def quote_set(con, key, book, market, which: str = "latest", before: str | None = None):
    """One complete set of prices {selection: odds} from a single snapshot time.

    which is "latest" or "open". before restricts to quotes taken at or before that time.
    Returns (prices, taken_at) or (None, None) when no complete set exists.
    """
    sels = SELECTIONS[market]
    q = "SELECT taken_at FROM quotes WHERE match_key=? AND book=? AND market=?"
    args = [key, book, market]
    if before:
        q += " AND taken_at<=?"
        args.append(before)
    times = [r[0] for r in con.execute(q + " GROUP BY taken_at ORDER BY taken_at " + ("ASC" if which == "open" else "DESC"), args)]
    for t in times:
        rows = con.execute("SELECT selection, odds FROM quotes WHERE match_key=? AND book=? AND market=? AND taken_at=?",
                           (key, book, market, t)).fetchall()
        prices = {r["selection"]: r["odds"] for r in rows}
        if all(s in prices for s in sels):
            return {s: prices[s] for s in sels}, t
    return None, None
