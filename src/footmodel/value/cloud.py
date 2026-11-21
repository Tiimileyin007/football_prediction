"""Always-on snapshots: a light collector for a scheduler (GitHub Actions) that needs no database.

    python -m footmodel.value.cloud snapshot --log data/live/quotes_log.csv

Each run downloads the upcoming-fixtures file and appends the odds it finds to a plain CSV log.
Text appends keep the repository small. On your own computer, `cloud-import` (CLI) or the app's Data
tab loads that log into football.db, so the app sees every snapshot taken while your PC was off.

To keep the log small it records only the books that matter (Pinnacle, Bet365, market average and best
price), only fixtures kicking off in the next 72 hours, and snapshots often only when a match is close:
every run within 3 hours of a kick-off, otherwise at most once every 6 hours.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    __package__ = "footmodel.value"

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

from . import store
from .sources import FootballDataFixtures, Quote

FIELDS = ["taken_at", "kickoff", "home", "away", "book", "market", "selection", "odds"]
KEEP_BOOKS = {"pinnacle", "bet365", "market_avg", "market_max"}


def read_log(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def last_taken_at(path) -> str | None:
    rows = read_log(path)
    return rows[-1]["taken_at"] if rows else None


def should_snapshot(quotes: list[Quote], last: str | None, now: str, near_hours: float = 3.0, idle_hours: float = 6.0) -> bool:
    t = datetime.fromisoformat(now)
    if last is None or t - datetime.fromisoformat(last) >= timedelta(hours=idle_hours):
        return True
    return any(timedelta(0) <= datetime.fromisoformat(q.kickoff) - t <= timedelta(hours=near_hours) for q in quotes)


def snapshot_to_log(quotes: list[Quote], path, now: str | None = None, force: bool = False) -> dict:
    now = now or store.now_utc()
    t = datetime.fromisoformat(now)
    keep = [q for q in quotes if q.book in KEEP_BOOKS and timedelta(0) < datetime.fromisoformat(q.kickoff) - t <= timedelta(hours=72)]
    if not keep:
        return {"written": 0, "reason": "no upcoming fixtures in the next 72 hours"}
    if not force and not should_snapshot(keep, last_taken_at(path), now):
        return {"written": 0, "reason": "no match close to kick-off and the last snapshot is recent"}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists() or p.stat().st_size == 0
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(FIELDS)
        for q in keep:
            w.writerow([now, q.kickoff, q.home, q.away, q.book, q.market, q.selection, q.odds])
    return {"written": len(keep), "taken_at": now}


def import_log(con, path) -> dict:
    """Load a cloud log into the database. Safe to repeat: rows already stored are skipped."""
    added = 0
    for r in read_log(path):
        key = store.upsert_fixture(con, r["kickoff"], r["home"], r["away"])
        if con.execute("SELECT 1 FROM quotes WHERE match_key=? AND book=? AND market=? AND selection=? AND taken_at=?",
                       (key, r["book"], r["market"], r["selection"], r["taken_at"])).fetchone():
            continue
        store.add_quote(con, key, r["book"], r["market"], r["selection"], float(r["odds"]), r["taken_at"])
        added += 1
    con.commit()
    return {"quotes_added": added}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--log", default="data/live/quotes_log.csv")
    s.add_argument("--location", default=None, help="fixtures file URL or path (default: football-data.co.uk)")
    s.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    src = FootballDataFixtures(*([a.location] if a.location else []))
    print(snapshot_to_log(src.fetch(), a.log, force=a.force))


if __name__ == "__main__":
    main()
