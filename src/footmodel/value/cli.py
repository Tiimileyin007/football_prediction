"""Command line for the value engine:  python -m footmodel.value.cli --db football.db <command>

  snapshot --source footballdata|oddsapi [--location PATH_OR_URL]   store one odds snapshot (schedule this)
  forecast --matches data/clean/matches.csv [--w 0.2]                forecasts for upcoming fixtures
  fixtures                                                           list upcoming fixtures and their keys
  quote --key KEY --market 1x2|ou25 --odds H D A | O U [--book sportybet]   type in a price set
  scan [--book sportybet] [--min-edge 0.03] [--bankroll 1000]        list value candidates
  bet --key KEY --market M --selection S --odds X [--stake Y]        log a bet you placed
  results --csv PATH_OR_URL | --key KEY --score 2-1                  load results (also fills in missing closing prices)
  settle                                                             settle open bets from results
  report                                                             ROI, CLV, margin, bets needed
"""

from __future__ import annotations

# Lets this file also be run directly (for example with VS Code's "Run Python File").
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    __package__ = "footmodel.value"

import argparse
import urllib.request

import pandas as pd

from . import collector, forecast, scan as scanmod, sources, store, tracker
from .forecast import latest_forecast, sharp_fair


def _read(loc: str) -> str:
    if loc.startswith("http"):
        with urllib.request.urlopen(loc, timeout=30) as r:
            return r.read().decode("utf-8-sig", errors="replace")
    return open(loc, encoding="utf-8-sig").read()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="football.db")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot"); s.add_argument("--source", default="footballdata"); s.add_argument("--location")
    f = sub.add_parser("forecast"); f.add_argument("--matches", default="data/clean/matches.csv"); f.add_argument("--w", type=float, default=0.2)
    sub.add_parser("fixtures")
    q = sub.add_parser("quote"); q.add_argument("--key", required=True); q.add_argument("--market", required=True); q.add_argument("--odds", nargs="+", type=float, required=True); q.add_argument("--book", default="sportybet")
    c = sub.add_parser("scan"); c.add_argument("--book", default="sportybet"); c.add_argument("--min-edge", type=float, default=0.03); c.add_argument("--bankroll", type=float, default=1000.0); c.add_argument("--w", type=float, default=0.2)
    b = sub.add_parser("bet"); b.add_argument("--key", required=True); b.add_argument("--market", required=True); b.add_argument("--selection", required=True); b.add_argument("--odds", type=float, required=True); b.add_argument("--stake", type=float, required=True); b.add_argument("--book", default="sportybet"); b.add_argument("--w", type=float, default=0.2)
    r = sub.add_parser("results"); r.add_argument("--csv"); r.add_argument("--key"); r.add_argument("--score")
    sub.add_parser("settle")
    sub.add_parser("report")
    a = ap.parse_args(argv)
    con = store.connect(a.db)

    if a.cmd == "snapshot":
        src = sources.TheOddsApi() if a.source == "oddsapi" else sources.FootballDataFixtures(*([a.location] if a.location else []))
        print(collector.snapshot(con, src.fetch()))
    elif a.cmd == "forecast":
        print(f"forecasts made for {forecast.run_forecasts(con, pd.read_csv(a.matches), a.w)} fixtures")
    elif a.cmd == "fixtures":
        for r_ in con.execute("SELECT * FROM fixtures WHERE kickoff>? ORDER BY kickoff", (store.now_utc(),)):
            print(r_["kickoff"], r_["match_key"])
    elif a.cmd == "quote":
        sels = store.SELECTIONS[a.market]
        if len(a.odds) != len(sels):
            raise SystemExit(f"{a.market} needs {len(sels)} prices ({'/'.join(sels)})")
        t = store.now_utc()
        for sel, o in zip(sels, a.odds):
            store.add_quote(con, a.key, a.book, a.market, sel, o, t)
        con.commit()
        print("stored", dict(zip(sels, a.odds)))
    elif a.cmd == "scan":
        rows = scanmod.scan(con, a.book, a.w, a.min_edge, a.bankroll)
        if not rows:
            print("No value candidates.")
        for x in rows:
            print(f"{x['kickoff']}  {x['match_key']}  {x['market']} {x['selection']} @ {x['odds']:.2f}  fair {x['p_fair']:.3f}  edge {x['edge']*100:+.1f}%  stake {x['stake']:.2f}  ({x['source']})")
    elif a.cmd == "bet":
        sels = store.SELECTIONS[a.market]
        fair = latest_forecast(con, a.key, a.market, f"final_w{a.w}")
        sharp, _ = sharp_fair(con, a.key, a.market)
        fair = fair or sharp
        if fair is None:
            raise SystemExit("no fair probability for this match yet: run snapshot (and forecast) first")
        model = latest_forecast(con, a.key, a.market, "dixon_coles") or {}
        cand = dict(match_key=a.key, book=a.book, market=a.market, selection=a.selection, odds=a.odds, p_fair=fair[a.selection],
                    edge=fair[a.selection] * a.odds - 1, p_sharp=(sharp or {}).get(a.selection), p_model=model.get(a.selection))
        print("bet id", scanmod.record_bet(con, cand, a.stake), f"(edge at bet time {cand['edge']*100:+.1f}%)")
    elif a.cmd == "results":
        if a.csv:
            text = _read(a.csv)
            print("results loaded:", tracker.results_from_csv(con, text))
            print("closing prices filled in from the file:", tracker.backfill_closes(con, text))
        else:
            hg, ag = map(int, a.score.split("-"))
            tracker.set_result(con, a.key, hg, ag)
    elif a.cmd == "settle":
        print("settled", tracker.settle(con), "bets")
    elif a.cmd == "report":
        print(tracker.report(con))


if __name__ == "__main__":
    main()
