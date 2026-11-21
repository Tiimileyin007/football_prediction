"""Results, settlement and the monitoring report (ROI, CLV, margin, bets needed)."""

from __future__ import annotations

import numpy as np

from . import store
from .forecast import sharp_fair


def outcome(market: str, selection: str, hg: int, ag: int) -> bool:
    if market == "1x2":
        return selection == ("H" if hg > ag else "D" if hg == ag else "A")
    return selection == ("O" if hg + ag >= 3 else "U")


def settle(con) -> int:
    n = 0
    for b in con.execute("SELECT b.*, r.home_goals hg, r.away_goals ag FROM bets b JOIN results r USING(match_key) WHERE b.status='open'").fetchall():
        won = outcome(b["market"], b["selection"], b["hg"], b["ag"])
        con.execute("UPDATE bets SET status=?, profit=? WHERE id=?",
                    ("won" if won else "lost", b["stake"] * (b["odds"] - 1) if won else -b["stake"], b["id"]))
        n += 1
    con.commit()
    return n


def set_result(con, key: str, hg: int, ag: int):
    con.execute("INSERT OR REPLACE INTO results VALUES (?,?,?)", (key, hg, ag))
    con.commit()


def results_from_csv(con, text: str) -> int:
    """Finished matches from a football-data season/fixtures CSV (Date, HomeTeam, AwayTeam, FTHG, FTAG)."""
    import csv, io
    from datetime import datetime, timedelta
    from .sources import norm_team
    n = 0
    for r in csv.DictReader(io.StringIO(text.lstrip("﻿"))):
        if not r.get("FTHG") or not r.get("Date"):
            continue
        d = datetime.strptime(r["Date"], "%d/%m/%Y" if len(r["Date"]) == 10 else "%d/%m/%y")
        h, a = norm_team(r["HomeTeam"]), norm_team(r["AwayTeam"])
        for fx in con.execute("SELECT match_key,kickoff FROM fixtures WHERE home=? AND away=?", (h, a)).fetchall():
            if abs((datetime.fromisoformat(fx["kickoff"]) - d).days) <= 1:
                set_result(con, fx["match_key"], int(float(r["FTHG"])), int(float(r["FTAG"])))
                n += 1
    return n


def backfill_closes(con, text: str, max_stale_min: float = 90.0) -> int:
    """Fill in missing closing prices from a finished-matches file (football-data season CSV).

    The file's PSCH/PSCD/PSCA (and PC>2.5 / PC<2.5) are Pinnacle's true closing prices. A fixture gets
    one only when our own last snapshot before kick-off is missing or older than max_stale_min, so real
    snapshots are never overwritten. The backfilled quote is stamped one minute before kick-off.
    """
    import csv, io
    from datetime import datetime, timedelta
    from .sources import norm_team, _num
    n = 0
    for r in csv.DictReader(io.StringIO(text.lstrip("\ufeff"))):
        if not r.get("Date"):
            continue
        d = datetime.strptime(r["Date"], "%d/%m/%Y" if len(r["Date"]) == 10 else "%d/%m/%y")
        h, a = norm_team(r["HomeTeam"]), norm_team(r["AwayTeam"])
        for fx in con.execute("SELECT match_key,kickoff FROM fixtures WHERE home=? AND away=?", (h, a)).fetchall():
            ko = datetime.fromisoformat(fx["kickoff"])
            if abs((ko - d).days) > 1:
                continue
            t = (ko - timedelta(minutes=1)).isoformat(timespec="seconds")
            sets = {"1x2": dict(zip("HDA", (_num(r.get("PSCH")), _num(r.get("PSCD")), _num(r.get("PSCA"))))),
                    "ou25": dict(zip("OU", (_num(r.get("PC>2.5")), _num(r.get("PC<2.5")))))}
            for market, prices in sets.items():
                if not all(prices.values()):
                    continue
                _, last = store.quote_set(con, fx["match_key"], "pinnacle", market, "latest", fx["kickoff"])
                if last is not None and (ko - datetime.fromisoformat(last)).total_seconds() / 60 <= max_stale_min:
                    continue                                   # we already have a fresh enough close
                for sel, o in prices.items():
                    store.add_quote(con, fx["match_key"], "pinnacle", market, sel, o, t)
                n += 1
    con.commit()
    return n


def bootstrap_roi(profit: np.ndarray, stake_: np.ndarray, n_boot: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(profit), (n_boot, len(profit)))
    r = profit[idx].sum(1) / stake_[idx].sum(1)
    return profit.sum() / stake_.sum(), *np.percentile(r, [2.5, 97.5])


def bets_needed(sd_unit: float, roi: float) -> int:
    """Bets for a ~95% interval of +-roi on flat stakes: (2 * sd / roi)^2."""
    return int(np.ceil((2 * sd_unit / roi) ** 2))


def clv_rows(con, max_stale_min: float = 90.0, now: str | None = None) -> list[dict]:
    now = now or store.now_utc()
    rows = []
    for b in con.execute("SELECT b.*, f.kickoff FROM bets b JOIN fixtures f USING(match_key)").fetchall():
        if b["kickoff"] > now:
            continue
        fair, t = sharp_fair(con, b["match_key"], b["market"], before=b["kickoff"])
        if fair is None:
            continue
        stale = (np.datetime64(b["kickoff"]) - np.datetime64(t)) / np.timedelta64(1, "m")
        if stale > max_stale_min:
            continue
        rows.append({"id": b["id"], "clv": b["odds"] * fair[b["selection"]] - 1, "stale_min": float(stale)})
    return rows


def margin_rows(con, book: str = "sportybet") -> list[float]:
    out = []
    for fx in con.execute("SELECT match_key FROM fixtures").fetchall():
        for m in store.SELECTIONS:
            prices, _ = store.quote_set(con, fx["match_key"], book, m, "latest")
            if prices:
                out.append(sum(1 / p for p in prices.values()) - 1)
    return out


def report(con, book: str = "sportybet") -> str:
    L = []
    bets = con.execute("SELECT * FROM bets").fetchall()
    done = [b for b in bets if b["status"] != "open"]
    L.append(f"Bets logged: {len(bets)} ({len(done)} settled, {len(bets) - len(done)} open)")
    if done:
        profit = np.array([b["profit"] for b in done], float)
        st = np.array([b["stake"] for b in done], float)
        roi, lo, hi = bootstrap_roi(profit, st)
        L.append(f"Staked {st.sum():.2f}, profit {profit.sum():+.2f}, ROI {roi*100:+.1f}% (95% interval {lo*100:+.1f}% to {hi*100:+.1f}%)")
        L.append(f"Average claimed edge at bet time: {np.mean([b['edge'] for b in done])*100:+.1f}%")
        unit = np.array([(b["odds"] - 1) if b["status"] == "won" else -1.0 for b in done])
        if len(unit) > 1 and unit.std(ddof=1) > 0:
            sd = unit.std(ddof=1)
            L.append("Bets needed to pin ROI to +-X% (flat stakes, at the observed spread): " +
                     ", ".join(f"{x}%: {bets_needed(sd, x/100):,}" for x in (2, 5, 10)))
        else:
            L.append("Bets needed: not enough settled bets to estimate the spread yet.")
    clv = clv_rows(con)
    if clv:
        c = np.array([r["clv"] for r in clv])
        se = 1.96 * c.std(ddof=1) / np.sqrt(len(c)) if len(c) > 1 else float("nan")
        L.append(f"CLV vs sharp closing fair price: {c.mean()*100:+.2f}% (+-{se*100:.2f}) on {len(c)} bets; "
                 f"average closing-snapshot age {np.mean([r['stale_min'] for r in clv]):.0f} min (closes older than 90 min are excluded)")
        L.append("Verdict rule: trust CLV first (it converges fast); trust ROI only once its interval excludes zero.")
    else:
        L.append("CLV: no bets with a usable closing snapshot yet.")
    m = margin_rows(con, book)
    if m:
        L.append(f"{book} margin: average {np.mean(m)*100:.1f}% over {len(m)} markets (min {min(m)*100:.1f}%, max {max(m)*100:.1f}%)")
    return "\n".join(L)
