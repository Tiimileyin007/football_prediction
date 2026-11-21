"""Edge, shrinkage toward the market, and stake sizing.

The fair probability is a log-linear blend of the sharp market (Pinnacle, margin
removed) and our model, with weight w on the model. Phase 3 found w = 0 in every
season, so the default is deliberately small: with w = 0 the engine is pure price
shopping (a soft book's price against the sharp book's fair price), which is the
effect the historical backtest measured.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-9


def pool(p_market, p_model, w: float) -> np.ndarray:
    """Blend two probability vectors (summing to 1) as market^(1-w) * model^w, renormalised."""
    lm = np.log(np.clip(np.asarray(p_market, dtype=float), EPS, 1.0))
    lb = np.log(np.clip(np.asarray(p_model, dtype=float), EPS, 1.0))
    z = (1.0 - w) * lm + w * lb
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


def edge(p: float, odds: float) -> float:
    """Expected profit per unit staked: p * odds - 1."""
    return p * odds - 1.0


def kelly_fraction(p: float, odds: float) -> float:
    b = odds - 1.0
    return max(0.0, (p * b - (1.0 - p)) / b) if b > 0 else 0.0


def stake(p: float, odds: float, bankroll: float, kelly_mult: float = 0.25, cap: float = 0.02) -> float:
    """Fractional Kelly, capped at `cap` of the bankroll. Returns 0 when there is no edge."""
    return round(bankroll * min(cap, kelly_mult * kelly_fraction(p, odds)), 2)


def candidates(fair: dict, prices: dict, min_edge: float = 0.03, max_odds: float = 12.0) -> list[dict]:
    """Selections whose price beats the fair probability by at least min_edge.

    fair and prices map selection -> probability / decimal odds. Long shots above
    max_odds are skipped: the fair probability there is least reliable.
    """
    out = []
    for sel, odds in prices.items():
        if sel not in fair or not odds or odds <= 1.0 or odds > max_odds:
            continue
        e = edge(fair[sel], odds)
        if e >= min_edge:
            out.append({"selection": sel, "odds": odds, "p_fair": fair[sel], "edge": e})
    return sorted(out, key=lambda r: -r["edge"])
