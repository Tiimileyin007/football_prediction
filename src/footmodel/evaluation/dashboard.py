"""Performance dashboard: one picture of how every model is doing.

    python -m footmodel.evaluation.dashboard --out data/results [--db football.db]

Reads the CSV tables the run scripts write and draws data/results/performance_dashboard.png.
The run scripts call it automatically when they finish, so the picture is refreshed every
time you run the code. Panels whose input files do not exist yet are skipped.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
OURS, SINGLE, MARKET, BASE = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8ae"

NAMES = {
    "base_rate_all": "Base rates (all history)", "base_rate_last3": "Base rates (last 3 seasons)", "elo_logit": "Elo",
    "dixon_coles": "Dixon–Coles", "poisson_features": "Poisson + form", "ens_avg3": "Average of 3 models",
    "ens_avg3_cal": "Average, calibrated", "ens_stack3": "Stacked ensemble", "ens_pool_avg3": "Ensemble pooled with market",
    "ens_goals_avg": "Goals average", "ens_goals_cal": "Goals ensemble, calibrated", "ens_o25_pool": "Ensemble pooled with market",
    "market_avg_open": "Market average (open)", "market_ps_open": "Pinnacle (open)", "market_ps_close": "Pinnacle (close)",
}


def _name(m: str) -> str:
    return NAMES.get(m, m)


def _color(m: str) -> str:
    if m.startswith("market"):
        return MARKET
    if m.startswith("base") or m == "elo_logit":
        return BASE
    return OURS


def _csv(out: Path, name: str):
    p = out / name
    return pd.read_csv(p) if p.exists() else None


def _style(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=8)
    ax.set_xlabel(xlabel, fontsize=9.5, color=INK2)
    ax.set_ylabel(ylabel, fontsize=9.5, color=INK2)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def _bars(ax, df: pd.DataFrame, col: str, title: str, xlabel: str, zoom: bool = True):
    d = df.sort_values(col, ascending=False)            # best (lowest) ends up on top
    y = np.arange(len(d))
    lo = d[col].min()
    left = (lo - 0.6 * (d[col].max() - lo) - 0.004) if zoom else 0
    ax.barh(y, d[col] - left, left=left, color=[_color(m) for m in d["model"]], height=0.62)
    for yi, v in zip(y, d[col]):
        ax.text(v + (d[col].max() - left) * 0.012, yi, f"{v:.4f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([_name(m) for m in d["model"]], fontsize=9, color=INK)
    ax.set_xlim(left, d[col].max() + (d[col].max() - left) * 0.16)
    _style(ax, title, xlabel)
    ax.grid(axis="y", visible=False)


def _legend(fig):
    from matplotlib.patches import Patch
    h = [Patch(color=OURS, label="Our models"), Patch(color=BASE, label="Baselines (base rate, Elo)"), Patch(color=MARKET, label="Bookmaker market")]
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, fontsize=10, labelcolor=INK2, bbox_to_anchor=(0.5, 0.955))


def _by_season(ax, out: Path):
    for f in ("poisson_by_season.csv", "phase2_by_season.csv", "table_by_season.csv"):
        d = _csv(out, f)
        if d is not None:
            break
    else:
        ax.axis("off"); return
    x = np.arange(len(d))
    style = {"elo_logit": (BASE, "o", "Elo"), "dixon_coles": (SINGLE, "s", "Dixon–Coles"), "poisson_features": (OURS, "o", "Poisson + form"),
             "market_ps_close": (MARKET, "D", "Pinnacle (close)"), "base_rate_last3": (BASE, "^", "Base rates")}
    for col, (c, mk, lab) in style.items():
        if col in d:
            ax.plot(x, d[col], color=c, marker=mk, ms=5.5, lw=2, mec=SURFACE, mew=1.2, label=lab, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(d["season"], rotation=45, ha="right")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.22 * (hi - lo))                  # headroom so the legend never covers a line
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left", ncol=3)
    _style(ax, "1X2 log loss by season (lower is better)", "", "Log loss")


def _calibration(ax, out: Path):
    d = _csv(out, "ensemble_calibration_ece.csv")
    if d is None:
        ax.axis("off"); return
    mk = list(dict.fromkeys(d["market"]))
    models = list(dict.fromkeys(d["model"]))
    h = 0.8 / len(models)
    for i, m in enumerate(models):
        g = d[d["model"] == m].set_index("market").reindex(mk)
        y = np.arange(len(mk)) + (i - (len(models) - 1) / 2) * h
        c = MARKET if m.startswith("market") else (OURS if m.startswith("ens") else SINGLE)
        ax.errorbar(g["ece"], y, xerr=[g["ece"] - g["lo"], g["hi"] - g["ece"]], fmt="o", color=c, ms=6, mec=SURFACE, lw=1, capsize=0, label=_name(m))
    ax.set_yticks(np.arange(len(mk)))
    ax.set_yticklabels(mk, fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2)
    _style(ax, "Calibration error (lower is better)", "Expected calibration error, with 95% interval")
    ax.grid(axis="y", visible=False)


def _accuracy(ax, df: pd.DataFrame):
    d = df.sort_values("accuracy")
    y = np.arange(len(d))
    ax.hlines(y, d["balanced_acc"], d["accuracy"], color=AXIS, lw=2, zorder=1)
    ax.scatter(d["accuracy"], y, color=OURS, s=46, zorder=3, label="Accuracy", edgecolor=SURFACE)
    ax.scatter(d["balanced_acc"], y, color=SINGLE, marker="s", s=40, zorder=3, label="Balanced accuracy", edgecolor=SURFACE)
    ax.set_yticks(y)
    ax.set_yticklabels([_name(m) for m in d["model"]], fontsize=9, color=INK)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    _style(ax, "Right calls: accuracy vs balanced accuracy", "Share of matches")
    ax.grid(axis="y", visible=False)


def _betting(ax, db: str | None):
    _style(ax, "Betting record (value engine)")
    rows = []
    if db and Path(db).exists():
        con = sqlite3.connect(db)
        try:
            rows = con.execute("SELECT placed_at, profit FROM bets WHERE status IN ('won','lost') ORDER BY placed_at").fetchall()
        except sqlite3.Error:
            rows = []
    if not rows:
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.5, "No settled bets yet.\nLog bets with the value engine and\nthis panel shows cumulative profit.", ha="center", va="center", fontsize=10, color=MUTED, transform=ax.transAxes)
        return
    cum = np.cumsum([r[1] for r in rows])
    ax.plot(np.arange(1, len(cum) + 1), cum, color=OURS, lw=2)
    ax.axhline(0, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_xlabel("Bets settled", fontsize=9.5, color=INK2)
    ax.set_ylabel("Cumulative profit (stake units)", fontsize=9.5, color=INK2)
    ax.text(0.02, 0.95, f"{len(cum)} bets, profit {cum[-1]:+.2f}. Too few bets to judge until the report shows a clear interval.", transform=ax.transAxes, fontsize=8.5, color=INK2, va="top", wrap=True)


def build(out: Path | str = "data/results", db: str | None = None, show: bool = False) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    one = _csv(out, "ensemble_1x2.csv")
    if one is None:
        one = _csv(out, "table_1x2_with_pinnacle_close.csv")
    if one is None:
        print("Dashboard skipped: no 1X2 tables found yet in", out)
        return None
    o25 = _csv(out, "ensemble_over25.csv")
    if o25 is None:
        o25 = _csv(out, "table_over25.csv")

    plt.rcParams["font.family"] = ["Inter", "DejaVu Sans"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.6), dpi=170, facecolor=SURFACE)
    n = int(one["n"].iloc[0])
    _bars(axes[0, 0], one, "log_loss", f"1X2 log loss, {n:,} matches (lower is better)", "Log loss (axis starts near the best score)")
    _by_season(axes[0, 1], out)
    if o25 is not None:
        _bars(axes[0, 2], o25, "log_loss", "Over 2.5 goals log loss (lower is better)", "Log loss (axis starts near the best score)")
    else:
        axes[0, 2].axis("off")
    _bars(axes[1, 0], one, "brier", "1X2 Brier score (lower is better)", "Brier score")
    _accuracy(axes[1, 1], one)
    # calibration and betting share the last cell when both exist: calibration first, betting below
    _calibration(axes[1, 2], out) if (out / "ensemble_calibration_ece.csv").exists() else _betting(axes[1, 2], db)
    if db and (out / "ensemble_calibration_ece.csv").exists():
        fig2, ax2 = plt.subplots(figsize=(7, 4.2), dpi=170, facecolor=SURFACE)
        _betting(ax2, db)
        fig2.tight_layout(); fig2.savefig(out / "betting_record.png", facecolor=SURFACE); plt.close(fig2)
    fig.suptitle("Model performance dashboard", x=0.012, y=0.995, ha="left", fontsize=15, fontweight="bold", color=INK)
    _legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out / "performance_dashboard.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print(f"Performance dashboard: {path}")
    if show or os.environ.get("FOOTMODEL_SHOW") == "1":
        import webbrowser
        webbrowser.open(path.resolve().as_uri())            # opens in your default image viewer or browser
    return path


def build_from_argv() -> None:
    """Used by the run scripts: reads --out (and --db) from the command line they were started with."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--out", default="data/results")
    ap.add_argument("--db", default=None)
    a, _ = ap.parse_known_args(sys.argv[1:])
    try:
        build(a.out, a.db)
    except Exception as e:                                  # the dashboard must never break a model run
        print(f"(dashboard not drawn: {e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/results")
    ap.add_argument("--db", default=None)
    ap.add_argument("--show", action="store_true", help="open the picture when done")
    a = ap.parse_args()
    build(a.out, a.db, a.show)
