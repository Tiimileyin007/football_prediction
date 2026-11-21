"""Reliability diagram: does a forecast of X% come true about X% of the time?

Matches are sorted by forecast and split into equal-count bins; each point is the
average forecast against the actual rate in that bin, with a 95% binomial interval
on the actual rate. A point on the dashed diagonal is perfectly calibrated.
"""

from __future__ import annotations

# Lets this file also be run directly (for example with VS Code's "Run Python File").
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    __package__ = "footmodel.evaluation"

import numpy as np
import pandas as pd

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
# categorical slots 1-3, fixed order (validated: ensemble, raw model, market)
SERIES = {"ensemble": ("#2a78d6", "o"), "single": ("#eb6834", "s"), "market": ("#1baf7a", "D")}


def reliability_points(p, o, bins: int = 8) -> pd.DataFrame:
    p, o = np.asarray(p, dtype=float), np.asarray(o, dtype=float)
    order = np.argsort(p)
    rows = []
    for chunk in np.array_split(order, bins):
        n = len(chunk)
        rate = o[chunk].mean()
        se = np.sqrt(max(rate * (1 - rate), 1e-9) / n)
        rows.append({"forecast": p[chunk].mean(), "actual": rate, "lo": rate - 1.96 * se, "hi": rate + 1.96 * se, "n": n})
    return pd.DataFrame(rows)


def draw(panels: list[dict], path: str, table_path: str | None = None, title: str = "") -> None:
    """panels: [{"title": str, "series": {"ensemble": (label, p, o), "single": ..., "market": ...}}]"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams["font.family"] = ["Inter", "DejaVu Sans"]
    cols = 3
    rows_n = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(11.5, 4.1 * rows_n + 0.9), dpi=200, facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()
    table_rows, legend_labels = [], {}

    for ax, panel in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        lo_v, hi_v = 1.0, 0.0
        for key, (label, p, o) in panel["series"].items():
            pts = reliability_points(p, o)
            color, marker = SERIES[key]
            legend_labels[key] = label if key != "single" else label
            ax.plot(pts["forecast"], pts["actual"], color=color, lw=2, marker=marker, ms=6.5, mec=SURFACE, mew=1.5,
                    zorder=3, label=label)
            if key == "ensemble":
                ax.vlines(pts["forecast"], pts["lo"], pts["hi"], color=color, lw=1, alpha=0.45, zorder=2)
            lo_v, hi_v = min(lo_v, pts["lo"].min(), pts["forecast"].min()), max(hi_v, pts["hi"].max(), pts["forecast"].max())
            for _, r in pts.iterrows():
                table_rows.append({"panel": panel["title"], "series": label, **r.to_dict()})
        pad = 0.03
        lo_v, hi_v = max(0.0, lo_v - pad), min(1.0, hi_v + pad)
        ax.plot([lo_v, hi_v], [lo_v, hi_v], ls=(0, (4, 3)), color=MUTED, lw=1, zorder=1)
        ax.set_xlim(lo_v, hi_v)
        ax.set_ylim(lo_v, hi_v)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(panel["title"], loc="left", fontsize=12, fontweight="bold", color=INK, pad=8)
        ax.set_xlabel("Forecast probability", fontsize=9.5, color=INK2)
        ax.set_ylabel("Actual rate", fontsize=9.5, color=INK2)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    for ax in axes[len(panels):]:
        ax.axis("off")

    handles = [Line2D([0], [0], color=SERIES[k][0], lw=2, marker=SERIES[k][1], ms=6.5, mec=SURFACE, label=legend_labels[k])
               for k in ("ensemble", "single", "market") if k in legend_labels]
    handles.append(Line2D([0], [0], color=MUTED, lw=1, ls=(0, (4, 3)), label="Perfect calibration"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=10, labelcolor=INK2,
               bbox_to_anchor=(0.5, 0.965))
    if title:
        fig.suptitle(title, x=0.012, y=0.995, ha="left", fontsize=13.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    if table_path:
        pd.DataFrame(table_rows).to_csv(table_path, index=False)
