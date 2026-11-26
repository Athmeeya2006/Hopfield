"""Shared figure style (palette follows the project deck: deep greens)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK = "#2f4b3c"
GRID = "#d9d6cc"
# ordered sequential greens (light -> dark) and a categorical set
GREENS = ["#a9b98f", "#7a9c6a", "#4f7d55", "#2f5a40", "#1b3a2a"]
CAT = {
    "M1-Hebb": "#1b3a2a",
    "M1-Storkey": "#7a9c6a",
    "M2-n3": "#c27c2c",
    "M2-n4": "#8c3b2a",
    "M2-exp": "#3f6aa1",
    "M3": "#6c4f9c",
    "M3-b1": "#6c4f9c",
    "M3-b0.1": "#a38bd0",
    "M4-softmax": "#2a7f86",
    "M4-power": "#b5527a",
}
STATE_COLORS = {"target": "#2f5a40", "other memory": "#a9b98f",
                "prototype": "#c27c2c", "spurious": "#8a8a8a", "blend": "#8a8a8a"}


def style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "text.color": INK, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "lines.linewidth": 1.6, "lines.markersize": 3.5,
    })


def band(ax, x, lo, hi, color, alpha=0.18):
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, lw=0)


def seq(n):
    """n sequential greens light -> dark."""
    if n <= len(GREENS):
        idx = np.linspace(0, len(GREENS) - 1, n).round().astype(int)
        return [GREENS[i] for i in idx]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("g", GREENS)
    return [cmap(t) for t in np.linspace(0, 1, n)]


def save(fig, path):
    fig.savefig(path)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)
