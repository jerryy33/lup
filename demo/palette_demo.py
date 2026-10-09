import argparse
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from lup.plot import MARKERS, PALETTES, cycle, div_cmap, figsize, save, use_style
from lup.repro import set_repro

VENUE = "acl"
EDGE = {"edgecolor": "white", "linewidth": 0.5}
LAYERS = np.arange(24)


def tag(ax: Axes, i: int, title: str) -> None:
    ax.set_title(rf"({chr(97 + i)}) {title}", loc="left")


def build(name: str, rng: np.random.Generator) -> Figure:
    fig = Figure(figsize=figsize(VENUE, "text", ratio=0.8))
    axes = fig.subplots(2, 3).ravel()
    colors = cycle("color")
    n = len(colors)
    for i, y in enumerate(np.cumsum(rng.normal(0.3, 1.0, size=(n, 24)), axis=1)):
        axes[0].plot(LAYERS, y, label=str(i + 1))
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    tag(axes[0], 0, "lines")
    axes[1].bar(np.arange(1, n + 1), rng.uniform(0.4, 1.0, n), color=colors, **EDGE)
    axes[1].set_xticks(range(1, n + 1))
    tag(axes[1], 1, "bars")
    for i in range(n):
        pts = rng.normal(loc=(i % 4, i // 4 * 2), scale=0.35, size=(25, 2))
        axes[2].scatter(*pts.T, s=14, marker=MARKERS[i])
    tag(axes[2], 2, "scatter")
    axes[3].stackplot(np.arange(12), rng.uniform(0.5, 2, (n, 12)), **EDGE)
    tag(axes[3], 3, "stacked area")
    im = axes[4].imshow(rng.uniform(size=(10, 14)).cumsum(axis=1), aspect="auto")
    fig.colorbar(im, ax=axes[4])
    tag(axes[4], 4, "sequential")
    im = axes[5].imshow(rng.normal(size=(10, 14)), aspect="auto", cmap=div_cmap(), vmin=-3, vmax=3)
    fig.colorbar(im, ax=axes[5])
    tag(axes[5], 5, "diverging")
    for ax in axes[4:]:
        ax.grid(False)
    fig.suptitle(name)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one demo figure per lup.plot palette")
    parser.add_argument("--out", type=Path, default=Path("demo/palettes"), help="output directory")
    parser.add_argument("--palettes", nargs="*", default=None, help="palette names (default: all)")
    parser.add_argument("--seed", type=int, default=0, help="seed for the demo data")
    parser.add_argument("--no-latex", action="store_true", help="render text without LaTeX")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in args.palettes or list(PALETTES):
        use_style(VENUE, palette=name, usetex=not args.no_latex)
        rng = set_repro(args.seed, "standard").np
        print(save(build(name, rng), name, final=True, outdir=args.out))


if __name__ == "__main__":
    main()
