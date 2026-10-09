import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from lup.plot import (
    DIV,
    HATCH,
    LINESTYLES,
    MARKERS,
    PALETTES,
    SEQ,
    cycle,
    div_cmap,
    figsize,
    use_style,
)
from lup.repro import set_repro

LAYERS = np.arange(24)
TOKENS = ["The", "capital", "of", "France", "is", "Paris"]
VENUE = "acl"
EDGE = {"edgecolor": "white", "linewidth": 0.5}


def fig_text(ratio: float) -> Figure:
    return Figure(figsize=figsize(VENUE, "text", ratio=ratio))


def color(i: int) -> str:
    colors = cycle("color")
    return colors[i % len(colors)]


def no_grid(*axes: Axes) -> None:
    for ax in axes:
        ax.grid(False)


def legend_out(ax: Axes) -> None:
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)


def tag(ax: Axes, i: int, title: str) -> None:
    ax.set_title(rf"({chr(97 + i)}) {title}", loc="left")


def curves(rng: np.random.Generator, k: int, n: int = 24) -> np.ndarray:
    return np.cumsum(rng.normal(0.3, 1.0, size=(k, n)), axis=1)


def kde(x: np.ndarray, data: np.ndarray, bw: float) -> np.ndarray:
    diff = (x[:, None] - data[None, :]) / bw
    return np.exp(-0.5 * diff**2).sum(axis=1) / (len(data) * bw * np.sqrt(2 * np.pi))


def page_lines(rng: np.random.Generator) -> Figure:
    fig = fig_text(0.95)
    axes = fig.subplots(3, 2).ravel()
    for i, y in enumerate(curves(rng, 5)):
        axes[0].plot(LAYERS, y, label=f"series {i + 1}")
    legend_out(axes[0])
    tag(axes[0], 0, "multiple series")
    for name in ["clean", "corrupted", "patched"]:
        runs = curves(rng, 20)
        mean, std = runs.mean(0), runs.std(0)
        (line,) = axes[1].plot(LAYERS, mean, label=name)
        axes[1].fill_between(
            LAYERS, mean - std, mean + std, color=line.get_color(), alpha=0.25, linewidth=0
        )
    axes[1].legend()
    tag(axes[1], 1, "mean and band")
    for tau in (4, 8, 16):
        axes[2].semilogy(LAYERS, np.exp(-LAYERS / tau), label=rf"$\tau={tau}$")
    axes[2].legend()
    tag(axes[2], 2, "log y")
    for a in (0.5, 1.0, 1.5):
        axes[3].loglog(LAYERS + 1, (LAYERS + 1.0) ** -a, label=rf"$n^{{-{a}}}$")
    axes[3].legend()
    tag(axes[3], 3, "log-log")
    loss, acc = 3 * np.exp(-LAYERS / 9) + 0.1, 1 - np.exp(-LAYERS / 7)
    axes[4].plot(LAYERS, loss)
    twin = axes[4].twinx()
    twin.plot(LAYERS, acc, color=cycle("color")[2])
    axes[4].set_ylabel("loss")
    twin.set_ylabel("accuracy")
    tag(axes[4], 4, "twin axes")
    xs = LAYERS[::3]
    for i in range(3):
        axes[5].errorbar(
            xs, np.sin(xs / 5 + i) + i, yerr=0.3 + 0.1 * i, capsize=2, label=f"run {i}"
        )
    axes[5].legend()
    tag(axes[5], 5, "error bars")
    for ax in axes[4:]:
        ax.set_xlabel(r"layer $\ell$")
    return fig


def page_bars(rng: np.random.Generator) -> Figure:
    fig = fig_text(0.9)
    axes = fig.subplots(3, 2).ravel()
    colors, names = cycle("color"), ["base", "tuned", "steered", "ablated"]
    groups = np.arange(3)
    vals = rng.uniform(0.3, 1.0, size=(4, 3))
    w = 0.8 / 4
    for j in range(4):
        axes[0].bar(groups + (j - 1.5) * w, vals[j], w, label=names[j], **EDGE)
    axes[0].set_xticks(groups, ["A", "B", "C"])
    legend_out(axes[0])
    tag(axes[0], 0, "grouped")
    bottom = np.zeros(3)
    for j in range(4):
        axes[1].bar(groups, vals[j], 0.6, bottom=bottom, label=names[j], **EDGE)
        bottom += vals[j]
    axes[1].set_xticks(groups, ["A", "B", "C"])
    legend_out(axes[1])
    tag(axes[1], 1, "stacked")
    words = ["Paris", "Lyon", "Rome", "Berlin", "Madrid", "Vienna"]
    probs = np.sort(rng.dirichlet(np.ones(6)))[::-1]
    bars = axes[2].barh(words, probs, **EDGE)
    axes[2].bar_label(bars, fmt="%.2f", padding=2)
    axes[2].invert_yaxis()
    axes[2].margins(x=0.15)
    tag(axes[2], 2, "horizontal, labelled")
    for j in range(4):
        axes[3].bar(
            groups + (j - 1.5) * w,
            vals[j],
            w,
            yerr=0.05,
            capsize=2,
            hatch=HATCH[j],
            label=names[j],
            **EDGE,
        )
    axes[3].set_xticks(groups, ["A", "B", "C"])
    legend_out(axes[3])
    tag(axes[3], 3, "hatched, error bars")
    counts, rate = rng.integers(20, 100, 6), rng.uniform(0.2, 0.9, 6)
    axes[4].bar(np.arange(6), counts, **EDGE)
    twin = axes[4].twinx()
    twin.plot(np.arange(6), rate, color=colors[2])
    axes[4].set_ylabel("count")
    twin.set_ylabel("rate")
    tag(axes[4], 4, "bars and line")
    axes[5].stem(np.arange(10), rng.normal(size=10))
    tag(axes[5], 5, "stem")
    return fig


def page_scatter(rng: np.random.Generator) -> Figure:
    fig = fig_text(0.95)
    axes = fig.subplots(3, 2).ravel()
    for i in range(3):
        pts = rng.normal(loc=(i * 2, -i), scale=0.7, size=(40, 2))
        axes[0].scatter(*pts.T, marker=MARKERS[i], label=f"class {i}")
    axes[0].legend()
    tag(axes[0], 0, "by class")
    pts = rng.normal(size=(150, 2))
    sc = axes[1].scatter(*pts.T, c=pts[:, 0] ** 2 + pts[:, 1], s=14)
    fig.colorbar(sc, ax=axes[1])
    tag(axes[1], 1, "colored by value")
    x, y = rng.uniform(0, 1, 40), rng.uniform(0, 1, 40)
    axes[2].scatter(x, y, s=20 + 300 * rng.uniform(size=40), alpha=0.6)
    tag(axes[2], 2, "bubble sizes")
    x = rng.uniform(0, 10, 50)
    y = 0.8 * x + rng.normal(0, 1.5, 50)
    slope, icept = np.polyfit(x, y, 1)
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    axes[3].scatter(x, y, s=14)
    axes[3].plot([0, 10], [icept, icept + 10 * slope], color=cycle("color")[2])
    axes[3].text(0.05, 0.9, rf"$R^2={r2:.2f}$", transform=axes[3].transAxes)
    tag(axes[3], 3, "fit")
    pts = rng.normal(size=(2000, 2))
    hb = axes[4].hexbin(*pts.T, gridsize=18)
    fig.colorbar(hb, ax=axes[4])
    tag(axes[4], 4, "hexbin")
    _, _, _, im = axes[5].hist2d(*pts.T, bins=20)
    fig.colorbar(im, ax=axes[5])
    tag(axes[5], 5, "2D histogram")
    no_grid(axes[4], axes[5])
    return fig


def page_dist(rng: np.random.Generator) -> Figure:
    fig = fig_text(0.95)
    axes = fig.subplots(3, 2).ravel()
    data = [rng.normal(m, s, 400) for m, s in ((0, 1), (1.5, 0.7), (-1, 1.3))]
    names = ["a", "b", "c"]
    for d, n in zip(data, names, strict=True):
        axes[0].hist(d, bins=25, density=True, alpha=0.6, label=n, **EDGE)
    legend_out(axes[0])
    tag(axes[0], 0, "histograms")
    grid = np.linspace(-5, 5, 200)
    for d, n in zip(data, names, strict=True):
        axes[1].plot(grid, kde(grid, d, 0.4), label=n)
    axes[1].legend()
    tag(axes[1], 1, "density")
    parts = axes[2].violinplot(data, showmedians=True)
    for body, c in zip(parts["bodies"], cycle("color"), strict=False):
        body.set_facecolor(c)
        body.set_alpha(0.8)
    axes[2].set_xticks([1, 2, 3], names)
    tag(axes[2], 2, "violin")
    bp = axes[3].boxplot(data, patch_artist=True, tick_labels=names)
    for box, c in zip(bp["boxes"], cycle("color"), strict=False):
        box.set_facecolor(c)
    tag(axes[3], 3, "box")
    for d, n in zip(data, names, strict=True):
        axes[4].ecdf(d, label=n)
    axes[4].legend()
    tag(axes[4], 4, "ECDF")
    axes[5].hist(rng.lognormal(0, 1, 2000), bins=np.logspace(-2, 2, 30), **EDGE)
    axes[5].set_xscale("log")
    tag(axes[5], 5, "log-spaced bins")
    return fig


def page_heat(rng: np.random.Generator) -> Figure:
    fig = fig_text(0.95)
    axes = fig.subplots(3, 2).ravel()
    im = axes[0].imshow(rng.uniform(size=(10, 14)).cumsum(axis=1), aspect="auto")
    fig.colorbar(im, ax=axes[0])
    tag(axes[0], 0, "sequential")
    im = axes[1].imshow(rng.normal(size=(10, 14)), aspect="auto", cmap=div_cmap(), vmin=-3, vmax=3)
    fig.colorbar(im, ax=axes[1])
    tag(axes[1], 1, "diverging, symmetric")
    conf = rng.dirichlet(np.ones(5) * 0.4, size=5)
    axes[2].imshow(conf)
    for i in range(5):
        for j in range(5):
            color = "white" if conf[i, j] > 0.5 else "black"
            axes[2].text(j, i, f"{conf[i, j]:.2f}", ha="center", va="center", color=color)
    tag(axes[2], 2, "annotated")
    xx, yy = np.meshgrid(np.linspace(-3, 3, 60), np.linspace(-3, 3, 60))
    zz = np.exp(-(xx**2 + yy**2) / 3) * np.cos(2 * xx)
    mesh = axes[3].pcolormesh(xx, yy, zz, cmap=div_cmap(), shading="auto")
    axes[3].contour(xx, yy, zz, colors="black", linewidths=0.5)
    fig.colorbar(mesh, ax=axes[3])
    tag(axes[3], 3, "mesh and contours")
    corr = np.corrcoef(rng.normal(size=(6, 50)) + rng.normal(size=(1, 50)))
    im = axes[4].imshow(corr, cmap=div_cmap(), vmin=-1, vmax=1)
    axes[4].set_xticks(range(6), list("abcdef"))
    axes[4].set_yticks(range(6), list("abcdef"))
    fig.colorbar(im, ax=axes[4])
    tag(axes[4], 4, "correlation")
    axes[5].imshow(rng.integers(0, 4, size=(8, 12)), aspect="auto")
    tag(axes[5], 5, "categorical cells")
    no_grid(*axes)
    return fig


def page_interp(rng: np.random.Generator) -> Figure:
    fig = fig_text(1.05)
    axes = fig.subplots(3, 2).ravel()
    for name, shift in (("clean", 0.0), ("corrupted", -1.0), ("patched", -0.4)):
        runs = np.tanh((LAYERS[None] - 10) / 6) * 2 + shift + rng.normal(0, 0.3, (15, 24))
        mean, std = runs.mean(0), runs.std(0)
        (line,) = axes[0].plot(LAYERS, mean, label=name)
        axes[0].fill_between(
            LAYERS, mean - std, mean + std, color=line.get_color(), alpha=0.25, linewidth=0
        )
    axes[0].legend()
    axes[0].set_ylabel("logit difference")
    tag(axes[0], 0, "layer curves")
    acc = 0.5 + 0.45 / (1 + np.exp(-(LAYERS - 9) / 2)) + rng.normal(0, 0.01, 24)
    axes[1].plot(LAYERS, acc)
    axes[1].axhline(0.5, color="gray", ls=":")
    best = int(acc.argmax())
    axes[1].annotate(
        "best", (best, acc[best]), (best - 9, acc[best] - 0.2), arrowprops={"arrowstyle": "->"}
    )
    axes[1].set_ylabel("probe accuracy")
    tag(axes[1], 1, "probe accuracy")
    effect = rng.normal(0, 0.15, (24, 6))
    effect[8:16, 3] += 0.9
    effect[14:, 5] += 0.6
    im = axes[2].imshow(effect, aspect="auto", cmap=div_cmap(), vmin=-1, vmax=1)
    axes[2].set_xticks(range(6), TOKENS, rotation=45, ha="right")
    axes[2].set_ylabel("layer")
    fig.colorbar(im, ax=axes[2])
    tag(axes[2], 2, "activation patching")
    att = np.tril(rng.dirichlet(np.ones(6) * 0.6, size=6))
    att = att / att.sum(1, keepdims=True)
    im = axes[3].imshow(np.where(np.triu(np.ones((6, 6), bool), 1), np.nan, att))
    axes[3].set_xticks(range(6), TOKENS, rotation=45, ha="right")
    axes[3].set_yticks(range(6), TOKENS)
    fig.colorbar(im, ax=axes[3])
    tag(axes[3], 3, "attention pattern")
    no_grid(axes[2], axes[3])
    axes[4].semilogy(LAYERS, 10 ** (0.04 * LAYERS + rng.normal(0, 0.03, 24)))
    axes[4].set_ylabel(r"$\|h\|_2$")
    tag(axes[4], 4, "residual norm")
    top = ["Paris", "Lyon", "Nice", "France", "city"]
    p = np.sort(rng.dirichlet(np.ones(5)))[::-1]
    bars = axes[5].barh(top, p, **EDGE)
    axes[5].invert_yaxis()
    axes[5].bar_label(bars, fmt="%.2f", padding=2)
    axes[5].margins(x=0.15)
    tag(axes[5], 5, "logit lens, top 5")
    for ax in (axes[0], axes[1], axes[4]):
        ax.set_xlabel(r"layer $\ell$")
    axes[5].set_xlabel("probability")
    return fig


def page_layouts(rng: np.random.Generator) -> Figure:
    fig = fig_text(1.1)
    gs = fig.add_gridspec(3, 2)
    sub = gs[0, 0].subgridspec(
        2, 2, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.05, hspace=0.05
    )
    main = fig.add_subplot(sub[1, 0])
    top = fig.add_subplot(sub[0, 0], sharex=main)
    right = fig.add_subplot(sub[1, 1], sharey=main)
    pts = rng.normal(size=(300, 2))
    main.scatter(*pts.T, s=8)
    top.hist(pts[:, 0], bins=20, **EDGE)
    right.hist(pts[:, 1], bins=20, orientation="horizontal", **EDGE)
    top.tick_params(labelbottom=False)
    right.tick_params(labelleft=False)
    tag(top, 0, "marginals")
    ax = fig.add_subplot(gs[0, 1])
    x = np.linspace(0, 10, 300)
    ax.plot(x, np.sin(x) * np.exp(-x / 8))
    inset = ax.inset_axes((0.5, 0.45, 0.45, 0.45))
    inset.plot(x, np.sin(x) * np.exp(-x / 8))
    inset.set_xlim(1, 3)
    inset.set_ylim(0.3, 1)
    inset.tick_params(labelsize="x-small")
    ax.indicate_inset_zoom(inset, edgecolor="black")
    tag(ax, 1, "inset zoom")
    sub = gs[1, 0].subgridspec(3, 1, hspace=0.12)
    first = fig.add_subplot(sub[0])
    stack = [first] + [fig.add_subplot(sub[i], sharex=first) for i in (1, 2)]
    for i, a in enumerate(stack):
        a.plot(x, np.sin(x * (i + 1)))
        a.set_ylabel(f"f{i + 1}")
        if i < 2:
            a.tick_params(labelbottom=False)
    tag(stack[0], 2, "shared x")
    ax = fig.add_subplot(gs[1, 1])
    for i in range(4):
        ax.plot(x, np.cos(x + i), label=f"phase {i}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    tag(ax, 3, "legend outside")
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(x, np.sin(x))
    ax.axhline(0.5, color="gray", ls="--")
    ax.axvspan(2, 4, alpha=0.2)
    ax.annotate("peak", (np.pi / 2, 1), (4.5, 0.9), arrowprops={"arrowstyle": "->"})
    ax.text(6, -0.8, "text box", bbox={"boxstyle": "round", "facecolor": "white"})
    tag(ax, 4, "references and notes")
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(x, np.sin(x), label="signal")
    ax.fill_between(x, np.sin(x), np.sin(x) + 0.5, alpha=0.3, label="margin")
    ax.legend()
    tag(ax, 5, "filled between")
    return fig


def page_other(rng: np.random.Generator) -> Figure:
    fig = fig_text(0.95)
    gs = fig.add_gridspec(3, 2)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    axes[0].stackplot(np.arange(12), rng.uniform(1, 3, (4, 12)), labels=list("abcd"), **EDGE)
    legend_out(axes[0])
    tag(axes[0], 0, "stacked area")
    axes[1].stairs(rng.uniform(1, 4, 10), np.arange(11))
    tag(axes[1], 1, "stairs")
    xx, yy = np.meshgrid(np.linspace(-2, 2, 12), np.linspace(-2, 2, 12))
    axes[2].quiver(xx, yy, -yy, xx)
    tag(axes[2], 2, "vector field")
    xx, yy = np.meshgrid(np.linspace(-3, 3, 60), np.linspace(-3, 3, 60))
    cs = axes[3].contourf(xx, yy, np.exp(-(xx**2 + yy**2) / 4), levels=8)
    fig.colorbar(cs, ax=axes[3])
    tag(axes[3], 3, "filled contours")
    no_grid(axes[3])
    axes[4].pie(
        rng.uniform(1, 3, 5), labels=list("abcde"), colors=cycle("color")[:5], wedgeprops=EDGE
    )
    tag(axes[4], 4, "pie")
    axes[5].remove()
    polar = fig.add_subplot(gs[2, 1], projection="polar")
    angles = np.linspace(0, 2 * np.pi, 7)
    for i in range(2):
        vals = rng.uniform(0.3, 1, 7)
        vals[-1] = vals[0]
        polar.plot(angles, vals, label=f"model {i}")
        polar.fill(angles, vals, alpha=0.2)
    polar.set_xticks(angles[:-1], list("abcdef"))
    polar.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    tag(polar, 5, "radar")
    return fig


def page_column(rng: np.random.Generator) -> Figure:
    fig = Figure(figsize=figsize(VENUE, "col", ratio=1.25))
    top, bottom = fig.subplots(2, 1)
    for i, y in enumerate(curves(rng, 3)):
        top.plot(LAYERS, y, label=f"run {i + 1}")
    top.legend()
    top.set_xlabel(r"layer $\ell$")
    top.set_ylabel("score")
    tag(top, 0, "one column")
    bottom.bar(np.arange(5), rng.uniform(0.3, 1, 5), **EDGE)
    bottom.set_ylabel("accuracy")
    tag(bottom, 1, "at 7.7 cm")
    return fig


def page_palettes(rng: np.random.Generator) -> Figure:
    fig = fig_text(1.4)
    ax = fig.subplots()
    names = list(PALETTES)
    for row, name in enumerate(names):
        _, spec = PALETTES[name]
        y = len(names) - row
        ax.text(-0.15, y, name, ha="right", va="center", fontsize="small")
        for j, c in enumerate(spec.split()):
            ax.plot([j, j + 0.7], [y, y], color=f"#{c}", ls=LINESTYLES[j], linewidth=2.2)
    ax.set_xlim(-3.6, 7)
    ax.set_ylim(0.3, len(names) + 0.7)
    ax.axis("off")
    ax.set_title(f"{len(PALETTES)} palettes: color and linestyle per series")
    return fig


def page_cmaps(rng: np.random.Generator) -> Figure:
    fig = fig_text(0.8)
    axes = fig.subplots(len(SEQ), 2)
    gradient = np.linspace(0, 1, 256)[None]
    box = {"facecolor": "black", "alpha": 0.35, "pad": 1.5, "linewidth": 0}
    for row, family in enumerate(SEQ):
        for col, kind in enumerate(("seq", "div")):
            ax = axes[row, col]
            ax.imshow(gradient, aspect="auto", cmap=f"{family}_{kind}")
            ax.text(
                0.5,
                0.5,
                rf"{family}\_{kind}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="white",
                bbox=box,
            )
            ax.axis("off")
    assert SEQ.keys() == DIV.keys()
    return fig


def page_type(rng: np.random.Generator) -> Figure:
    fig = fig_text(1.1)
    gs = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(gs[0, 0])
    sizes = ["xx-small", "x-small", "small", "medium", "large", "x-large"]
    for i, size in enumerate(sizes):
        y = 1 - (i + 0.5) / (len(sizes) + 1.5)
        text = rf"{size}: The quick brown fox $\alpha+\beta=\gamma$"
        ax.text(0, y, text, fontsize=size, va="center")
    ax.text(0, 0.05, r"$\sum_{i=1}^{n}\frac{x_i}{\sqrt{n}}\;\;\int_0^1 f(x)\,dx$", va="center")
    ax.axis("off")
    tag(ax, 0, "text sizes")
    ax = fig.add_subplot(gs[0, 1])
    for i, ls in enumerate(LINESTYLES):
        ax.plot([0, 1], [i, i], ls=ls, color=color(i), linewidth=2)
        ax.text(1.05, i, f"linestyle {i + 1}", va="center")
    ax.set_xlim(0, 1.8)
    ax.invert_yaxis()
    ax.axis("off")
    tag(ax, 1, "linestyles")
    ax = fig.add_subplot(gs[1, 0])
    for i, m in enumerate(MARKERS):
        ax.plot([0.1], [i], marker=m, ls="", color=color(i), markersize=7)
        ax.text(0.3, i, f"marker {i + 1}", va="center")
    ax.set_xlim(0, 1.2)
    ax.invert_yaxis()
    ax.axis("off")
    tag(ax, 2, "markers")
    ax = fig.add_subplot(gs[1, 1])
    for i, h in enumerate(HATCH):
        ax.add_patch(Rectangle((0, i), 0.8, 0.8, facecolor=color(i), hatch=h, **EDGE))
        ax.text(1.0, i + 0.4, f"hatch {i + 1}", va="center")
    ax.set_xlim(0, 1.9)
    ax.set_ylim(len(HATCH), 0)
    ax.axis("off")
    tag(ax, 3, "hatches")
    return fig


PAGES: list[Callable[[np.random.Generator], Figure]] = [
    page_lines,
    page_bars,
    page_scatter,
    page_dist,
    page_heat,
    page_interp,
    page_layouts,
    page_other,
    page_column,
    page_palettes,
    page_cmaps,
    page_type,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a multi-page demo of lup.plot styles")
    parser.add_argument("--out", type=Path, default=Path("demo/plot_demo.pdf"), help="output PDF")
    parser.add_argument("--seed", type=int, default=0, help="seed for the demo data")
    parser.add_argument("--no-latex", action="store_true", help="render text without LaTeX")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_style(VENUE, usetex=not args.no_latex)
    rng = set_repro(args.seed, "standard").np
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out, metadata={"Title": "lup plot demo (ACL, default palette)"}) as pdf:
        for build in PAGES:
            pdf.savefig(build(rng))
    print(args.out)


if __name__ == "__main__":
    main()
