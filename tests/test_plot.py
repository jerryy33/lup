import math
import re
import shutil
import subprocess
import sys
from collections.abc import Iterator
from itertools import pairwise
from pathlib import Path

import matplotlib as mpl
import pytest
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.layout_engine import ConstrainedLayoutEngine

from lup.plot import (
    DEFAULT_PALETTE,
    DIV,
    LINESTYLES,
    PALETTES,
    SEQ,
    STYLE_FILE,
    VENUES,
    cycle,
    div_cmap,
    figsize,
    headroom,
    palette_rc,
    register,
    save,
    use_style,
)

HAS_LATEX = shutil.which("latex") is not None and shutil.which("dvipng") is not None

NO_SIDE_EFFECTS = """
import sys
import matplotlib as mpl

def snapshot():
    return {k: mpl.rcParams[k] for k in mpl.rcParams if k != "backend"}

before, styles, cmaps = snapshot(), list(mpl.style.available), set(mpl.colormaps)
import lup.plot
assert "matplotlib.pyplot" not in sys.modules, "pyplot was imported"
assert snapshot() == before, "rcParams changed"
assert list(mpl.style.available) == styles, "styles registered"
assert set(mpl.colormaps) == cmaps, "colormaps registered"
print("ok")
"""


@pytest.fixture(autouse=True)
def restore_rc() -> Iterator[None]:
    with mpl.rc_context():
        yield


@pytest.mark.parametrize(
    ("venue", "span", "width"),
    [
        ("icml", "col", 3.25),
        ("icml", "text", 6.75),
        ("iclr", "col", 5.5),
        ("iclr", "text", 5.5),
        ("neurips", "col", 5.5),
        ("colm", "text", 5.5),
        ("tmlr", "col", 6.5),
        ("acl", "col", 77 / 25.4),
        ("acl", "text", 160 / 25.4),
        ("emnlp", "col", 77 / 25.4),
        ("nature", "col", 89 / 25.4),
        ("nature", "text", 183 / 25.4),
    ],
)
def test_widths(venue: str, span: str, width: float) -> None:
    assert figsize(venue, span)[0] == pytest.approx(width)


def test_acl_columns_fill_text_width() -> None:
    col, text = figsize("acl", "col")[0], figsize("acl", "text")[0]
    assert 2 * col + 6 / 25.4 == pytest.approx(text)


def test_icml_columns_fill_text_width() -> None:
    col, text = figsize("icml", "col")[0], figsize("icml", "text")[0]
    assert 2 * col + 0.25 == pytest.approx(text)


def test_frac_ratio_and_height() -> None:
    assert figsize("icml", "text", frac=0.5)[0] == pytest.approx(3.375)
    w, h = figsize("neurips", ratio=0.5)
    assert h == pytest.approx(w / 2)
    assert figsize("neurips", height=2.0)[1] == 2.0


def test_nature_height_is_capped() -> None:
    assert figsize("nature", "text", ratio=2.0)[1] == pytest.approx(170 / 25.4)


def test_venue_name_is_case_insensitive() -> None:
    assert figsize("ICML") == figsize("icml")


def test_unknown_venue_and_span() -> None:
    with pytest.raises(ValueError, match="unknown venue"):
        figsize("cvpr")
    with pytest.raises(ValueError, match="span must be"):
        figsize("icml", "page")  # type: ignore[arg-type]


def test_all_venues_have_positive_sizes() -> None:
    for name in VENUES:
        w, h = figsize(name)
        assert 0 < h < w


def test_import_has_no_side_effects() -> None:
    out = subprocess.run(
        [sys.executable, "-c", "import matplotlib.style\n" + NO_SIDE_EFFECTS],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr[-1500:]
    assert out.stdout.strip() == "ok"


def test_style_file_ships_with_the_package() -> None:
    assert STYLE_FILE.is_file()
    assert STYLE_FILE.parent.name == "lup"
    assert "text.usetex: True" in STYLE_FILE.read_text()


def test_register_adds_colormaps_and_styles_idempotently() -> None:
    register()
    register()
    for family in SEQ:
        assert f"{family}_seq" in mpl.colormaps
    for family in DIV:
        assert f"{family}_div" in mpl.colormaps
    available = mpl.style.available
    assert {"paper", *VENUES, *PALETTES} <= set(available)
    assert len(available) == len(set(available))
    assert available == sorted(available)


def test_sequential_and_diverging_families_match() -> None:
    assert SEQ.keys() == DIV.keys()


def test_colormaps_run_from_first_to_last_color() -> None:
    register()
    for family, spec in SEQ.items():
        first, last = (f"#{c}" for c in (spec.split()[0], spec.split()[-1]))
        cmap = mpl.colormaps[f"{family}_seq"]
        assert cmap.N == 256
        assert cmap(0.0) == pytest.approx(to_rgba(first))
        assert cmap(1.0) == pytest.approx(to_rgba(last))


def test_there_are_24_palettes_and_the_default_exists() -> None:
    assert len(PALETTES) == 24
    assert DEFAULT_PALETTE == "kiki-medium"
    assert DEFAULT_PALETTE in PALETTES


def test_palette_names_cover_every_family_and_shade() -> None:
    shades = ("dark", "light", "medium")
    families = [f for f in SEQ if f not in ("marnie1", "marnie2")]
    expected = {f"{fam}-{shade}" for fam in families for shade in shades}
    expected |= {f"marnie-{shade}{i}" for shade in shades for i in (1, 2)}
    assert set(PALETTES) == expected


def linear(channel: int) -> float:
    c = channel / 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def oklab(hexcol: str) -> tuple[float, float, float]:
    r, g, b = (linear(int(hexcol[i : i + 2], 16)) for i in (0, 2, 4))
    lm = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    mm = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    sm = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (
        0.2104542553 * lm + 0.7936177850 * mm - 0.0040720468 * sm,
        1.9779984951 * lm - 2.4285922050 * mm + 0.4505937099 * sm,
        0.0259040371 * lm + 0.7827717662 * mm - 0.8086757660 * sm,
    )


def color_distance(a: str, b: str) -> float:
    return 100 * math.dist(oklab(a), oklab(b))


@pytest.mark.parametrize("name", sorted(PALETTES))
def test_consecutive_palette_colors_are_distinguishable(name: str) -> None:
    colors = PALETTES[name][1].split()
    assert len(set(colors)) == 7
    assert min(color_distance(a, b) for a, b in pairwise(colors)) >= 3.5


def test_curated_families_keep_the_pick_order_in_their_medium_palette() -> None:
    first = {fam: PALETTES[f"{fam}-medium"][1].split() for fam in ("kiki", "mononoke", "ponyo")}
    assert first["kiki"][:5] == ["0E84B4", "B50A2A", "9E8356", "1C1A1F", "7EBAC2"]
    assert first["mononoke"][:6] == ["3D4F7D", "E48C2A", "657060", "CD4F38", "06141F", "742C14"]
    assert first["ponyo"][:5] == ["278B9A", "E75B64", "D8AF39", "4C413F", "5A6F80"]


@pytest.mark.parametrize("name", sorted(PALETTES))
def test_palette_is_valid(name: str) -> None:
    register()
    cmap, spec = PALETTES[name]
    colors = spec.split()
    assert len(colors) == 7
    assert all(len(c) == 6 and int(c, 16) >= 0 for c in colors)
    assert cmap in mpl.colormaps
    by_key = palette_rc(name)["axes.prop_cycle"].by_key()
    assert by_key["color"] == [f"#{c}" for c in colors]
    assert by_key["linestyle"] == LINESTYLES[: len(colors)]
    assert by_key.keys() == {"color", "linestyle"}


@pytest.mark.parametrize("name", sorted(PALETTES))
def test_every_palette_has_a_diverging_partner(name: str) -> None:
    use_style(palette=name, usetex=False)
    assert div_cmap() in mpl.colormaps


def test_named_styles_apply_through_matplotlib() -> None:
    register()
    mpl.style.use(["paper", "icml", "ponyo-medium"])
    assert mpl.rcParams["axes.grid"] is True
    assert mpl.rcParams["font.size"] == VENUES["icml"].fig_pt
    assert tuple(mpl.rcParams["figure.figsize"]) == pytest.approx(figsize("icml"))
    assert cycle("color")[0] == "#278B9A"
    assert mpl.rcParams["image.cmap"] == "ponyo_seq"


def test_base_style_loads_by_package_name_without_registering() -> None:
    mpl.style.use("lup.paper")
    assert mpl.rcParams["axes.grid"] is True
    assert mpl.rcParams["text.usetex"] is True


def test_use_style_defaults() -> None:
    use_style()
    assert mpl.rcParams["text.usetex"] is True
    assert mpl.rcParams["font.family"] == ["sans-serif"]
    assert mpl.rcParams["font.size"] == 10
    assert tuple(mpl.rcParams["figure.figsize"]) == (6.4, 4.8)
    assert mpl.rcParams["pdf.fonttype"] == 42
    assert mpl.rcParams["savefig.dpi"] == 300
    assert mpl.rcParams["image.cmap"] == "kiki_seq"
    assert cycle("color") == [f"#{c}" for c in PALETTES[DEFAULT_PALETTE][1].split()]


def test_use_style_venue_sets_font_size_and_figure_size() -> None:
    use_style("ICML")
    assert mpl.rcParams["font.size"] == VENUES["icml"].fig_pt
    assert tuple(mpl.rcParams["figure.figsize"]) == pytest.approx(figsize("icml"))
    assert mpl.rcParams["axes.grid"] is True


def test_use_style_palette_and_usetex_arguments() -> None:
    use_style("neurips", palette="marnie-dark1", usetex=False)
    assert mpl.rcParams["text.usetex"] is False
    assert mpl.rcParams["image.cmap"] == "marnie1_seq"
    assert cycle("color")[0] == "#004749"
    assert cycle("linestyle") == LINESTYLES


def test_venue_style_turns_on_constrained_layout() -> None:
    use_style("icml", usetex=False)
    assert mpl.rcParams["figure.constrained_layout.use"] is True
    assert isinstance(Figure().get_layout_engine(), ConstrainedLayoutEngine)


def test_base_style_alone_keeps_the_default_layout() -> None:
    use_style(usetex=False)
    assert mpl.rcParams["figure.constrained_layout.use"] is False
    assert Figure().get_layout_engine() is None


def test_palette_cycle_has_no_markers() -> None:
    use_style(usetex=False)
    assert mpl.rcParams["axes.prop_cycle"].keys == {"color", "linestyle"}


def test_fill_works_with_the_palette_cycle() -> None:
    use_style(usetex=False)
    Figure().subplots().fill([0, 1, 1], [0, 0, 1])


def test_style_keeps_legends_frameless_and_patch_edges_visible() -> None:
    use_style(usetex=False)
    assert mpl.rcParams["legend.frameon"] is False
    assert mpl.rcParams["patch.force_edgecolor"] is False
    assert mpl.rcParams["patch.edgecolor"] == "black"


def test_annotation_arrows_are_visible() -> None:
    use_style(usetex=False)
    ax = Figure().subplots()
    note = ax.annotate("x", (1, 1), (0.5, 0.5), arrowprops={"arrowstyle": "->"})
    assert note.arrow_patch.get_edgecolor()[:3] == (0.0, 0.0, 0.0)


def test_bars_have_no_edge_unless_asked() -> None:
    use_style(usetex=False)
    ax = Figure().subplots()
    plain = ax.bar([0], [1])[0]
    white = ax.bar([1], [1], edgecolor="white")[0]
    assert plain.get_edgecolor()[3] == 0.0 or plain.get_linewidth() == 0
    assert white.get_edgecolor()[:3] == (1.0, 1.0, 1.0)


def test_use_style_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="unknown palette"):
        use_style(palette="shrek")
    with pytest.raises(ValueError, match="unknown venue"):
        use_style("cvpr")


def test_div_cmap_follows_the_palette_family() -> None:
    use_style(palette="ponyo-medium", usetex=False)
    assert div_cmap() == "ponyo_div"
    use_style(palette="marnie-dark1", usetex=False)
    assert div_cmap() == "marnie1_div"
    use_style(palette="mononoke-medium", usetex=False)
    assert div_cmap() == "mononoke_div"
    mpl.rcParams["image.cmap"] = "viridis"
    assert div_cmap() == "coolwarm"


def test_cycle_returns_one_property() -> None:
    use_style(palette="ponyo-medium", usetex=False)
    assert len(cycle("color")) == 7
    assert cycle("linestyle")[0] == "-"
    assert cycle("linestyle") == LINESTYLES


def test_headroom_extends_upper_limit() -> None:
    ax = Figure().subplots()
    ax.set_ylim(0, 2)
    headroom(ax, 0.25)
    assert ax.get_ylim() == pytest.approx((0, 2.5))


@pytest.mark.parametrize(("final", "suffix"), [(False, ".png"), (True, ".pdf")])
def test_save_writes_png_or_pdf(tmp_path: Path, final: bool, suffix: str) -> None:
    use_style("iclr", usetex=False)
    fig = Figure(figsize=figsize("iclr"))
    fig.subplots().plot([0, 1], [0, 1])
    path = save(fig, "curve", final=final, outdir=tmp_path / "nested" / "figs")
    assert path == tmp_path / "nested" / "figs" / f"curve{suffix}"
    assert path.stat().st_size > 0


EMBEDDED_FONT = {
    "icml": "NimbusRomNo9L",
    "iclr": "NimbusRomNo9L",
    "neurips": "NimbusRomNo9L",
    "acl": "NimbusRomNo9L",
    "emnlp": "NimbusRomNo9L",
    "colm": "URWPalladioL",
    "tmlr": "LMRoman",
    "nature": "NimbusSanL",
}


def embedded_fonts(pdf: Path) -> set[str]:
    pattern = rb"/(?:BaseFont|FontName)\s*/(?:[A-Z]{6}\+)?([A-Za-z0-9\-]+)"
    return {m.decode() for m in re.findall(pattern, pdf.read_bytes())}


def test_every_venue_has_a_font_expectation() -> None:
    assert EMBEDDED_FONT.keys() == VENUES.keys()


@pytest.mark.parametrize("venue", sorted(VENUES))
def test_venue_style_sets_the_venue_font(venue: str) -> None:
    use_style(venue, usetex=False)
    spec = VENUES[venue]
    assert mpl.rcParams["font.family"] == [spec.family]
    assert mpl.rcParams["text.latex.preamble"] == spec.preamble


def test_without_a_venue_the_base_font_is_kept() -> None:
    use_style(usetex=False)
    assert mpl.rcParams["font.family"] == ["sans-serif"]
    assert mpl.rcParams["text.latex.preamble"] == ""


@pytest.mark.skipif(not HAS_LATEX, reason="latex not installed")
@pytest.mark.parametrize("venue", sorted(VENUES))
def test_pdf_embeds_the_venue_font(tmp_path: Path, venue: str) -> None:
    use_style(venue)
    fig = Figure(figsize=figsize(venue))
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1], label="series")
    ax.set_xlabel(r"layer $\ell$ and text")
    ax.legend()
    fonts = embedded_fonts(save(fig, "fonts", final=True, outdir=tmp_path))
    assert any(f.startswith(EMBEDDED_FONT[venue]) for f in fonts), fonts
    if venue == "nature":
        assert not any(f.startswith("CMR") for f in fonts), fonts


@pytest.mark.skipif(not HAS_LATEX, reason="latex not installed")
def test_save_with_latex(tmp_path: Path) -> None:
    use_style("neurips")
    fig = Figure(figsize=figsize("neurips"))
    ax = fig.subplots()
    for i in range(3):
        ax.plot([0, 1], [0, i + 1], label=rf"series $s_{i}$")
    ax.set_xlabel(r"layer $\ell$")
    ax.legend()
    assert save(fig, "tex", final=True, outdir=tmp_path).stat().st_size > 0
