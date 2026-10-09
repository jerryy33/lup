"""Figure styling: a base style, venue styles, color palettes, venue figure sizes and save().

Everything is plain data plus a few functions; importing this module changes nothing in
matplotlib. `use_style` (or `register`) adds the named styles and colormaps on demand.

Styles are layered with `plt.style.use([...])`, later entries overriding earlier ones:
    "paper": the base style, `paper.mplstyle` next to this file (also loadable as "lup.paper"
        without registering anything).
    a venue ("icml", "iclr", ...): the venue's body font (family and LaTeX preamble), figure font
        size, the default single-column figure size and constrained layout.
    a palette ("kiki-medium", ...): color and linestyle cycle and the image colormap.
        Markers are not in the cycle (a marker on every point is overwhelming on dense lines
        and `markevery` in a cycle breaks `Axes.fill`); `MARKERS` lists the marker order for
        explicit use.

Widths are the official text-block sizes of each venue's 2026 template, so a figure made at
`figsize(...)` is included at 1:1 and its fonts keep the intended size:
    ICML: 2 columns, 6.75 in overall with a 0.25 in gap (ICML 2026 example paper).
    ICLR, NeurIPS, COLM: single column, 5.5 in wide (ICLR 2026 style file, NeurIPS 2026 and
        COLM 2026 template instructions).
    ACL, EMNLP: A4, 2 columns of 7.7 cm with a 0.6 cm gap, 16 cm overall (ACLPUB guidelines).
    TMLR: single column, 6.5 in wide (tmlr.sty).
    Nature: 89 mm (1 column) and 183 mm (2 columns), at most 170 mm high, figure text 5-7 pt
        (Nature research figure guide).
Figure font sizes (`Venue.fig_pt`) are defaults chosen here, except for Nature.

Figure text uses the venue's body font, rendered by LaTeX (math stays Computer Modern except
for TMLR and Nature, as in the papers):
    ICML, ICLR, NeurIPS, ACL, EMNLP: Times (`\\usepackage{times}`). Confirmed from the ICML 2026
        example paper and the NeurIPS 2026 instructions (both embed Nimbus Roman No. 9 with
        Computer Modern math), the ICLR 2026 template and the ACLPUB guidelines (Times Roman).
    COLM: Palatino, which the COLM 2026 template makes mandatory for the main text. Its math font
        is unverified, so math is left as Computer Modern.
    TMLR: Latin Modern (`\\usepackage{lmodern}`, loaded by tmlr.sty).
    Nature: Helvetica (Nature asks for standard fonts, e.g. Arial or Helvetica), including digits.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

try:
    import matplotlib as mpl
    import matplotlib.style
    from cycler import cycler
    from matplotlib.axes import Axes
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    from matplotlib.figure import Figure
except ImportError as err:
    raise ImportError("lup.plot needs matplotlib; install it with the `lup[plot]` extra") from err

IN_PER_MM = 1 / 25.4
GOLDEN = 0.618
STYLE_FILE = Path(__file__).with_name("paper.mplstyle")
DEFAULT_PALETTE = "kiki-medium"

type Span = Literal["col", "text"]


TIMES = r"\usepackage{times}"
PALATINO = r"\usepackage{palatino}"
LMODERN = r"\usepackage{lmodern}"
HELVETICA = r"\usepackage{helvet}\usepackage{sfmath}"


@dataclass(frozen=True, slots=True)
class Venue:
    text: float
    col: float | None
    fig_pt: float
    family: str
    preamble: str
    max_height: float | None = None


ACL = Venue(text=160 * IN_PER_MM, col=77 * IN_PER_MM, fig_pt=9, family="serif", preamble=TIMES)

VENUES: dict[str, Venue] = {
    "icml": Venue(text=6.75, col=3.25, fig_pt=8, family="serif", preamble=TIMES),
    "iclr": Venue(text=5.5, col=None, fig_pt=8, family="serif", preamble=TIMES),
    "neurips": Venue(text=5.5, col=None, fig_pt=8, family="serif", preamble=TIMES),
    "colm": Venue(text=5.5, col=None, fig_pt=8, family="serif", preamble=PALATINO),
    "acl": ACL,
    "emnlp": ACL,
    "tmlr": Venue(text=6.5, col=None, fig_pt=8, family="serif", preamble=LMODERN),
    "nature": Venue(
        text=183 * IN_PER_MM,
        col=89 * IN_PER_MM,
        fig_pt=7,
        family="sans-serif",
        preamble=HELVETICA,
        max_height=170 * IN_PER_MM,
    ),
}

LINESTYLES: list[Any] = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (8, 3))]
MARKERS: list[str] = ["o", "s", "^", "D", "v", "P", "X"]

PALETTES: dict[str, tuple[str, str]] = {
    "kiki-dark": ("kiki_seq", "06425A 590514 695B50 50412A 1A1A22 405D61 0E0C0F"),
    "kiki-light": ("kiki_seq", "D98594 86C2DA 8E8C8F D0C1AA 9A9AA2 C0DDE1 E9DBD0"),
    "kiki-medium": ("kiki_seq", "0E84B4 B50A2A 9E8356 1C1A1F 7EBAC2 333544 D1B79E"),
    "laputa-dark": ("laputa_seq", "776A3D 1F1935 574A5E 5A6D73 2F2C49 0D1321 090D10"),
    "laputa-light": ("laputa_seq", "9F99B5 D7CADE 898D90 AFACC9 8D93A1 F7EABD DAEDF3"),
    "laputa-medium": ("laputa_seq", "403369 F0D77B AE93BE 5C5992 14191F 1D2645 B4DAE5"),
    "marnie-dark1": ("marnie1_seq", "004749 73684C 2F1619 335D6B 635143 0E3B52 15110E"),
    "marnie-dark2": ("marnie2_seq", "66650B 14231C 22513D 17382F 2C5223 404D3C 0E130D"),
    "marnie-light1": ("marnie1_seq", "AF9699 80C7C9 95918E 8EBBD2 E3D1C3 B3DDEB F3E8CC"),
    "marnie-light2": ("marnie2_seq", "94A39C ACD2A3 97B8AF 8E938D A2D1BD C0CDBC E6E58B"),
    "marnie-medium1": ("marnie1_seq", "1C77A3 28231D E9D097 67B8D6 C5A387 5E2D30 008E90"),
    "marnie-medium2": ("marnie2_seq", "58A449 1D271C 2C715F CEC917 274637 819A7A 44A57C"),
    "mononoke-dark": ("mononoke_seq", "67271B 756D49 1F273E 3A160A 724615 333831 030A10"),
    "mononoke-light": ("mononoke_seq", "BA968A 838A90 E7A79B B3B8B1 9FA7BE F2C695 F5EDC9"),
    "mononoke-medium": ("mononoke_seq", "3D4F7D E48C2A 657060 CD4F38 06141F 742C14 EAD890"),
    "ponyo-dark": ("ponyo_seq", "742D33 262020 746353 14454C 6C581D 2D3740 6E3C31"),
    "ponyo-light": ("ponyo_seq", "A6A0A0 F4ADB3 94C5CC ADB7C0 ECD89D EEBCB1 F4E3D3"),
    "ponyo-medium": ("ponyo_seq", "278B9A E75B64 D8AF39 4C413F 5A6F80 DE7862 E8C4A2"),
    "spirited-dark": ("spirited_seq", "345C75 411A1B 474048 61574B 5B6B78 1A1C17 0F1217"),
    "spirited-light": ("spirited_seq", "C19A9B 8F9297 C7C0C8 9A9C97 B4DCF5 E1D7CB DBEBF8"),
    "spirited-medium": ("spirited_seq", "833437 67B9E9 1F262E 8F8093 C3AF97 353831 B7D9F2"),
    "totoro-dark": ("totoro_seq", "564029 2C1D16 213148 5C5344 282629 151412 05090A"),
    "totoro-light": ("totoro_seq", "AC9D96 85898A D6C0A9 A1B1C8 959492 A8A6A9 DCD3C4"),
    "totoro-medium": ("totoro_seq", "AD8152 2D2A25 446590 BBA78C 534C53 583B2B 0A1215"),
}

SEQ: dict[str, str] = {
    "ponyo": "F4E3D3 94C5CC 278B9A 14454C",
    "totoro": "DCD3C4 A1B1C8 446590 213148",
    "spirited": "DBEBF8 67B9E9 345C75 1F262E",
    "kiki": "E9DBD0 7EBAC2 0E84B4 06425A",
    "mononoke": "F5EDC9 CD4F38 3A160A",
    "laputa": "DAEDF3 AE93BE 5C5992 1D2645",
    "marnie1": "F3E8CC 67B8D6 008E90 004749",
    "marnie2": "E6E58B 58A449 2C5223",
}

DIV: dict[str, str] = {
    "ponyo": "14454C 278B9A 94C5CC F4F2EE F4ADB3 E75B64 742D33",
    "totoro": "213148 446590 A1B1C8 F4F2EE BBA78C 5C5344 2C1D16",
    "spirited": "345C75 67B9E9 F4F2EE C19A9B 833437",
    "kiki": "06425A 0E84B4 7EBAC2 E9DBD0 D98594 B50A2A 590514",
    "mononoke": "1F273E 3D4F7D 9FA7BE F5EDC9 E48C2A 724615 3A160A",
    "laputa": "5C5992 D7CADE F4F2EE F0D77B 776A3D",
    "marnie1": "004749 67B8D6 F4F2EE AF9699 5E2D30",
    "marnie2": "2C715F A2D1BD F4F2EE CEC917 66650B",
}

HATCH: list[str] = ["", "////", "\\\\\\\\", "xxxx", "....", "++++", "oo"]


def get_venue(name: str) -> Venue:
    try:
        return VENUES[name.lower()]
    except KeyError:
        raise ValueError(f"unknown venue {name!r}, choose from {sorted(VENUES)}") from None


def figsize(
    venue: str,
    span: Span = "col",
    frac: float = 1.0,
    ratio: float = GOLDEN,
    height: float | None = None,
) -> tuple[float, float]:
    """Return (width, height) in inches for a figure in the given venue's template.

    Args:
        venue: one of VENUES, case-insensitive ("icml", "iclr", "neurips", "acl", "emnlp",
            "colm", "tmlr", "nature").
        span: "col" for one column, "text" for the full text width. Single-column venues have
            the same width for both.
        frac: fraction of that width, e.g. 0.5 for two figures side by side.
        ratio: height / width, ignored if `height` is given.
        height: explicit height in inches.

    Nature figures are capped at the journal's maximum height.
    """
    spec = get_venue(venue)
    match span:
        case "col":
            base = spec.col if spec.col is not None else spec.text
        case "text":
            base = spec.text
        case _:
            raise ValueError(f"span must be 'col' or 'text', got {span!r}")
    width = base * frac
    fig_height = height if height is not None else width * ratio
    if spec.max_height is not None:
        fig_height = min(fig_height, spec.max_height)
    return width, fig_height


def colors_of(spec: str) -> list[str]:
    return [f"#{c}" for c in spec.split()]


def make_cmap(name: str, spec: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, [to_rgb(c) for c in colors_of(spec)], N=256)


def palette_rc(name: str) -> dict[str, Any]:
    cmap, spec = PALETTES[name]
    colors = colors_of(spec)
    n = len(colors)
    cycle = cycler(color=colors) + cycler(linestyle=LINESTYLES[:n])
    return {"axes.prop_cycle": cycle, "image.cmap": cmap}


def venue_rc(name: str) -> dict[str, Any]:
    spec = get_venue(name)
    return {
        "font.size": spec.fig_pt,
        "font.family": spec.family,
        "text.latex.preamble": spec.preamble,
        "figure.figsize": figsize(name),
        "figure.constrained_layout.use": True,
    }


def register() -> None:
    """Register the colormaps and the named styles (idempotent; `use_style` calls it).

    Colormaps: "<family>_seq" and "<family>_div" for every family in SEQ / DIV, e.g. "kiki_seq".
    Styles, usable as `plt.style.use(["paper", "icml", "kiki-medium"])`: "paper", every venue
    in VENUES and every palette in PALETTES.
    """
    for kind, data in (("seq", SEQ), ("div", DIV)):
        for family, spec in data.items():
            name = f"{family}_{kind}"
            if name not in mpl.colormaps:
                mpl.colormaps.register(make_cmap(name, spec), name=name)
    library = mpl.style.library
    library["paper"] = mpl.rc_params_from_file(STYLE_FILE, use_default_template=False)
    for name in VENUES:
        library[name] = venue_rc(name)
    for name in PALETTES:
        library[name] = palette_rc(name)
    mpl.style.available[:] = sorted(n for n in library if not n.startswith("_"))


def use_style(
    venue: str | None = None, *, palette: str = DEFAULT_PALETTE, usetex: bool = True
) -> None:
    """Apply the base style, the venue style and a palette to matplotlib's global rcParams.

    Args:
        venue: one of VENUES; sets the venue's body font (family and LaTeX preamble), the figure
            font size and the default figure size (one column, golden ratio). Without it the
            base style's sans-serif font and sizes are kept.
        palette: one of PALETTES; sets the color, linestyle and marker cycle and the image
            colormap.
        usetex: render text with LaTeX (needs latex, dvipng and ghostscript installed); the
            base style turns it on, so pass False where LaTeX is not available (the venue
            fonts then fall back to matplotlib's default serif / sans-serif).

    Layers are applied in the order base, venue, palette, then `usetex`.
    """
    if palette not in PALETTES:
        raise ValueError(f"unknown palette {palette!r}, choose from {sorted(PALETTES)}")
    layers = ["paper"]
    if venue is not None:
        get_venue(venue)
        layers.append(venue.lower())
    register()
    mpl.style.use([*layers, palette])
    mpl.rcParams["text.usetex"] = usetex


def div_cmap() -> str:
    """Return the diverging colormap that pairs with the current image colormap.

    For a palette's "<family>_seq" colormap this is "<family>_div"; any other colormap gets
    "coolwarm". Use it for signed data such as activation patching effects.
    """
    name = mpl.rcParams["image.cmap"]
    return f"{name.removesuffix('_seq')}_div" if name.endswith("_seq") else "coolwarm"


def cycle(key: str) -> list[Any]:
    """Return the values of one property in the current property cycle, e.g. "color"."""
    return mpl.rcParams["axes.prop_cycle"].by_key()[key]


def headroom(ax: Axes, frac: float) -> None:
    """Extend the upper y-limit of `ax` by `frac` of its range, e.g. for legends or labels."""
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + frac * (hi - lo))


def save(fig: Figure, name: str, *, final: bool = False, outdir: str | Path = "figs") -> Path:
    """Save `fig` as `<outdir>/<name>.pdf` if `final` (vector) else `<outdir>/<name>.png`.

    Creates `outdir` if needed and returns the written path. Resolution and cropping come from
    the `savefig.*` settings of the base style.
    """
    path = Path(outdir) / f"{name}.{'pdf' if final else 'png'}"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path
