"""Shared figure style: a fixed palette, a theme context and a display helper.

Each colour role has one value per theme, so a plot reads the same wherever it
appears; contrast against the page is checked in ``tests/test_style.py``.
``style`` applies a theme's rcParams and registers the bundled Source Sans 3
font; ``show`` displays a finished figure at its authored size.
"""

import io
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cycler, font_manager
from matplotlib.figure import Figure

THEMES: tuple[str, str] = ("light", "dark")


@dataclass(frozen=True)
class Palette:
    """One theme's colours, as ``#rrggbb`` strings."""

    background: str  # the page colour
    ink: str  # instruments, observed data, neutral curves; primary text
    muted: str  # axis labels, tick labels, secondary text
    rule: str  # axis lines
    grid: str  # grid lines
    truth: str  # the true configuration, or what it predicts
    belief: str  # an estimate or a candidate: the filter's posterior
    belief_alt: str  # the second of two candidates
    threshold: str  # noise levels, targets, limits; always dashed
    ordered: tuple[str, str, str]  # an ordered family, least to most contrast


_LIGHT = Palette(
    background="#f9f5d7",
    ink="#282828",
    muted="#665c54",
    rule="#d5c4a1",
    grid="#ebe2bd",
    truth="#bf4a0a",
    belief="#1a6fa3",
    belief_alt="#a0306e",
    threshold="#665c54",
    ordered=("#3f86bd", "#1a6fa3", "#0d456b"),
)

_DARK = Palette(
    background="#1d2021",
    ink="#fbf1c7",
    muted="#bdae93",
    rule="#504945",
    grid="#2b2f30",
    truth="#e0701a",
    belief="#4595cf",
    belief_alt="#d05c9c",
    threshold="#bdae93",
    ordered=("#2f74aa", "#4595cf", "#8fc1e8"),
)

_PALETTES = {"light": _LIGHT, "dark": _DARK}


def palette(theme: str) -> Palette:
    """Return the palette for ``theme``, which must be one of ``THEMES``."""
    if theme not in _PALETTES:
        raise ValueError(f"unknown theme {theme!r}; expected one of {THEMES}")
    return _PALETTES[theme]


_FONT_DIR = Path(__file__).parent / "_fonts"
_FONT_FILES = ("SourceSans3-Regular.ttf", "SourceSans3-Semibold.ttf")
_fonts_registered = False

_BASE_RC = {
    "font.family": "Source Sans 3",
    "font.size": 10.5,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.linewidth": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.8,
    "lines.solid_capstyle": "round",
    "figure.figsize": (7.5, 4.2),
    "figure.dpi": 100,
    "figure.autolayout": False,  # it would take priority over constrained
    "figure.constrained_layout.use": True,
}


@contextmanager
def style(theme: str) -> Iterator[Palette]:
    """Apply ``theme``'s rcParams for the block and yield its palette."""
    colors = palette(theme)
    _register_fonts()
    rc = {
        **_BASE_RC,
        "figure.facecolor": colors.background,
        "axes.facecolor": colors.background,
        "savefig.facecolor": colors.background,
        "text.color": colors.ink,
        "axes.labelcolor": colors.muted,
        "xtick.color": colors.muted,
        "ytick.color": colors.muted,
        "axes.edgecolor": colors.rule,
        "grid.color": colors.grid,
        "axes.prop_cycle": cycler(color=[colors.ink]),
    }
    with matplotlib.rc_context(rc):
        yield colors


def show(fig: Figure) -> None:
    """Display ``fig`` at its authored size, sharp on retina screens, then close it.

    The PNG is rendered at twice the figure's dpi and never cropped, so a
    7.5 by 4.2 inch figure is declared 750 CSS pixels wide and 420 tall in the page.
    """
    try:
        from IPython.display import Image, display
    except ImportError as error:
        raise ImportError(
            "clocks.viz.show needs IPython; it is meant for Jupyter and Quarto cells"
        ) from error
    # Interactive HiDPI backends scale fig.dpi by the device pixel ratio.
    dpi = 2 * fig.dpi / fig.canvas.device_pixel_ratio
    buffer = io.BytesIO()
    try:
        with matplotlib.rc_context({"savefig.bbox": None}):
            fig.savefig(
                buffer,
                format="png",
                dpi=dpi,
                bbox_inches=None,
                transparent=False,
                facecolor=fig.get_facecolor(),
            )
    finally:
        plt.close(fig)
    display(Image(data=buffer.getvalue(), retina=True))


def _register_fonts() -> None:
    """Add the bundled fonts to matplotlib once, all or none."""
    global _fonts_registered
    if _fonts_registered:
        return
    paths = [_FONT_DIR / name for name in _FONT_FILES]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"bundled font missing: {path}")
    # addfont appends to ttflist at once, so a failure part-way through would
    # leave the earlier files registered; roll the list back before re-raising.
    ttflist = font_manager.fontManager.ttflist
    before = len(ttflist)
    try:
        for path in paths:
            font_manager.fontManager.addfont(str(path))
    except BaseException:
        del ttflist[before:]
        raise
    _fonts_registered = True
