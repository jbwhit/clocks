"""Fixed colour roles shared by every figure on the site.

Each role has one value per theme, so a plot reads the same wherever it
appears. Contrast against the page is checked in ``tests/test_style.py``.
"""

from dataclasses import dataclass

THEMES: tuple[str, str] = ("light", "dark")


@dataclass(frozen=True)
class Palette:
    """One theme's colours, as ``#rrggbb`` strings."""

    background: str
    ink: str
    muted: str
    rule: str
    grid: str
    truth: str
    belief: str
    belief_alt: str
    threshold: str
    ordered: tuple[str, str, str]


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
