"""Figures, clips and the shared style (public facade).

Implementations live in ``clocks._figures`` (figures drawn from replays),
``clocks._echo_study`` (the range-study figure), ``clocks._clips`` (clips drawn
from replays), ``clocks._replay`` (the archive loader) and ``clocks._style``.
"""

from clocks._clips import render_clip
from clocks._echo_study import range_study_figure
from clocks._figures import density_figure
from clocks._replay import load_replay
from clocks._style import THEMES, Palette, palette, show, style

__all__ = [
    "THEMES",
    "Palette",
    "density_figure",
    "load_replay",
    "palette",
    "range_study_figure",
    "render_clip",
    "show",
    "style",
]
