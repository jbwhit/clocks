"""Quarto pre-render step: draw the site's clips and posters into media/."""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from clocks._site_media import render_site_media

SITE = Path(__file__).resolve().parent
render_site_media(
    SITE.parent / "assets" / "replays",
    SITE / "media",
    full=os.environ.get("QUARTO_PROJECT_RENDER_ALL") == "1",
)
