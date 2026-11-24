"""Helpers that inspect the artists a figure function drew (not collected)."""

from collections import defaultdict
from dataclasses import fields
from pathlib import Path

import numpy as np
from matplotlib.artist import Artist
from matplotlib.colors import to_hex, to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from clocks._style import Palette

_COLOUR_GETTERS = (
    "get_color",
    "get_facecolor",
    "get_markerfacecolor",
    "get_markeredgecolor",
)


def palette_hexes(colors: Palette) -> set[str]:
    """Every colour of a palette, ``ordered`` included, as lower-case ``#rrggbb``."""
    hexes: set[str] = set()
    for field in fields(colors):
        value = getattr(colors, field.name)
        for text in value if isinstance(value, tuple) else (value,):
            hexes.add(to_hex(text, keep_alpha=False).lower())
    return hexes


def visible_colours(fig: Figure) -> set[str]:
    """The colours of every visible artist that paints something, as ``#rrggbb``.

    Edges are read only when the artist's line width is above 0: the figure
    patch carries a white edge of width 0 that is never drawn.
    """
    found: set[str] = set()
    for artist in fig.findobj():
        if not artist.get_visible():
            continue
        getters = list(_COLOUR_GETTERS)
        width = _line_width(artist)
        if width is not None and width > 0:
            getters.append("get_edgecolor")
        for name in getters:
            getter = getattr(artist, name, None)
            if getter is None:
                continue
            found.update(_opaque_hexes(getter()))
    return found


def one(container: Artist, gid: str) -> Artist:
    """The single artist under ``container`` that carries ``gid``."""
    matches = [a for a in container.findobj() if a.get_gid() == gid]
    assert len(matches) == 1, (
        f"expected one artist with gid {gid!r}, got {len(matches)}"
    )
    return matches[0]


def gid_census(fig: Figure) -> dict[str, list[str]]:
    """Each artist gid in the figure, mapped to the gids of the axes holding it.

    The axes' own gids are left out. One entry per artist, in ``fig.axes`` order.
    """
    census: dict[str, list[str]] = defaultdict(list)
    for ax in fig.axes:
        for artist in ax.findobj():
            gid = artist.get_gid()
            if gid is None or artist is ax:
                continue
            census[gid].append(ax.get_gid())
    return dict(census)


def band_edges(collection: Artist, x: NDArray) -> tuple[NDArray, NDArray]:
    """For each ``x``, the smallest and largest y among the band's path vertices."""
    vertices = np.concatenate([path.vertices for path in collection.get_paths()])
    lower, upper = [], []
    for value in x:
        ys = vertices[np.isclose(vertices[:, 0], value, rtol=0, atol=1e-12), 1]
        assert ys.size, f"the band has no vertex at x = {value}"
        lower.append(ys.min())
        upper.append(ys.max())
    return np.array(lower), np.array(upper)


def png_size(path: Path) -> tuple[int, int]:
    """Width and height of a PNG, from the IHDR bytes 16 to 24."""
    head = path.read_bytes()[16:24]
    return int.from_bytes(head[:4], "big"), int.from_bytes(head[4:], "big")


def _line_width(artist: Artist) -> float | None:
    for name in ("get_linewidth", "get_linewidths"):
        getter = getattr(artist, name, None)
        if getter is not None:
            width = np.atleast_1d(getter())
            return float(width.max()) if width.size else 0.0
    return None


def _opaque_hexes(value: object) -> set[str]:
    colours = np.atleast_2d(np.asarray(_as_rgba(value), dtype=float))
    return {
        to_hex(tuple(row), keep_alpha=False).lower() for row in colours if row[3] > 0
    }


def _as_rgba(value: object) -> object:
    if isinstance(value, str):
        return to_rgba(value)
    array = np.asarray(value)
    if array.dtype.kind in "US":
        return [to_rgba(str(v)) for v in array.ravel()]
    return array


def assert_drawn(line: Line2D) -> None:
    """Assert that a line is visible, opaque and has a width above 0."""
    assert line.get_visible()
    assert line.get_alpha() in (None, 1)
    assert line.get_linewidth() > 0
    if line.get_marker() not in ("None", "", None):
        assert line.get_markersize() > 0


def assert_marks_drawn(line: Line2D) -> None:
    """Assert that a marker-only line shows: visible, faint but not clear, sized."""
    assert line.get_visible()
    assert line.get_alpha() is not None and 0 < line.get_alpha() < 1
    assert line.get_marker() not in ("None", "", None)
    assert line.get_markersize() > 0
