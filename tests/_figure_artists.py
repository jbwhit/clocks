"""Helpers that inspect the artists a figure function drew (not collected)."""

from collections import defaultdict
from dataclasses import fields
from pathlib import Path

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import Collection
from matplotlib.colors import to_hex, to_rgba, to_rgba_array
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Patch
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


def _in_visible_axes(artist: Artist) -> bool:
    owner = getattr(artist, "axes", None)
    return owner is None or owner.get_visible()


def one(container: Artist, gid: str) -> Artist:
    """The single artist under ``container`` that carries ``gid``.

    An artist whose axes is hidden is not drawn, so it does not count.
    """
    matches = [
        a for a in container.findobj() if a.get_gid() == gid and _in_visible_axes(a)
    ]
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
            if gid is None or artist is ax or not _in_visible_axes(artist):
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


def all_axes(fig: Figure) -> list[Axes]:
    """Each axes of ``fig.axes`` followed by its child axes (insets)."""
    found: list[Axes] = []
    for ax in fig.axes:
        found.append(ax)
        found.extend(ax.child_axes)
    return found


def off_palette(fig: Figure, colors: Palette) -> set[str]:
    """Visible colours that are neither the palette's nor a background-ink blend.

    Reads ``visible_colours`` and, for every 3D axes, what ``fig.findobj()``
    never reaches: each axis's pane face, its pane edge when drawn, its axis
    line and its grid lines. Draw the figure first: a 3D axes sets its grid
    colours at draw time.
    """
    found = set(visible_colours(fig))
    for ax in all_axes(fig):
        if ax.name != "3d":
            continue
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane = axis.pane
            found.update(_opaque_hexes(pane.get_facecolor()))
            if pane.get_linewidth() > 0:
                found.update(_opaque_hexes(pane.get_edgecolor()))
            found.update(_opaque_hexes(axis.line.get_color()))
            found.update(_opaque_hexes(axis.gridlines.get_color()))
    allowed = palette_hexes(colors)
    return {
        hexes
        for hexes in found
        if hexes not in allowed and not _near_blend(hexes, colors)
    }


def _near_blend(colour: str, colors: Palette) -> bool:
    """Whether a colour lies within 1.5/255 per channel of a background-ink blend.

    Each channel bounds t to an interval; the colour is near the segment when
    the intervals and [0, 1] share a point.
    """
    tolerance = 1.5 / 255
    rgb = np.array(to_rgba(colour)[:3])
    start = np.array(to_rgba(colors.background)[:3])
    span = np.array(to_rgba(colors.ink)[:3]) - start
    low, high = 0.0, 1.0
    for value, origin, step in zip(rgb, start, span, strict=True):
        if step == 0:
            if abs(value - origin) > tolerance:
                return False
            continue
        first = (value - tolerance - origin) / step
        second = (value + tolerance - origin) / step
        low, high = max(low, min(first, second)), min(high, max(first, second))
    return low <= high


def opaque_fills_above(ax: Axes, artist: Artist) -> list[Artist]:
    """Visible artists drawn on ``ax`` after ``artist`` that paint an opaque fill.

    After means a higher ``zorder``, or an equal one and a later place in
    ``ax.get_children()`` (Matplotlib draws equal ``zorder`` in that order). The
    axes' background patch and its child axes are not counted. For 2D axes
    only; it does not ask whether the fill overlaps ``artist`` (decision D46).
    """
    children = ax.get_children()
    place = next(i for i, child in enumerate(children) if child is artist)
    level = artist.get_zorder()
    found = []
    for i, other in enumerate(children):
        if other is artist or other is ax.patch or other in ax.child_axes:
            continue
        if not other.get_visible():
            continue
        later = other.get_zorder() > level or (
            other.get_zorder() == level and i > place
        )
        if later and _paints_opaque_fill(other):
            found.append(other)
    return found


def _paints_opaque_fill(artist: Artist) -> bool:
    if isinstance(artist, (Patch, Collection)):
        faces = to_rgba_array(artist.get_facecolor())
        return bool(np.any(faces[:, 3] == 1))
    if isinstance(artist, Line2D):
        face = artist.get_markerfacecolor()
        return (
            MarkerStyle(artist.get_marker()).is_filled()
            and not (isinstance(face, str) and face == "none")
            and to_rgba(face)[3] == 1
            and artist.get_alpha() in (None, 1)
        )
    return False


def assert_shown(artist: Artist, kind: str) -> None:
    """Assert that ``artist`` puts ink on the canvas, by ``kind``.

    Common to every kind: the artist and each ancestor up to the figure are
    visible, and its alpha is ``None`` or above 0. Then ``"marks"`` (a
    marker-only line), ``"line"``, ``"band"``, ``"segments"`` (a line
    collection, drawn first), ``"scatter"`` or ``"text"`` each need something
    to draw and a stroke or a fill to draw it with, taken from one and the same
    item where an artist holds several.
    """
    for owner in [artist, *_ancestors(artist)]:
        assert owner.get_visible(), f"{_name(artist)}: {_name(owner)} is hidden"
    alpha = artist.get_alpha()
    assert alpha is None or alpha > 0, f"{_name(artist)}: alpha {alpha}"
    checks = {
        "marks": _check_marks,
        "line": _check_line,
        "band": _check_band,
        "segments": _check_segments,
        "scatter": _check_scatter,
        "text": _check_text,
    }
    assert kind in checks, f"unknown kind {kind!r}"
    checks[kind](artist)


def assert_hidden(artist: Artist) -> None:
    """Assert that ``artist`` or one of its ancestors is not visible."""
    owners = [artist, *_ancestors(artist)]
    assert not all(owner.get_visible() for owner in owners), (
        f"{_name(artist)} is visible"
    )


def _ancestors(artist: Artist) -> list[Artist]:
    """The axes holding ``artist``, their parent axes, and the figure."""
    chain: list[Artist] = []
    current = artist
    while True:
        owner = getattr(current, "axes", None)
        if owner is None or owner is current:
            break
        chain.append(owner)
        current = owner
    figure = artist.get_figure()
    if figure is not None:
        chain.append(figure)
    return chain


def _name(artist: Artist) -> str:
    return f"{type(artist).__name__} {artist.get_gid()!r}"


def _point_count(line: Line2D) -> int:
    if hasattr(line, "get_data_3d"):
        return len(np.atleast_1d(line.get_data_3d()[0]))
    return len(np.atleast_1d(line.get_xdata()))


def _visible_alpha(colour: object) -> bool:
    if isinstance(colour, str) and colour.lower() == "none":
        return False
    return to_rgba(colour)[3] > 0


def _has_marker(line: Line2D) -> bool:
    return line.get_marker() not in (None, "None", "none", "", " ")


def _check_marks(line: Line2D) -> None:
    name = _name(line)
    assert _point_count(line) >= 1, f"{name}: no data point"
    assert _has_marker(line), f"{name}: no marker"
    assert line.get_markersize() > 0, f"{name}: marker size 0"
    stroke = line.get_markeredgewidth() > 0 and _visible_alpha(
        line.get_markeredgecolor()
    )
    fill = MarkerStyle(line.get_marker()).is_filled() and _visible_alpha(
        line.get_markerfacecolor()
    )
    assert stroke or fill, f"{name}: the marker has neither a stroke nor a fill"


def _check_line(line: Line2D) -> None:
    name = _name(line)
    assert _point_count(line) >= 2, f"{name}: fewer than two points"
    assert line.get_linestyle() not in ("None", "none", "", " "), f"{name}: no style"
    assert line.get_linewidth() > 0, f"{name}: line width 0"
    assert _visible_alpha(line.get_color()), f"{name}: transparent stroke"


def _check_band(band: Collection) -> None:
    faces = to_rgba_array(band.get_facecolor())
    assert any(
        len(path.vertices) >= 3 and len(faces) and faces[i % len(faces), 3] > 0
        for i, path in enumerate(band.get_paths())
    ), f"{_name(band)}: no path of three vertices with a visible fill"


def _check_segments(collection: Collection) -> None:
    widths = np.atleast_1d(collection.get_linewidths())
    edges = to_rgba_array(collection.get_edgecolors())

    def draws(i: int, segment: NDArray) -> bool:
        segment = np.asarray(segment)
        return (
            len(segment) >= 2
            and not np.array_equal(segment[0], segment[-1])
            and len(widths) > 0
            and widths[i % len(widths)] > 0
            and len(edges) > 0
            and edges[i % len(edges), 3] > 0
        )

    assert any(draws(i, s) for i, s in enumerate(collection.get_segments())), (
        f"{_name(collection)}: no segment with length and a visible stroke"
    )


def _check_scatter(collection: Collection) -> None:
    offsets = collection.get_offsets()
    sizes = np.atleast_1d(collection.get_sizes())
    widths = np.atleast_1d(collection.get_linewidths())
    edges = to_rgba_array(collection.get_edgecolors())
    faces = to_rgba_array(collection.get_facecolors())

    def entry(values: NDArray, i: int) -> object:
        return values[i % len(values)] if len(values) else None

    def draws(i: int) -> bool:
        size, width = entry(sizes, i), entry(widths, i)
        edge, face = entry(edges, i), entry(faces, i)
        stroke = width is not None and width > 0 and edge is not None and edge[3] > 0
        fill = face is not None and face[3] > 0
        return size is not None and size > 0 and (stroke or fill)

    assert any(draws(i) for i in range(len(offsets))), (
        f"{_name(collection)}: no item with an offset, a size and visible ink"
    )


def _check_text(text: Artist) -> None:
    name = _name(text)
    assert text.get_text(), f"{name}: empty text"
    assert _visible_alpha(text.get_color()), f"{name}: transparent text"
