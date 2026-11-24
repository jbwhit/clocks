"""Replays the clip tests draw from, and the tables they read (not collected).

Each loader caches its result per name; callers must not change the arrays of
a cached replay, and use ``edited`` for a changed copy. Tasks 4 to 6 extend the
tables with their clips.
"""

import functools
import importlib
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from clocks._replay import Replay, load_replay, validate_replay

ROOT = Path(__file__).resolve().parents[1]


@functools.cache
def small(name: str) -> Replay:
    """Nine observations of forty particles."""
    return _record(name, n_observations=9, n_particles=40)


@functools.cache
def shortest(name: str) -> Replay:
    """One observation of sixteen particles."""
    return _record(name, n_observations=1, n_particles=16)


@functools.cache
def one_particle(name: str) -> Replay:
    """Two observations of one particle: every standard deviation is 0."""
    return _record(name, n_observations=2, n_particles=1)


@functools.cache
def committed(name: str) -> Replay:
    """The archive committed under ``assets/replays``."""
    return load_replay(ROOT / "assets" / "replays" / f"{name}.npz")


def edited(replay: Replay, **arrays: ArrayLike) -> Replay:
    """A validated copy of ``replay`` with the named arrays replaced."""
    changed = Replay(
        dict(replay.meta),
        {**replay.arrays, **{key: np.asarray(value) for key, value in arrays.items()}},
    )
    validate_replay(changed, "edited")
    return changed


def _record(name: str, *, n_observations: int, n_particles: int) -> Replay:
    module = importlib.import_module(f"clocks._demos.{name}")
    return module.record(n_observations=n_observations, n_particles=n_particles)


# demo -> (axes gid, artist gid, (x column, y column) of cloud, palette role)
DOTS = {
    "demo_1d": [("cloud", "dots", (0, 1), "belief")],
    "demo_2d": [("map", "dots", (0, 1), "belief")],
    "demo_multi_mass": [
        ("cloud", "dots-1", (0, 2), "belief"),
        ("cloud", "dots-2", (1, 3), "belief_alt"),
    ],
}
# demo -> (axes gid, artist gid, (i, j): the block of mean and cov, palette role)
ELLIPSES = {
    "demo_1d": [("cloud", "ellipse", (0, 1), "belief")],
    "demo_2d": [("map", "ellipse", (0, 1), "belief")],
    "demo_multi_mass": [
        ("cloud", "ellipse-1", (0, 2), "belief"),
        ("cloud", "ellipse-2", (1, 3), "belief_alt"),
    ],
}
# demo -> (axes gid, artist gid, (i, j): the entries of true_params)
TRUTHS = {
    "demo_1d": [("cloud", "truth", (0, 1))],
    "demo_2d": [("map", "truth", (0, 1))],
    "demo_multi_mass": [("cloud", "truth-1", (0, 2)), ("cloud", "truth-2", (1, 3))],
}
# demo -> (parent axes gid, inset gid, (i, j), y name, the parent artists it repeats);
# each copy's gid is f"{inset_gid}-{artist_gid}", both taken from the row
INSETS = {
    "demo_1d": [("cloud", "inset", (0, 1), "M", ("dots", "truth", "ellipse"))],
    "demo_2d": [("map", "inset", (0, 1), "y", ("dots", "truth", "ellipse"))],
    "demo_multi_mass": [
        ("cloud", "inset-1", (0, 2), "M₁", ("dots-1", "truth-1", "ellipse-1")),
        ("cloud", "inset-2", (1, 3), "M₂", ("dots-2", "truth-2", "ellipse-2")),
    ],
}

_TRACE_ENTRIES = [
    ("trace-mean", "line", "after the prior"),
    ("trace-band", "band", "after the prior"),
    ("trace-truth", "line", "always"),
]
# The clock strip and the slowing panel of either track clip.
_TRACK_ENTRIES = [
    ("strip", "clocks", "marks", "always"),
    ("slowing", "slowing-truth", "marks", "always"),
    ("slowing", "slowing-predicted", "marks", "always"),
    ("slowing", "slowing-predicted-bar", "segments", "always"),
    ("slowing", "slowing-observed", "marks", "after the prior"),
    ("slowing", "slowing-observed-bar", "segments", "after the prior"),
]


def _cloud_entries(suffix: str, inset: str) -> list[tuple[str, str, str, str]]:
    """One mass's cloud, truth and ellipse in "cloud", and their inset copies."""
    kinds = [("dots", "marks"), ("truth", "marks"), ("ellipse", "line")]
    return [("cloud", f"{name}{suffix}", kind, "always") for name, kind in kinds] + [
        (inset, f"{inset}-{name}{suffix}", kind, "always") for name, kind in kinds
    ]


def _traces(*params: str) -> list[tuple[str, str, str, str]]:
    return [
        (f"trace-{param}", artist, kind, when)
        for param in params
        for artist, kind, when in _TRACE_ENTRIES
    ]


# demo -> (axes gid or None for a figure text, artist gid, kind of assert_shown,
# when it shows: "always", "after the prior" or "only the prior")
ARTISTS: dict[str, list[tuple[str | None, str, str, str]]] = {
    "demo_2d": [
        ("map", "dots", "marks", "always"),
        ("map", "clocks", "scatter", "always"),
        ("map", "truth", "marks", "always"),
        ("map", "ellipse", "line", "always"),
        ("inset", "inset-dots", "marks", "always"),
        ("inset", "inset-truth", "marks", "always"),
        ("inset", "inset-ellipse", "line", "always"),
        *[
            (axes, artist, kind, when)
            for axes in ("trace-x", "trace-y", "trace-M")
            for artist, kind, when in _TRACE_ENTRIES
        ],
        (None, "status", "text", "always"),
    ],
    "demo_1d": [
        *_cloud_entries("", "inset"),
        *_TRACK_ENTRIES,
        *_traces("x", "M"),
        (None, "status", "text", "always"),
    ],
    "demo_multi_mass": [
        *_cloud_entries("-1", "inset-1"),
        *_cloud_entries("-2", "inset-2"),
        *_TRACK_ENTRIES,
        *_traces("x₁", "x₂", "M₁", "M₂"),
        (None, "status", "text", "always"),
    ],
}
