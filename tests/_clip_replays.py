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
DOTS = {"demo_2d": [("map", "dots", (0, 1), "belief")]}
# demo -> (axes gid, artist gid, (i, j): the block of mean and cov, palette role)
ELLIPSES = {"demo_2d": [("map", "ellipse", (0, 1), "belief")]}
# demo -> (axes gid, artist gid, (i, j): the entries of true_params)
TRUTHS = {"demo_2d": [("map", "truth", (0, 1))]}
# demo -> (parent axes gid, inset gid, (i, j), y name, the parent artists it repeats);
# each copy's gid is f"{inset_gid}-{artist_gid}", both taken from the row
INSETS = {"demo_2d": [("map", "inset", (0, 1), "y", ("dots", "truth", "ellipse"))]}

# demo -> (axes gid or None for a figure text, artist gid, kind of assert_shown,
# when it shows: "always", "after the prior" or "only the prior")
_TRACE_ENTRIES = [
    ("trace-mean", "line", "after the prior"),
    ("trace-band", "band", "after the prior"),
    ("trace-truth", "line", "always"),
]
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
    ]
}
