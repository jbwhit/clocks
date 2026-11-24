"""Video clips drawn from replays: the registry, the shot list and ``render_clip``.

Each clip is a figure built once from a replay, with fixed axes, and redrawn
for every row; each drawing is written to the encoder as many times as its
dwell asks. The clips are read from the archive alone: no module here calls
the forward model.
"""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from clocks._clip_echo import azimuth, build_echo
from clocks._clip_maps import build_2d, build_multi_2d
from clocks._clip_models import build_models
from clocks._clip_parts import Clip
from clocks._clip_tracks import build_1d, build_multi_1d
from clocks._replay import Replay
from clocks._style import Palette, frame_rgba, style
from clocks._video import dwell_frames, encode, find_encoder

FRAME_WIDTH = 1500  # pixels: 750 CSS pixels at twice the style's 100 dpi

# Declared heights in CSS pixels; a frame is twice as tall.
CLIP_HEIGHTS: dict[str, int] = {
    "demo_1d": 450,
    "demo_2d": 450,
    "demo_multi_mass": 500,
    "demo_multi_mass_2d": 550,
    "demo_model_comparison": 420,
    "demo_echolocation_3d": 550,
}
BUILDERS: dict[str, Callable[[Replay, Palette], Clip]] = {
    "demo_1d": build_1d,
    "demo_2d": build_2d,
    "demo_multi_mass": build_multi_1d,
    "demo_multi_mass_2d": build_multi_2d,
    "demo_model_comparison": build_models,
    "demo_echolocation_3d": build_echo,
}


class Shot(NamedTuple):
    """One drawing of a clip: the archive row, its repeats, the camera azimuth."""

    row: int
    repeat: int
    azimuth: float | None = None


def shots(replay: Replay) -> list[Shot]:
    """Every drawing of the clip, in order, each held for its dwell.

    The echolocation clip's camera turns through the F video frames before the
    final hold, so each of those frames is a drawing of its own, at its own
    azimuth; the last row is held at the azimuth of frame F - 1.
    """
    dwell = dwell_frames(int(replay.meta["n_observations"]))
    if replay.meta["demo"] != "demo_echolocation_3d":
        return [Shot(k, n) for k, n in enumerate(dwell)]
    turning = sum(dwell[:-1])
    rows = [k for k, n in enumerate(dwell[:-1]) for _ in range(n)]
    return [Shot(k, 1, azimuth(f, turning)) for f, k in enumerate(rows)] + [
        Shot(len(dwell) - 1, dwell[-1], azimuth(turning - 1, turning))
    ]


def build_clip(replay: Replay, colors: Palette) -> Clip:
    """Build the clip of ``replay``'s demo. Call inside ``style(theme)``."""
    demo = replay.meta["demo"]
    if demo not in BUILDERS:
        raise ValueError(f"no clip for demo {demo!r}")
    return BUILDERS[demo](replay, colors)


def final_frame(replay: Replay, colors: Palette) -> Figure:
    """The clip's figure at its last shot, the poster; the caller closes it.

    Call inside ``style(theme)``.
    """
    clip = build_clip(replay, colors)
    last = shots(replay)[-1]
    try:
        clip.show(last.row, last.azimuth)
    except BaseException:
        plt.close(clip.fig)
        raise
    return clip.fig


def render_clip(
    replay: Replay, path: Path, *, theme: str = "light", encoder: str | None = None
) -> int:
    """Draw ``replay``'s clip in ``theme`` and encode it to ``path`` as H.264.

    The frames are 1500 pixels wide and twice the clip's declared height tall,
    at 20 a second; each row is drawn only when the encoder asks for it. The
    encoder is looked up before anything is drawn. ``path`` is written by
    ``clocks._video.encode`` alone, whole or not at all. Returns the number of
    video frames written.

    A model-comparison replay may hold at most three candidates, one per
    colour of the palette's ``ordered``; more raise ``ValueError``.
    """
    if encoder is None:
        encoder = find_encoder()
    with style(theme) as colors:
        clip = build_clip(replay, colors)
        try:
            return encode(
                _frames(clip, shots(replay)),
                path,
                width=FRAME_WIDTH,
                height=2 * CLIP_HEIGHTS[replay.meta["demo"]],
                encoder=encoder,
            )
        finally:
            plt.close(clip.fig)


def _frames(clip: Clip, plan: list[Shot]) -> Iterator[tuple[bytes, int]]:
    """Each shot's pixels and repeats, drawn only when the next is asked for."""
    for shot in plan:
        clip.show(shot.row, shot.azimuth)
        yield frame_rgba(clip.fig), shot.repeat
