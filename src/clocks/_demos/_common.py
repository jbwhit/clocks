"""Shared command-line contracts for packaged demos."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from clocks._clips import BUILDERS, render_clip
from clocks._replay import Replay, save_replay
from clocks._style import Palette, save_png, style
from clocks._video import EncoderError, probe

STAGED = ".staged-"


def positive_int(value: str) -> int:
    """Parse a strictly positive integer for a command-line option."""
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected an integer, got {value!r}"
        ) from error
    if result <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return result


def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    output: Path,
    observations: int,
    particles: int,
) -> None:
    """Add the common reproducible-output and fast-smoke overrides."""
    parser.add_argument(
        "--output",
        type=Path,
        default=output,
        help=f"output image or animation (default: {output})",
    )
    parser.add_argument(
        "--observations",
        type=positive_int,
        default=observations,
        help=f"number of simulated observations (default: {observations})",
    )
    parser.add_argument(
        "--particles",
        type=positive_int,
        default=particles,
        help=f"number of SMC particles (default: {particles})",
    )


def write_outputs(
    replay: Replay,
    directory: Path,
    figure: Callable[[Replay, Palette], Figure],
    *,
    theme: str = "light",
    encoder: str | None = None,
) -> list[Path]:
    """Write ``replay``'s archive and PNG into ``directory``, and its clip when
    ``encoder`` is given; return the paths written, archive, PNG, clip.

    Every output is staged under its final name with ``STAGED`` in front, and
    none is moved into place until all are staged and the clip decodes to every
    frame sent (decision C6 of the commands plan). Nothing is printed until
    every rename is made. For a clip demo run without ``encoder``, an entry at
    ``<demo>.mp4`` (a link, dangling or not, included) is kept, and a line says
    so; the density demo, which has no clip, says nothing of one.
    """
    demo = replay.meta["demo"]
    directory.mkdir(parents=True, exist_ok=True)
    finals = [directory / f"{demo}.npz", directory / f"{demo}.png"]
    if encoder is not None:
        finals.append(directory / f"{demo}.mp4")
    for final in finals:
        # is_symlink first: is_dir follows a link, and on Python 3.12 can raise
        # PermissionError for a link into a directory the caller cannot enter.
        if not final.is_symlink() and final.is_dir():
            raise IsADirectoryError(
                f"{final} is a directory: move it, or choose another --output-dir"
            )
    staged = [final.with_name(STAGED + final.name) for final in finals]
    try:
        for path in staged:
            path.unlink(missing_ok=True)  # left by an interrupted run
        save_replay(staged[0], replay)
        _stage_png(replay, staged[1], figure, theme)
        if encoder is not None:
            sent = render_clip(replay, staged[2], theme=theme, encoder=encoder)
            _check_clip(staged[2], sent, encoder)
        for path, final in zip(staged, finals, strict=True):
            os.replace(path, final)
    except BaseException as error:
        _remove(staged, error)
        raise
    for final in finals:
        print(f"Saved: {final}")
    clip = directory / f"{demo}.mp4"
    has_clip = demo in BUILDERS  # demo_density has no clip and no --video
    if has_clip and encoder is None and (clip.is_symlink() or clip.exists()):
        print(
            f"Kept: {clip} is from an earlier run and was not updated "
            "(--video writes a new one)"
        )
    return finals


def _stage_png(
    replay: Replay, path: Path, figure: Callable[[Replay, Palette], Figure], theme: str
) -> None:
    """Draw the figure inside ``style(theme)`` into a file created at ``path``."""
    with style(theme) as colors:
        fig = figure(replay, colors)
        try:
            with path.open("xb") as file:
                save_png(fig, file)
        finally:
            plt.close(fig)


def _check_clip(path: Path, sent: int, encoder: str) -> None:
    """Refuse, with ``EncoderError``, a staged clip that does not decode to all
    ``sent`` frames (``probe`` runs the same encoder binary as a decoder)."""
    try:
        decoded = probe(path, encoder).frames
    except ValueError as error:
        raise EncoderError(f"{path} does not decode: {error}") from error
    if decoded != sent:
        raise EncoderError(f"{path} decodes to {decoded} of the {sent} frames sent")


def _remove(paths: list[Path], error: BaseException) -> None:
    """Remove ``paths``; a removal that fails is a note on ``error``."""
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as cleanup_error:
            error.add_note(f"{path} was left behind: {cleanup_error}")
