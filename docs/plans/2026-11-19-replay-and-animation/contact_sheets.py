"""Draw contact sheets of the six clips and the two figures, both themes.

A figure's sheet has the light and dark renderings at full width in the top row
and the same two resampled to 780 pixels wide (390 CSS pixels at twice density,
what a phone shows) in the bottom row. A clip's sheet is drawn by the library's
own clip builder from the committed archive, at rows 0, 1, 8 and the last, each
at its shot's camera azimuth, and each row is shown the same way: both themes at
full width above, both at 780 pixels below. Run from anywhere:

    uv run python docs/plans/2026-11-19-replay-and-animation/contact_sheets.py \
        --out docs/plans/2026-11-19-replay-and-animation
"""

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from clocks._clips import BUILDERS, build_clip, shots
from clocks._style import save_png
from clocks.viz import THEMES, density_figure, load_replay, range_study_figure, style

ROOT = Path(__file__).resolve().parents[3]
PHONE_WIDTH = 780
MARGIN = 24
LABEL_HEIGHT = 36
BACKGROUND = (255, 255, 255)


def _render(name: str, theme: str, directory: Path) -> Image.Image:
    with style(theme) as colors:
        if name == "density":
            replay = load_replay(ROOT / "assets/replays/demo_density.npz")
            fig = density_figure(replay, colors)
        else:
            text = (ROOT / "assets/echolocation_range_study.json").read_text()
            fig = range_study_figure(json.loads(text), colors)
        path = directory / f"{name}-{theme}.png"
        save_png(fig, path)
    return _read(path)


def _render_clip(demo: str, theme: str, row: int, directory: Path) -> Image.Image:
    """The clip's drawing of ``row`` in ``theme``, at that row's first shot."""
    replay = load_replay(ROOT / "assets/replays" / f"{demo}.npz")
    azimuth = next(shot.azimuth for shot in shots(replay) if shot.row == row)
    with style(theme) as colors:
        clip = build_clip(replay, colors)
        try:
            clip.show(row, azimuth)
            path = directory / f"{demo}-{theme}-{row}.png"
            save_png(clip.fig, path)
        finally:
            plt.close(clip.fig)
    return _read(path)


def _read(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _clip_rows(demo: str) -> list[int]:
    replay = load_replay(ROOT / "assets/replays" / f"{demo}.npz")
    return [0, 1, 8, int(replay.meta["n_observations"])]


def _sheet(images: dict[str, Image.Image]) -> Image.Image:
    full = [(theme, images[theme]) for theme in THEMES]
    phone = [
        (
            theme,
            image.resize(
                (PHONE_WIDTH, round(image.height * PHONE_WIDTH / image.width)),
                Image.LANCZOS,
            ),
        )
        for theme, image in full
    ]
    rows = [("full width", full), ("390 px at 2x", phone)]
    width = MARGIN + sum(image.width + MARGIN for _, image in full)
    height = MARGIN + sum(
        LABEL_HEIGHT + max(image.height for _, image in row) + MARGIN for _, row in rows
    )
    sheet = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=22)
    y = MARGIN
    for width_label, row in rows:
        x = MARGIN
        for theme, image in row:
            draw.text(
                (x, y), f"{theme}, {width_label} ({image.width} px)", "black", font
            )
            sheet.paste(image, (x, y + LABEL_HEIGHT))
            x += image.width + MARGIN
        y += LABEL_HEIGHT + max(image.height for _, image in row) + MARGIN
    return sheet


def _clip_sheet(
    images: dict[tuple[int, str], Image.Image], rows: list[int]
) -> Image.Image:
    """Task 1's layout: per row, both themes at full width, then both at 780.

    Each theme keeps its column, so a phone-width image sits under its own
    full-width one.
    """
    column = max(image.width for image in images.values())
    bands = []
    for row in rows:
        full = [images[row, theme] for theme in THEMES]
        bands.append((row, full, [_phone(image) for image in full]))
    width = MARGIN + len(THEMES) * (column + MARGIN)
    height = MARGIN + sum(
        2 * LABEL_HEIGHT + full[0].height + phone[0].height + 3 * MARGIN
        for _, full, phone in bands
    )
    sheet = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=22)
    y = MARGIN
    for row, full, phone in bands:
        for label, band in (("full width", full), ("390 px at 2x", phone)):
            x = MARGIN
            for theme, image in zip(THEMES, band, strict=True):
                text = f"row {row}, {theme}, {label} ({image.width} px)"
                draw.text((x, y), text, "black", font)
                sheet.paste(image, (x, y + LABEL_HEIGHT))
                x += column + MARGIN
            y += LABEL_HEIGHT + band[0].height + MARGIN
        y += MARGIN
    return sheet


def _phone(image: Image.Image) -> Image.Image:
    """``image`` resampled to the phone width, 390 CSS pixels at twice density."""
    return image.resize(
        (PHONE_WIDTH, round(image.height * PHONE_WIDTH / image.width)), Image.LANCZOS
    )


def _report(target: Path) -> None:
    try:
        print(target.resolve().relative_to(ROOT))
    except ValueError:
        print(target.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    names = {"density": "density.jpg", "range-study": "range-study.jpg"}
    with tempfile.TemporaryDirectory() as scratch:
        directory = Path(scratch)
        for name, filename in names.items():
            images = {theme: _render(name, theme, directory) for theme in THEMES}
            target = args.out / filename
            _sheet(images).save(target, quality=85)
            _report(target)
        for demo in BUILDERS:
            rows = _clip_rows(demo)
            images = {
                (row, theme): _render_clip(demo, theme, row, directory)
                for row in rows
                for theme in THEMES
            }
            target = args.out / f"clip-{demo}.jpg"
            _clip_sheet(images, rows).save(target, quality=85)
            _report(target)


if __name__ == "__main__":
    main()
