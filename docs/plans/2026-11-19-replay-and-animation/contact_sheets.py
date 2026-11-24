"""Draw contact sheets of the density and range-study figures, both themes.

Each sheet has the light and dark renderings at full width in the top row and
the same two resampled to 780 pixels wide (390 CSS pixels at twice density, what
a phone shows) in the bottom row. Run from anywhere:

    uv run python docs/plans/2026-11-19-replay-and-animation/contact_sheets.py \
        --out docs/plans/2026-11-19-replay-and-animation
"""

import argparse
import json
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

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
    with Image.open(path) as image:
        return image.convert("RGB")


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    names = {"density": "density.jpg", "range-study": "range-study.jpg"}
    with tempfile.TemporaryDirectory() as scratch:
        for name, filename in names.items():
            images = {theme: _render(name, theme, Path(scratch)) for theme in THEMES}
            target = args.out / filename
            _sheet(images).save(target, quality=85)
            try:
                print(target.resolve().relative_to(ROOT))
            except ValueError:
                print(target.name)


if __name__ == "__main__":
    main()
