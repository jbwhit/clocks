import importlib.util
from pathlib import Path

import pytest
import test_site_figure_style as site_style

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_site_figures.py"
_SPEC = importlib.util.spec_from_file_location("check_site_figures", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
check_site_figures = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_site_figures)

# Captured before the autouse fixture below replaces the constant.
REAL_EXPECTED_PAGES = dict(check_site_figures.EXPECTED_PAGES)

PAGE = "story/demo.html"
CAPTION = "Rate of <code>x</code> versus distance."


def _wrapper(
    theme: str, images: list[tuple[str, int]], captions: list[str], height: int
) -> str:
    imgs = "".join(
        f'<p><img src="{src}" width="{width}" height="{height}" '
        'class="figure-img"></p>\n'
        for src, width in images
    )
    caps = "".join(f"<figcaption>{c}</figcaption>\n" for c in captions)
    return (
        f'<div class="cell-output cell-output-display {theme}-content">\n'
        '<div class="quarto-figure quarto-figure-center">\n'
        f'<figure class="figure">\n{imgs}{caps}</figure>\n</div>\n</div>\n'
    )


def _build(
    tmp_path: Path,
    *,
    light: list[tuple[str, int]] | None = None,
    dark: list[tuple[str, int]] | None = None,
    light_captions: list[str] | None = None,
    dark_captions: list[str] | None = None,
    dark_height: int = 420,
) -> Path:
    for name in ("light.png", "dark.png", "plot.png"):
        (tmp_path / "story" / "demo_files").mkdir(parents=True, exist_ok=True)
        (tmp_path / "story" / "demo_files" / name).write_bytes(name.encode())
    light = light if light is not None else [("demo_files/light.png", 750)]
    dark = dark if dark is not None else [("demo_files/dark.png", 750)]
    lc = light_captions if light_captions is not None else [CAPTION]
    dc = dark_captions if dark_captions is not None else [CAPTION]
    body = (
        '<p><img src="demo_files/light.png" width="1" height="1"></p>\n'  # ignored
        + _wrapper("light", light, lc, 420)
        + _wrapper("dark", dark, dc, dark_height)
    )
    (tmp_path / "story" / "demo.html").write_text(
        f"<html><body>{body}</body></html>", encoding="utf-8"
    )
    return tmp_path


@pytest.fixture(autouse=True)
def one_expected_page(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check_site_figures, "EXPECTED_PAGES", {PAGE: 420})


def _single_failure(tmp_path: Path, reason: str) -> None:
    failures = check_site_figures.figure_failures(tmp_path)
    assert len(failures) == 1, failures
    assert PAGE in failures[0]
    assert reason in failures[0]


def test_a_well_formed_page_passes(tmp_path: Path) -> None:
    assert check_site_figures.figure_failures(_build(tmp_path)) == []


def test_missing_dark_image(tmp_path: Path) -> None:
    _single_failure(
        _build(tmp_path, dark=[]), "dark-content: expected 1 image, found 0"
    )


def test_two_light_images(tmp_path: Path) -> None:
    light = [("demo_files/light.png", 750), ("demo_files/plot.png", 750)]
    _single_failure(
        _build(tmp_path, light=light), "light-content: expected 1 image, found 2"
    )


def test_wrong_width(tmp_path: Path) -> None:
    _single_failure(
        _build(tmp_path, light=[("demo_files/light.png", 597)]),
        "light-content: width 597, expected 750",
    )


def test_wrong_height(tmp_path: Path) -> None:
    _single_failure(
        _build(tmp_path, dark_height=540), "dark-content: height 540, expected 420"
    )


def test_same_file_in_both_wrappers(tmp_path: Path) -> None:
    _single_failure(
        _build(
            tmp_path,
            light=[("demo_files/plot.png", 750)],
            dark=[("./demo_files/plot.png", 750)],
        ),
        "same image file",
    )


def test_src_that_does_not_exist(tmp_path: Path) -> None:
    _single_failure(
        _build(tmp_path, dark=[("demo_files/gone.png", 750)]),
        "image 'demo_files/gone.png' does not exist",
    )


def test_no_figcaption_in_dark_wrapper(tmp_path: Path) -> None:
    _single_failure(
        _build(tmp_path, dark_captions=[]),
        "dark-content: expected 1 figcaption, found 0",
    )


def test_two_figcaptions_in_one_wrapper(tmp_path: Path) -> None:
    _single_failure(
        _build(tmp_path, light_captions=[CAPTION, CAPTION]),
        "light-content: expected 1 figcaption, found 2",
    )


def test_captions_differing_only_inside_nested_code(tmp_path: Path) -> None:
    _single_failure(
        _build(tmp_path, dark_captions=["Rate of <code>y</code> versus distance."]),
        "captions differ",
    )


def test_empty_caption_in_dark_wrapper(tmp_path: Path) -> None:
    _single_failure(_build(tmp_path, dark_captions=[""]), "dark-content: empty caption")


def test_a_missing_page_is_a_failure(tmp_path: Path) -> None:
    _single_failure(tmp_path, "page is missing")


def test_expected_pages_match_the_plot_pages_and_their_heights() -> None:
    plot_pages = {page.replace(".qmd", ".html") for page in site_style.PLOT_PAGES}
    assert len(plot_pages) == 9 and len(REAL_EXPECTED_PAGES) == 9
    assert set(REAL_EXPECTED_PAGES) == plot_pages
    for page, height in REAL_EXPECTED_PAGES.items():
        cell = site_style._only_plot_cell(page.replace(".html", ".qmd"))
        cell_height = 540 if "figsize=(7.5, 5.4)" in cell else 420
        assert height == cell_height, page
