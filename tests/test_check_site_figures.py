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
    theme: str,
    images: list[tuple[str, int]],
    captions: list[str],
    height: int,
    tag: str = "div",
    extra_class: str = "",
    before_figure: str = "",
) -> str:
    imgs = "".join(
        f'<p><img src="{src}" width="{width}" height="{height}" '
        'class="figure-img"></p>\n'
        for src, width in images
    )
    caps = "".join(f"<figcaption>{c}</figcaption>\n" for c in captions)
    classes = f"cell-output cell-output-display {theme}-content{extra_class}"
    return (
        f'<{tag} class="{classes}">\n{before_figure}'
        '<div class="quarto-figure quarto-figure-center">\n'
        f'<figure class="figure">\n{imgs}{caps}</figure>\n</div>\n</{tag}>\n'
    )


def _build(
    tmp_path: Path,
    *,
    light: list[tuple[str, int]] | None = None,
    dark: list[tuple[str, int]] | None = None,
    light_captions: list[str] | None = None,
    dark_captions: list[str] | None = None,
    light_height: int = 420,
    dark_height: int = 420,
    dark_extra_class: str = "",
    tag: str = "div",
    before_figure: str = "",
    nest_dark: bool = False,
    extra: str = "",
) -> Path:
    for name in ("light.png", "dark.png", "plot.png"):
        (tmp_path / "story" / "demo_files").mkdir(parents=True, exist_ok=True)
        (tmp_path / "story" / "demo_files" / name).write_bytes(name.encode())
    light = light if light is not None else [("demo_files/light.png", 750)]
    dark = dark if dark is not None else [("demo_files/dark.png", 750)]
    lc = light_captions if light_captions is not None else [CAPTION]
    dc = dark_captions if dark_captions is not None else [CAPTION]
    light_html = _wrapper("light", light, lc, light_height, tag, "", before_figure)
    dark_html = _wrapper(
        "dark", dark, dc, dark_height, tag, dark_extra_class, before_figure
    )
    if nest_dark:
        closing = f"</{tag}>\n"
        light_html = light_html.removesuffix(closing) + dark_html + closing
        dark_html = ""
    body = (
        '<p><img src="demo_files/light.png" width="1" height="1"></p>\n'  # ignored
        + light_html
        + dark_html
        + extra
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


def test_a_well_formed_page_with_section_wrappers_passes(tmp_path: Path) -> None:
    # The committed image after the wrappers is ignored only if each wrapper's
    # end is found at its own </section>, not at the last </div> inside it.
    later = '<div><p><img src="demo_files/plot.png" width="1" height="1"></p></div>\n'
    site = _build(tmp_path, tag="section", extra=later)
    assert check_site_figures.figure_failures(site) == []


def test_a_dark_caption_in_a_second_dark_wrapper(tmp_path: Path) -> None:
    second = _wrapper("dark", [], [CAPTION], 420)
    _single_failure(
        _build(tmp_path, dark_captions=[], extra=second),
        "dark-content: expected 1 wrapper, found 2",
    )


def test_a_wrapper_with_both_theme_classes(tmp_path: Path) -> None:
    _single_failure(
        _build(tmp_path, dark_extra_class=" light-content"),
        "wrapper has both theme classes",
    )


def test_an_absolute_image_path_is_outside_the_site(tmp_path: Path) -> None:
    site = tmp_path / "site"
    target = site / "story" / "demo_files" / "dark.png"
    _single_failure(
        _build(site, dark=[(str(target), 750)]), "image is outside the site"
    )
    assert target.is_file()


def test_a_percent_encoded_absolute_path_is_outside_the_site(tmp_path: Path) -> None:
    site = tmp_path / "site"
    target = site / "story" / "demo_files" / "dark.png"
    encoded = "%2F" + target.as_posix().removeprefix("/")
    _single_failure(_build(site, dark=[(encoded, 750)]), "image is outside the site")
    assert target.is_file()


def test_a_malformed_image_url_is_a_failure(tmp_path: Path) -> None:
    _single_failure(_build(tmp_path, dark=[("//[bad", 750)]), "image URL is malformed")


def test_an_image_path_escaping_the_site_is_outside_it(tmp_path: Path) -> None:
    site = tmp_path / "site"
    (tmp_path / "outside.png").write_bytes(b"outside")
    _single_failure(
        _build(site, dark=[("../../outside.png", 750)]), "image is outside the site"
    )


def test_a_well_formed_tall_page_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _build(tmp_path, light_height=540, dark_height=540)
    monkeypatch.setattr(check_site_figures, "EXPECTED_PAGES", {PAGE: 540})
    assert check_site_figures.figure_failures(site) == []
    monkeypatch.setattr(check_site_figures, "EXPECTED_PAGES", {PAGE: 420})
    failures = check_site_figures.figure_failures(site)
    assert len(failures) == 2, failures
    for theme, failure in zip(("light", "dark"), failures, strict=True):
        assert f"{PAGE}: {theme}-content: height 540, expected 420" == failure


def test_section_wrappers_with_a_div_before_the_image_pass(tmp_path: Path) -> None:
    # A wrapper that closed at the first </div> inside it would lose its image.
    site = _build(
        tmp_path, tag="section", before_figure='<div class="cell-output"></div>\n'
    )
    assert check_site_figures.figure_failures(site) == []


def test_a_dark_wrapper_nested_inside_the_light_one(tmp_path: Path) -> None:
    _single_failure(
        _build(tmp_path, nest_dark=True), "theme wrapper nested inside another"
    )


def test_a_query_string_on_the_image_urls_passes(tmp_path: Path) -> None:
    site = _build(
        tmp_path,
        light=[("demo_files/light.png?v=2", 750)],
        dark=[("demo_files/dark.png?v=2", 750)],
    )
    assert check_site_figures.figure_failures(site) == []


def test_a_percent_escaped_image_url_names_its_file(tmp_path: Path) -> None:
    site = _build(tmp_path, dark=[("demo_files/dark%20plot.png", 750)])
    (site / "story" / "demo_files" / "dark plot.png").write_bytes(b"dark plot")
    assert check_site_figures.figure_failures(site) == []


def test_urls_differing_only_in_query_are_the_same_file(tmp_path: Path) -> None:
    _single_failure(
        _build(
            tmp_path,
            light=[("demo_files/plot.png?v=1", 750)],
            dark=[("demo_files/plot.png?v=2", 750)],
        ),
        "same image file",
    )


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
