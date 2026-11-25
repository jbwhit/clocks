import dataclasses
import io
import re
import shutil
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import IPython.display
import matplotlib
import matplotlib.ft2font
import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
import pytest
from _figure_artists import png_size
from matplotlib import font_manager
from matplotlib.colors import to_rgb
from matplotlib.font_manager import FontProperties

import clocks._style
from clocks._style import (
    THEMES,
    Palette,
    frame_rgba,
    palette,
    save_png,
    show,
    style,
)


def _luminance(hex_color: str) -> float:
    channels = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(a: str, b: str) -> float:
    lighter, darker = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


_STR_FIELDS = tuple(
    field.name for field in dataclasses.fields(Palette) if field.name != "ordered"
)


def test_themes_are_light_then_dark() -> None:
    assert THEMES == ("light", "dark")


@pytest.mark.parametrize("theme", THEMES)
def test_palette_colours_are_six_digit_hex(theme: str) -> None:
    p = palette(theme)
    colours = [getattr(p, name) for name in _STR_FIELDS] + list(p.ordered)
    for colour in colours:
        assert re.fullmatch(r"#[0-9a-f]{6}", colour), colour
    assert len(p.ordered) == 3


@pytest.mark.parametrize("name", (*_STR_FIELDS, "ordered"))
def test_themes_differ_in_every_role(name: str) -> None:
    assert getattr(palette("light"), name) != getattr(palette("dark"), name)


@pytest.mark.parametrize("theme", THEMES)
def test_semantic_hues_are_distinct(theme: str) -> None:
    p = palette(theme)
    assert len({p.truth, p.belief, p.belief_alt}) == 3


@pytest.mark.parametrize("theme", THEMES)
def test_data_colours_clear_three_to_one(theme: str) -> None:
    p = palette(theme)
    data = [p.truth, p.belief, p.belief_alt, p.muted, p.threshold, *p.ordered]
    for colour in data:
        assert _contrast(colour, p.background) >= 3.0, colour


@pytest.mark.parametrize("theme", THEMES)
def test_ink_clears_seven_to_one(theme: str) -> None:
    p = palette(theme)
    assert _contrast(p.ink, p.background) >= 7.0


@pytest.mark.parametrize("theme", THEMES)
def test_ordered_runs_from_least_to_most_contrast(theme: str) -> None:
    p = palette(theme)
    ratios = [_contrast(c, p.background) for c in p.ordered]
    assert ratios[0] < ratios[1] < ratios[2]


@pytest.mark.parametrize("bad", ["sepia", "Light"])
def test_palette_rejects_unknown_theme(bad: str) -> None:
    with pytest.raises(ValueError) as excinfo:
        palette(bad)
    message = str(excinfo.value)
    assert bad in message and "light" in message and "dark" in message


def test_palette_is_frozen() -> None:
    assert isinstance(palette("light"), Palette)
    with pytest.raises(AttributeError):
        palette("light").ink = "#000000"  # type: ignore[misc]


_SENTINELS = {
    "figure.facecolor": "#123456",
    "axes.edgecolor": "#654321",
    "text.color": "#abcdef",
    "font.family": ["monospace"],
}
_SENTINEL_CYCLE = ["#010203"]


def _set_sentinels() -> None:
    plt.rcParams.update(_SENTINELS)
    plt.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=_SENTINEL_CYCLE)


def _assert_sentinels() -> None:
    for key, value in _SENTINELS.items():
        assert plt.rcParams[key] == value, key
    assert plt.rcParams["axes.prop_cycle"].by_key()["color"] == _SENTINEL_CYCLE


@pytest.mark.parametrize("theme", THEMES)
def test_style_applies_theme_and_restores(theme: str) -> None:
    with matplotlib.rc_context():
        _set_sentinels()
        with style(theme) as colors:
            rc = plt.rcParams
            assert colors == palette(theme)
            assert rc["figure.facecolor"] == colors.background
            assert rc["axes.edgecolor"] == colors.rule
            assert rc["text.color"] == colors.ink
            assert rc["font.family"] == ["Source Sans 3"]
            assert rc["figure.figsize"] == [7.5, 4.2]
            assert rc["figure.autolayout"] is False
            assert rc["axes.prop_cycle"].by_key()["color"] == [colors.ink]
            assert rc["axes.axisbelow"] is True
            assert rc["axes.facecolor"] == colors.background
            assert rc["savefig.facecolor"] == colors.background
            assert rc["axes.labelcolor"] == colors.muted
            assert rc["xtick.color"] == colors.muted
            assert rc["ytick.color"] == colors.muted
            assert rc["grid.color"] == colors.grid
            assert rc["figure.constrained_layout.use"] is True
            assert rc["font.size"] == 10.5
        _assert_sentinels()


@pytest.mark.parametrize("theme", THEMES)
def test_style_restores_when_the_block_raises(theme: str) -> None:
    with matplotlib.rc_context():
        _set_sentinels()
        with pytest.raises(RuntimeError):
            with style(theme):
                raise RuntimeError("boom")
        _assert_sentinels()


def test_nested_style_restores_the_outer_theme() -> None:
    with matplotlib.rc_context():
        with style("light"):
            with style("dark"):
                assert plt.rcParams["figure.facecolor"] == palette("dark").background
            assert plt.rcParams["figure.facecolor"] == palette("light").background


@pytest.mark.parametrize("bad", ["sepia", "Light"])
def test_style_rejects_unknown_theme(bad: str) -> None:
    with matplotlib.rc_context():
        before = dict(plt.rcParams)
        with pytest.raises(ValueError) as error:
            with style(bad):
                pass
        message = str(error.value)
        assert bad in message and "light" in message and "dark" in message
        assert dict(plt.rcParams) == before


def test_bundled_fonts_are_registered_once() -> None:
    font_dir = Path(clocks._style.__file__).parent / "_fonts"

    def bundled() -> list[font_manager.FontEntry]:
        return [
            entry
            for entry in font_manager.fontManager.ttflist
            if Path(entry.fname).is_relative_to(font_dir)
        ]

    with style("light"):
        pass
    entries = bundled()
    assert len(entries) == 2
    assert {entry.name for entry in entries} == {"Source Sans 3"}
    assert {entry.weight for entry in entries} == {400, 600}
    with style("dark"):
        pass
    assert len(bundled()) == 2
    for weight in (400, 600):
        font_manager.findfont(
            FontProperties(family="Source Sans 3", weight=weight),
            fallback_to_default=False,
        )


def test_missing_font_file_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(clocks._style, "_FONT_DIR", tmp_path)
    monkeypatch.setattr(clocks._style, "_fonts_registered", False)
    with pytest.raises(FileNotFoundError, match=r"SourceSans3-Regular\.ttf"):
        with style("light"):
            pass


def test_a_partly_missing_font_set_registers_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regular = clocks._style._FONT_DIR / "SourceSans3-Regular.ttf"
    shutil.copy(regular, tmp_path / regular.name)
    monkeypatch.setattr(clocks._style, "_FONT_DIR", tmp_path)
    monkeypatch.setattr(clocks._style, "_fonts_registered", False)
    before = len(font_manager.fontManager.ttflist)
    with pytest.raises(FileNotFoundError, match=r"SourceSans3-Semibold\.ttf"):
        with style("light"):
            pass
    assert len(font_manager.fontManager.ttflist) == before


def _bundled_entries() -> list[font_manager.FontEntry]:
    font_dir = Path(clocks._style.__file__).parent / "_fonts"
    return [
        entry
        for entry in font_manager.fontManager.ttflist
        if Path(entry.fname).is_relative_to(font_dir)
    ]


def test_a_font_that_fails_to_load_registers_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regular = clocks._style._FONT_DIR / "SourceSans3-Regular.ttf"
    shutil.copy(regular, tmp_path / regular.name)
    (tmp_path / "SourceSans3-Semibold.ttf").write_bytes(b"not a font\n" * 64)
    monkeypatch.setattr(clocks._style, "_FONT_DIR", tmp_path)
    monkeypatch.setattr(clocks._style, "_fonts_registered", False)
    before = len(font_manager.fontManager.ttflist)
    # matplotlib 3.10's FT2Font raises RuntimeError for a file FreeType cannot load.
    with pytest.raises(RuntimeError, match="Can not load face"):
        with style("light"):
            pass
    assert len(font_manager.fontManager.ttflist) == before


def test_a_font_whose_metadata_fails_registers_nothing_and_retry_is_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = font_manager.fontManager
    bundled = _bundled_entries()
    without_bundled = [entry for entry in manager.ttflist if entry not in bundled]
    monkeypatch.setattr(manager, "ttflist", without_bundled)
    monkeypatch.setattr(clocks._style, "_fonts_registered", False)
    real_ttf_font_property = font_manager.ttfFontProperty

    def failing_on_semibold(font: matplotlib.ft2font.FT2Font) -> object:
        # A face can open in FT2Font yet carry a name record addfont cannot
        # decode; this stands in for that file.
        if Path(font.fname).name == "SourceSans3-Semibold.ttf":
            raise UnicodeDecodeError("utf-16", b"", 0, 1, "bad name record")
        return real_ttf_font_property(font)

    monkeypatch.setattr(font_manager, "ttfFontProperty", failing_on_semibold)
    before = len(manager.ttflist)
    for _attempt in ("first", "retry"):
        with pytest.raises(UnicodeDecodeError, match="bad name record"):
            with style("light"):
                pass
        assert len(manager.ttflist) == before

    monkeypatch.setattr(font_manager, "ttfFontProperty", real_ttf_font_property)
    with style("light"):
        pass
    assert len(_bundled_entries()) == 2


def test_concurrent_first_use_registers_once(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = font_manager.fontManager
    # Start from a registry without the bundled fonts, as in a fresh process;
    # monkeypatch puts the real list back afterwards.
    bundled = _bundled_entries()
    without_bundled = [entry for entry in manager.ttflist if entry not in bundled]
    monkeypatch.setattr(manager, "ttflist", without_bundled)
    monkeypatch.setattr(clocks._style, "_fonts_registered", False)
    real_ft2font = clocks._style.ft2font.FT2Font

    def slow_ft2font(path: str) -> object:
        # Widen the gap between checking the flag and setting it, so an
        # unguarded first use is caught every time, not only when unlucky.
        time.sleep(0.02)
        return real_ft2font(path)

    monkeypatch.setattr(clocks._style.ft2font, "FT2Font", slow_ft2font)
    barrier = threading.Barrier(8)
    errors: list[BaseException] = []

    def first_use() -> None:
        try:
            barrier.wait()
            with style("light"):
                pass
        except BaseException as error:  # reported by the assertion below
            errors.append(error)

    # rc_context is not thread-safe: threads leaving their style blocks out of
    # order can leave a theme applied, so the outer context restores rcParams.
    with matplotlib.rc_context():
        threads = [threading.Thread(target=first_use) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert errors == []
    assert len(_bundled_entries()) == 2


@pytest.fixture
def agg_backend() -> Iterator[None]:
    # On macOS the interactive backend doubles fig.dpi on a retina display,
    # which would change the pixel counts these tests pin.
    previous = matplotlib.get_backend()
    plt.switch_backend("agg")
    yield
    plt.close("all")
    plt.switch_backend(previous)


def _record_display(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    calls: list[object] = []
    monkeypatch.setattr(IPython.display, "display", calls.append)
    return calls


def _png_array(image: IPython.display.Image) -> object:
    return matplotlib.image.imread(io.BytesIO(image.data), format="png")


@pytest.mark.usefixtures("agg_backend")
def test_show_emits_the_authored_canvas(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _record_display(monkeypatch)
    with matplotlib.rc_context({"savefig.bbox": "tight", "savefig.pad_inches": 0.5}):
        with style("light"):
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            show(fig)
    assert len(calls) == 1
    image = calls[0]
    assert isinstance(image, IPython.display.Image)
    assert image.retina
    pixels = _png_array(image)
    assert pixels.shape in {(840, 1500, 4), (840, 1500, 3)}
    if pixels.shape[2] == 4:
        assert pixels[:, :, 3].min() == 1
    assert image.width == 750
    assert image.height == 420
    assert not plt.fignum_exists(fig.number)


@pytest.mark.usefixtures("agg_backend")
def test_show_scales_with_figure_size_and_dpi(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _record_display(monkeypatch)
    with matplotlib.rc_context({"savefig.bbox": "tight", "savefig.pad_inches": 0.5}):
        with style("light"):
            fig = plt.figure(figsize=(5, 3), dpi=120)
            fig.add_subplot().plot([0, 1], [0, 1])
            show(fig)
    image = calls[0]
    assert _png_array(image).shape[:2] == (720, 1200)
    assert (image.width, image.height) == (600, 360)


@pytest.mark.usefixtures("agg_backend")
def test_show_closes_the_figure_when_saving_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _record_display(monkeypatch)
    with style("light"):
        fig, ax = plt.subplots()
        ax.set_title(r"$\frac{1}{$")  # invalid mathtext: fails at draw time
        # matplotlib 3.10 reports a mathtext parse error as a ValueError.
        with pytest.raises(ValueError, match="frac"):
            show(fig)
    assert calls == []
    assert not plt.fignum_exists(fig.number)


def test_show_explains_a_missing_ipython(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "IPython.display", None)
    with matplotlib.rc_context():
        fig = plt.figure()
        try:
            with pytest.raises(ImportError, match=r"clocks\.viz\.show needs IPython"):
                show(fig)
        finally:
            plt.close(fig)


@pytest.mark.usefixtures("agg_backend")
def test_show_ignores_the_canvas_device_pixel_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _record_display(monkeypatch)
    with style("light"):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        fig.canvas._set_device_pixel_ratio(2)
        assert fig.canvas.device_pixel_ratio == 2
        assert fig.dpi == 200
        show(fig)
    image = calls[0]
    assert _png_array(image).shape[:2] == (840, 1500)
    assert (image.width, image.height) == (750, 420)


@pytest.mark.parametrize(
    "caller_settings",
    [
        pytest.param({}, id="plain"),
        pytest.param(
            {
                "savefig.bbox": "tight",
                "savefig.transparent": True,
                "savefig.facecolor": "magenta",
            },
            id="hostile",
        ),
    ],
)
def test_save_png_writes_the_authored_canvas(
    tmp_path: Path, caller_settings: dict[str, object]
) -> None:
    with style("light"):
        fig, ax = plt.subplots(figsize=(7.5, 5.4))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        (line,) = ax.plot([0, 1], [0, 1])
    try:
        with matplotlib.rc_context(caller_settings):
            save_png(fig, tmp_path / "out.bin")
        assert [p.name for p in tmp_path.iterdir()] == ["out.bin"]
        # The same figure with only its line hidden: limits and ticks agree, so
        # any difference between the two images is the plotted line.
        line.set_visible(False)
        with matplotlib.rc_context(caller_settings):
            save_png(fig, tmp_path / "hidden.bin")
    finally:
        plt.close(fig)

    assert (tmp_path / "out.bin").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert png_size(tmp_path / "out.bin") == (1500, 1080)
    image = matplotlib.image.imread(tmp_path / "out.bin", format="png")
    hidden = matplotlib.image.imread(tmp_path / "hidden.bin", format="png")
    pixel = image[0, 0]
    assert hidden.shape == image.shape
    assert np.any(image != hidden), "the plotted line is missing"
    np.testing.assert_allclose(
        pixel[:3], to_rgb(palette("light").background), rtol=0, atol=1 / 255
    )
    assert pixel[3] == 1.0


@pytest.mark.parametrize(
    "caller_settings",
    [
        pytest.param({}, id="plain"),
        pytest.param(
            {
                "savefig.bbox": "tight",
                "savefig.transparent": True,
                "savefig.facecolor": "magenta",
            },
            id="hostile",
        ),
    ],
)
def test_frame_rgba_is_what_save_png_writes(
    tmp_path: Path, caller_settings: dict[str, object]
) -> None:
    with style("light"):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot([0, 1], [0, 1])
    path = tmp_path / "poster.png"
    try:
        with matplotlib.rc_context(caller_settings):
            frame = frame_rgba(fig)
            save_png(fig, path)
    finally:
        plt.close(fig)

    assert len(frame) == 1500 * 900 * 4
    pixels = np.frombuffer(frame, dtype=np.uint8).reshape(900, 1500, 4)
    written = np.round(matplotlib.image.imread(path, format="png") * 255)
    np.testing.assert_array_equal(pixels, written.astype(np.uint8))
    background = tuple(round(c * 255) for c in to_rgb(palette("light").background))
    assert tuple(pixels[0, 0]) == (*background, 255)
