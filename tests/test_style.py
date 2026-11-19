import dataclasses
import io
import re
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path

import IPython.display
import matplotlib
import matplotlib.image
import matplotlib.pyplot as plt
import pytest
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

import clocks._style
from clocks._style import THEMES, Palette, palette, show, style


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
