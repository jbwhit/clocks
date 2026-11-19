import re

import pytest

from clocks._style import THEMES, Palette, palette


def _luminance(hex_color: str) -> float:
    channels = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(a: str, b: str) -> float:
    lighter, darker = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


_STR_FIELDS = (
    "background",
    "ink",
    "muted",
    "rule",
    "grid",
    "truth",
    "belief",
    "belief_alt",
    "threshold",
)


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
