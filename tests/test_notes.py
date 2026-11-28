"""The best-effort note on an exception that is already propagating."""

import pytest

from clocks._notes import add_note


class _InterruptedNote(ValueError):
    def add_note(self, note: str) -> None:
        raise KeyboardInterrupt("while adding the note")


@pytest.mark.parametrize("case", ["list", "tuple", "interrupted"])
def test_a_note_is_added_when_it_can_be_and_never_raises(case: str) -> None:
    """Only this catches the helper raising when ``__notes__`` is not a list
    (``[tuple]``) or when an interrupt arrives while it adds the note
    (``[interrupted]``, which an ``except Exception`` would let through)."""
    error = _InterruptedNote("boom") if case == "interrupted" else ValueError("boom")
    if case == "tuple":
        error.__notes__ = ()

    add_note(error, "the note")

    expected = {"list": ["the note"], "tuple": (), "interrupted": None}[case]
    assert getattr(error, "__notes__", None) == expected
