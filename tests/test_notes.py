"""A cleanup step whose failure becomes a note and never replaces the error."""

import pytest

from clocks._notes import attempt


class _Unprintable(Exception):
    def __str__(self) -> str:
        raise RuntimeError("cannot describe it")


class _InterruptedNote(ValueError):
    def add_note(self, note: str) -> None:
        raise KeyboardInterrupt("while adding the note")


def _raises(error: BaseException):
    def step() -> None:
        raise error

    return step


CASES = {
    "succeeds": (lambda: None, ValueError("boom"), None),
    "oserror": (_raises(PermissionError("busy")), ValueError("boom"), ["step: busy"]),
    "interrupted_step": (
        _raises(KeyboardInterrupt("mid-step")),
        ValueError("boom"),
        ["step: mid-step"],
    ),
    "describe_raises": (_raises(_Unprintable()), ValueError("boom"), None),
    "notes_not_a_list": (_raises(OSError("busy")), ValueError("boom"), ()),
    "interrupted_note": (_raises(OSError("busy")), _InterruptedNote("boom"), None),
}


@pytest.mark.parametrize("case", list(CASES))
def test_a_failed_step_is_a_note_and_never_raises(case: str) -> None:
    """Catches the helper letting anything out: a step's own interrupt
    (``[interrupted_step]``), a description that raises while it is built
    (``[describe_raises]``), a ``__notes__`` that is not a list, or an
    interrupt while the note is added; and a note added for a step that
    succeeded. The Task 7 rows name the tests that share each."""
    step, error, expected = CASES[case]
    if case == "notes_not_a_list":
        error.__notes__ = ()

    try:
        attempt(error, step, lambda failure: f"step: {failure}")
    except BaseException as escaped:  # an escaped interrupt would end the session
        pytest.fail(f"attempt raised {escaped!r}")

    assert getattr(error, "__notes__", None) == expected
