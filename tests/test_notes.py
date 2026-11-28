"""A cleanup step whose failure becomes a note and never replaces the error."""

from pathlib import Path

import pytest

from clocks._notes import attempt, left_behind


class _Unprintable(Exception):
    def __str__(self) -> str:
        raise RuntimeError("cannot describe it")


class _InterruptedNote(ValueError):
    def add_note(self, note: str) -> None:
        raise KeyboardInterrupt("while adding the note")


def _raises(make_error):
    def step() -> None:
        raise make_error()

    return step


# Each case builds its step and its error afresh, so a run in the same
# process as another starts from an error with no notes.
CASES = {
    "succeeds": (lambda: lambda: None, lambda: ValueError("boom"), None),
    "oserror": (
        lambda: _raises(lambda: PermissionError("busy")),
        lambda: ValueError("boom"),
        ["step: busy"],
    ),
    "interrupted_step": (
        lambda: _raises(lambda: KeyboardInterrupt("mid-step")),
        lambda: ValueError("boom"),
        ["step: mid-step"],
    ),
    "describe_raises": (
        lambda: _raises(_Unprintable),
        lambda: ValueError("boom"),
        None,
    ),
    "notes_not_a_list": (
        lambda: _raises(lambda: OSError("busy")),
        lambda: ValueError("boom"),
        (),
    ),
    "interrupted_note": (
        lambda: _raises(lambda: OSError("busy")),
        lambda: _InterruptedNote("boom"),
        None,
    ),
}


@pytest.mark.parametrize("case", list(CASES))
def test_a_failed_step_is_a_note_and_never_raises(case: str) -> None:
    """Catches the helper letting anything out: a step's own interrupt
    (``[interrupted_step]``), a description that raises while it is built
    (``[describe_raises]``), a ``__notes__`` that is not a list, or an
    interrupt while the note is added; and a note added for a step that
    succeeded. The Task 7 rows name the tests that share each."""
    make_step, make_error, expected = CASES[case]
    step, error = make_step(), make_error()
    if case == "notes_not_a_list":
        error.__notes__ = ()

    try:
        attempt(error, step, lambda failure: f"step: {failure}")
    except BaseException as escaped:  # an escaped interrupt would end the session
        pytest.fail(f"attempt raised {escaped!r}")

    assert getattr(error, "__notes__", None) == expected


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        pytest.param(
            PermissionError("busy"),
            "out/x.part may have been left behind: PermissionError: busy",
            id="with_a_message",
        ),
        pytest.param(
            KeyboardInterrupt(),
            "out/x.part may have been left behind: KeyboardInterrupt: ",
            id="bare_interrupt",
        ),
    ],
)
def test_a_left_behind_note_names_the_path_and_the_cause(
    failure: BaseException, expected: str
) -> None:
    """Only this catches the note asserting the file *was* left behind (an
    interrupt can land after the removal took effect), or naming no cause
    for a bare interrupt, whose message is empty."""
    assert left_behind(Path("out/x.part"), failure) == expected
