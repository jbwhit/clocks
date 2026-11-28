"""A cleanup step whose failure becomes a note on an error that is propagating."""

from collections.abc import Callable


def attempt(
    error: BaseException,
    step: Callable[[], object],
    describe: Callable[[BaseException], str],
) -> None:
    """Run the cleanup ``step``; if it raises anything, note it on ``error``.

    Cleanup code runs while ``error`` propagates, and nothing it does may
    replace that error. So any exception from ``step`` (an interrupt
    included) is caught, and ``describe(step_error)`` is both built and added
    as a note inside the same guard: a description that raises, a
    ``__notes__`` that is not a list, or an interrupt while the note is added
    each drop the note, never ``error``. This never raises.
    """
    try:
        step()
    except BaseException as step_error:
        try:
            error.add_note(describe(step_error))
        except BaseException:
            pass  # the original error is what the caller must see, note or not
