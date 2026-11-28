"""A cleanup step whose failure becomes a note on an error that is propagating."""

from collections.abc import Callable


def attempt(
    error: BaseException,
    step: Callable[[], object],
    describe: Callable[[BaseException], str],
) -> None:
    """Run the cleanup ``step``; if it raises anything, note it on ``error``.

    Cleanup code runs while ``error`` propagates. An exception raised by
    ``step`` (an interrupt included), or while its note is built or added, is
    caught here: ``describe(step_error)`` is both built and added inside the
    same guard, so a description that raises, a ``__notes__`` that is not a
    list, or an interrupt while the note is added each drop the note, never
    ``error``, and the caller goes on with its next step. That is the limit of
    the guarantee: Python can deliver an asynchronous interrupt (SIGINT)
    between any two bytecodes, so one that lands outside this call (before it
    is entered, or after it returns) is not excluded; it propagates, with
    ``error`` as its ``__context__``.
    """
    try:
        step()
    except BaseException as step_error:
        try:
            error.add_note(describe(step_error))
        except BaseException:
            pass  # the original error is what the caller must see, note or not


def left_behind(path: object, failure: BaseException) -> str:
    """The note for a temporary file whose removal raised ``failure``.

    It says "may have been": an interrupt can land after the removal took
    effect. The cause is named by its type, since a bare interrupt (a real
    SIGINT) has no message.
    """
    return f"{path} may have been left behind: {type(failure).__name__}: {failure}"
