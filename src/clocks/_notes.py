"""A best-effort note on an exception that is already propagating."""


def add_note(error: BaseException, message: str) -> None:
    """Add ``message`` to ``error``'s notes if it can be added, and never raise.

    Cleanup code records what it could not undo as a note on the error that
    made it run; that note must never replace the error. ``add_note`` raises
    ``TypeError`` when ``__notes__`` is not a list, and an interrupt can arrive
    while it runs: either way the note is dropped and ``error`` is kept.
    """
    try:
        error.add_note(message)
    except BaseException:
        pass  # the original error is what the caller must see, note or not
