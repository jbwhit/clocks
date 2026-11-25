"""The package's source tree, read as bytes: the standard library only.

``site/prerender.py`` runs this module from its file, before ``clocks`` is
imported, so the snapshot it takes is of the source the run then compiles.
"""

import stat
from collections.abc import Iterator
from pathlib import Path


def read_sources(root: Path) -> list[tuple[str, bytes]]:
    """Each regular file under ``root``, outside ``__pycache__`` directories, as
    its path relative to ``root`` and its bytes, sorted by that path.

    A directory that cannot be listed, ``root`` itself included, or an entry
    whose status cannot be read, raises its ``OSError``: an unread part of the
    tree would leave it out of the digest.
    """
    items = [
        (path.relative_to(root).as_posix(), path.read_bytes()) for path in _files(root)
    ]
    return sorted(items)


def _files(root: Path) -> Iterator[Path]:
    for directory, subdirectories, files in root.walk(on_error=_raise):
        subdirectories[:] = [d for d in subdirectories if d != "__pycache__"]
        for name in files:
            path = directory / name
            if stat.S_ISREG(path.stat().st_mode):  # a failed stat raises
                yield path


def _raise(error: OSError) -> None:
    raise error
