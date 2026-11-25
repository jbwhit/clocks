"""Quarto pre-render step: draw the site's clips and posters into media/.

The package's source tree is read once, before any clocks module is imported,
and written to a private temporary directory; the renderer is imported from
that copy. So the code that draws is exactly the snapshot each pair's record
names, whatever happens to the live tree, or to bytecode cached beside it,
meanwhile. The live tree is read again before each record is written, so an
edit that persists is caught too.
"""

import functools
import importlib.util
import os
import sys
import tempfile
import types
from collections.abc import Callable
from pathlib import Path

import matplotlib
from matplotlib import font_manager

SITE = Path(__file__).resolve().parent


def main(
    archives: Path, media: Path, *, full: bool, demos: tuple[str, ...] | None = None
) -> None:
    """Render the media from a private copy of the package; restore what it set.

    Refuses to run when any ``clocks`` module is already imported: that code,
    not the snapshot, would draw. Writes no bytecode while it runs, so nothing
    of the copy is cached outside it (a ``PYTHONPYCACHEPREFIX`` would keep one
    per run), and releases the copy's fonts before the copy is removed.
    """
    loaded = sorted(n for n in sys.modules if n == "clocks" or n.startswith("clocks."))
    if loaded:
        raise RuntimeError(
            f"clocks is already imported ({', '.join(loaded)}): run the "
            "pre-render step in a fresh process, so the snapshot is what draws"
        )
    spec = importlib.util.find_spec("clocks")  # locates the package, imports nothing
    package = Path(spec.submodule_search_locations[0]).resolve()
    saved_path = list(sys.path)
    saved_fonts = list(font_manager.fontManager.ttflist)
    saved_bytecode = sys.dont_write_bytecode
    with tempfile.TemporaryDirectory(prefix="clocks-snapshot-") as root:
        copy = Path(root) / "clocks"
        try:
            sys.dont_write_bytecode = True
            snapshot = _snapshot(package)
            _write_copy(copy, snapshot)
            render_site_media = _import_renderer(Path(root))
            options = {} if demos is None else {"demos": demos}
            render_site_media(
                archives,
                media,
                full=full,
                sources=package,
                sources_snapshot=snapshot,
                check_code=functools.partial(_check_origins, copy),
                **options,
            )
            _check_origins(copy)  # a module imported after the last pair
        finally:
            for name in [n for n in sys.modules if n.split(".")[0] == "clocks"]:
                del sys.modules[name]
            sys.path[:] = saved_path
            sys.dont_write_bytecode = saved_bytecode
            _restore_fonts(saved_fonts)


def _restore_fonts(saved: list[font_manager.FontEntry]) -> None:
    """Forget the fonts the copy registered, and close the faces opened from its
    files, before the copy is removed (an open file blocks that on Windows).

    The private caches are cleared by name, so a Matplotlib that renames them
    fails here rather than keeping a deleted path.
    """
    font_manager.fontManager.ttflist[:] = saved
    font_manager.fontManager._findfont_cached.cache_clear()
    font_manager._get_font.cache_clear()


def _snapshot(package: Path) -> list[tuple[str, bytes]]:
    """Read the package with its own ``_sources.py``, compiled here from its
    bytes: no bytecode cache is consulted, and no clocks module is imported."""
    path = package / "_sources.py"
    module = types.ModuleType("clocks_sources")
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module.read_sources(package)


def _write_copy(copy: Path, snapshot: list[tuple[str, bytes]]) -> None:
    for name, data in snapshot:
        target = copy / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def _import_renderer(root: Path) -> Callable[..., object]:
    """Import ``render_site_media`` from the copy under ``root``.

    The package is first located without running it, and refused unless it is
    the copy (an import hook ahead of the path could choose the live tree);
    then every clocks module the import loaded is checked to come from there.
    """
    sys.path.insert(0, str(root))
    copy = root / "clocks"
    spec = importlib.util.find_spec("clocks")
    found = Path(spec.origin or "") if spec is not None else Path()
    if not found.is_relative_to(copy):
        raise RuntimeError(f"clocks would be imported from {found}, not the snapshot")
    from clocks._site_media import render_site_media

    _check_origins(copy)
    return render_site_media


def _check_origins(copy: Path) -> None:
    """Raise unless every loaded clocks module came from ``copy``."""
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] != "clocks":
            continue
        origin = Path(getattr(module, "__file__", None) or "")
        if not origin.is_relative_to(copy):
            raise RuntimeError(f"{name} was imported from {origin}, not the snapshot")


if __name__ == "__main__":
    matplotlib.use("Agg")  # only when run: importing the script keeps a backend
    main(
        SITE.parent / "assets" / "replays",
        SITE / "media",
        full=os.environ.get("QUARTO_PROJECT_RENDER_ALL") == "1",
    )
