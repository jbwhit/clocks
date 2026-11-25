"""Quarto pre-render step: draw the site's clips and posters into media/.

The package's source tree is read once, before any clocks module is imported,
and written to a private temporary directory; the renderer is imported from
that copy. So the code that draws is exactly the snapshot each pair's record
names, whatever happens to the live tree, or to bytecode cached beside it,
meanwhile. The live tree is read again before each record is written, so an
edit that persists is caught too.
"""

import importlib.util
import os
import sys
import tempfile
import types
from collections.abc import Callable
from pathlib import Path

import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")

SITE = Path(__file__).resolve().parent


def main(
    archives: Path, media: Path, *, full: bool, demos: tuple[str, ...] | None = None
) -> None:
    """Render the media from a private copy of the package; restore what it set.

    Refuses to run when any ``clocks`` module is already imported: that code,
    not the snapshot, would draw.
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
    with tempfile.TemporaryDirectory(prefix="clocks-snapshot-") as root:
        try:
            snapshot = _snapshot(package)
            _write_copy(Path(root) / "clocks", snapshot)
            render_site_media = _import_renderer(Path(root))
            options = {} if demos is None else {"demos": demos}
            render_site_media(
                archives,
                media,
                full=full,
                sources=package,
                sources_snapshot=snapshot,
                **options,
            )
        finally:
            for name in [n for n in sys.modules if n.split(".")[0] == "clocks"]:
                del sys.modules[name]
            sys.path[:] = saved_path
            _restore_fonts(saved_fonts)


def _restore_fonts(saved: list[font_manager.FontEntry]) -> None:
    """Forget the fonts the copy registered: their files are about to go."""
    font_manager.fontManager.ttflist[:] = saved
    cached = getattr(font_manager.fontManager, "_findfont_cached", None)
    if cached is not None:  # lookups made while the copy's fonts were listed
        cached.cache_clear()


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
    """Import ``render_site_media`` from the copy under ``root``; check every
    clocks module it loaded came from there."""
    sys.path.insert(0, str(root))
    from clocks._site_media import render_site_media

    copy = root / "clocks"
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] != "clocks":
            continue
        origin = Path(getattr(module, "__file__", None) or "")
        if not origin.is_relative_to(copy):
            raise RuntimeError(f"{name} was imported from {origin}, not the snapshot")
    return render_site_media


if __name__ == "__main__":
    main(
        SITE.parent / "assets" / "replays",
        SITE / "media",
        full=os.environ.get("QUARTO_PROJECT_RENDER_ALL") == "1",
    )
