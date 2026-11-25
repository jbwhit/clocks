"""Quarto pre-render step: draw the site's clips and posters into media/.

The package's source tree is read before any clocks module is imported, and
clocks is then compiled from source into a fresh bytecode cache: cached
bytecode is trusted by its source's size and time, which an edit can keep, so
without this the digest each pair's record holds could name code that did not
draw it.
"""

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

SITE = Path(__file__).resolve().parent


def main(
    archives: Path, media: Path, *, full: bool, demos: tuple[str, ...] | None = None
) -> None:
    spec = importlib.util.find_spec("clocks")  # locates the package, imports nothing
    package = Path(spec.submodule_search_locations[0]).resolve()
    with tempfile.TemporaryDirectory(prefix="clocks-bytecode-") as cache:
        sys.pycache_prefix = cache
        sources = _load("clocks_sources", package / "_sources.py")
        snapshot = sources.read_sources(package)
        from clocks._site_media import render_site_media

        options = {} if demos is None else {"demos": demos}
        render_site_media(
            archives,
            media,
            full=full,
            sources=package,
            sources_snapshot=snapshot,
            **options,
        )


def _load(name: str, path: Path):
    """Run the module at ``path`` alone, without importing its package."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main(
        SITE.parent / "assets" / "replays",
        SITE / "media",
        full=os.environ.get("QUARTO_PROJECT_RENDER_ALL") == "1",
    )
