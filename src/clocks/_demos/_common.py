"""Shared command-line contracts for packaged demos."""

from __future__ import annotations

import argparse
from pathlib import Path


def positive_int(value: str) -> int:
    """Parse a strictly positive integer for a command-line option."""
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected an integer, got {value!r}"
        ) from error
    if result <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return result


def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    output: Path,
    observations: int,
    particles: int,
) -> None:
    """Add the common reproducible-output and fast-smoke overrides."""
    parser.add_argument(
        "--output",
        type=Path,
        default=output,
        help=f"output image or animation (default: {output})",
    )
    parser.add_argument(
        "--observations",
        type=positive_int,
        default=observations,
        help=f"number of simulated observations (default: {observations})",
    )
    parser.add_argument(
        "--particles",
        type=positive_int,
        default=particles,
        help=f"number of SMC particles (default: {particles})",
    )
