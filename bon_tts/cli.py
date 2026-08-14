"""Shared argument-parser plumbing for the scripts in ``scripts/``."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_CONFIG = "configs/librispeech_pc.yaml"


def build_parser(doc: str | None) -> argparse.ArgumentParser:
    """Parser that renders the module docstring as-is in ``--help``.

    The scripts document their usage in the docstring, so the default help
    formatter's paragraph re-wrapping would mangle the example commands.
    """
    return argparse.ArgumentParser(
        description=doc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the ``--config`` / ``--set`` pair every script accepts."""
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        metavar="KEY=VALUE",
        help="Override a config value, e.g. --set pool.n_candidates=5",
    )
    return parser
