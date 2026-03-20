"""
deck_utils.py — Utilities for listing, filtering, and randomly selecting decks.

Used by the training loop to randomly pick decks from Assets/ each game.
"""

from __future__ import annotations

import random
from pathlib import Path

# Default Assets/ directory (relative to this file → talishar_ai/ → Talishar/)
ASSETS_DIR = Path(__file__).resolve().parents[1] / "Assets"

# Minimum total cards (deck + inventory) required per format
_FORMAT_MIN_TOTAL = {
    "cc": 60,
    "blitz": 40,
}


def list_decks(
    format_filter: str | None = None,
    assets_dir: Path | None = None,
    exclude: list[str] | None = None,
) -> list[str]:
    """
    List available deck names from Assets/, optionally filtered by format.

    Parameters
    ----------
    format_filter : str | None
        Game format (e.g. "cc", "blitz").  Filters decks by total card count.
    assets_dir : Path | None
        Override Assets/ directory path.
    exclude : list[str] | None
        Deck names to exclude (e.g. ["Dummy"]).

    Returns
    -------
    list[str]
        Sorted list of deck stem names (without .txt).
    """
    d = assets_dir or ASSETS_DIR
    skip = set(exclude or ["Dummy"])
    decks = []
    for f in sorted(d.glob("*.txt")):
        name = f.stem
        if name in skip:
            continue
        if format_filter and not _deck_matches_format(f, format_filter):
            continue
        decks.append(name)
    return decks


def random_deck(
    format_filter: str | None = None,
    assets_dir: Path | None = None,
) -> str:
    """Pick a random deck name suitable for the given format."""
    decks = list_decks(format_filter=format_filter, assets_dir=assets_dir)
    if not decks:
        raise ValueError(
            f"No decks available"
            + (f" for format '{format_filter}'" if format_filter else "")
        )
    return random.choice(decks)


def random_deck_pair(
    format_filter: str | None = None,
    assets_dir: Path | None = None,
) -> tuple[str, str]:
    """Pick two random decks (may be the same hero)."""
    return random_deck(format_filter, assets_dir), random_deck(format_filter, assets_dir)


def _deck_matches_format(deck_path: Path, fmt: str) -> bool:
    """Check if a deck is suitable for a format based on total card count."""
    try:
        lines = deck_path.read_text().strip().splitlines()
    except Exception:
        return False
    if len(lines) < 2:
        return False

    deck_count = len(lines[1].strip().split())
    inv_count = sum(1 for line in lines[2:] if line.strip())
    total = deck_count + inv_count

    fmt_lower = fmt.lower()
    min_total = _FORMAT_MIN_TOTAL.get(fmt_lower)
    if min_total is not None:
        return total >= min_total
    return True  # unknown format — include all
