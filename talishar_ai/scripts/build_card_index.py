"""
build_card_index.py — CLI to build the card knowledge index from PHP source.

Parses GeneratedCardDictionaries.php, computes rate-system valuations,
and indexes all ~4,633 cards into lucisearch for fast retrieval.

Usage
-----
    uv run python -m talishar_ai.scripts.build_card_index
    uv run python -m talishar_ai.scripts.build_card_index --php-path /path/to/GeneratedCardDictionaries.php
    uv run python -m talishar_ai.scripts.build_card_index --index-dir ./my_indices
"""

from __future__ import annotations

import argparse
from pathlib import Path

from talishar_ai.rag.card_index import CardIndex, _DEFAULT_PHP_PATH, _DEFAULT_INDEX_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the card knowledge index from GeneratedCardDictionaries.php"
    )
    parser.add_argument(
        "--php-path",
        default=str(_DEFAULT_PHP_PATH),
        help=f"Path to GeneratedCardDictionaries.php (default: {_DEFAULT_PHP_PATH})",
    )
    parser.add_argument(
        "--index-dir",
        default=str(_DEFAULT_INDEX_DIR),
        help=f"Directory to store the lucisearch index (default: {_DEFAULT_INDEX_DIR})",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print summary statistics after building",
    )
    args = parser.parse_args()

    index = CardIndex.build(
        php_path=args.php_path,
        persist_dir=args.index_dir,
    )

    if args.stats:
        cards = index.all_cards()
        print(f"\n--- Index Statistics ---")
        print(f"Total cards: {len(cards):,}")

        # Type distribution
        type_counts: dict[str, int] = {}
        for c in cards:
            type_counts[c.card_type] = type_counts.get(c.card_type, 0) + 1
        print(f"\nBy type:")
        for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"  {t:>6}: {count:,}")

        # Keyword distribution (top 15)
        kw_counts: dict[str, int] = {}
        for c in cards:
            for kw in c.keywords:
                kw_counts[kw] = kw_counts.get(kw, 0) + 1
        print(f"\nTop keywords:")
        for kw, count in sorted(kw_counts.items(), key=lambda x: -x[1])[:15]:
            print(f"  {kw:>15}: {count:,}")

        # Value distribution
        above_rate = sum(1 for c in cards if c.best_use_value >= 3.0)
        print(f"\nAbove rate (>=3.0): {above_rate:,} ({100*above_rate/len(cards):.0f}%)")

        # Sample high-value cards
        print(f"\nTop 10 by best_use_value:")
        top = sorted(cards, key=lambda c: c.best_use_value, reverse=True)[:10]
        for c in top:
            print(f"  {c.card_id:40s} val={c.best_use_value:.1f}  type={c.card_type}")


if __name__ == "__main__":
    main()
