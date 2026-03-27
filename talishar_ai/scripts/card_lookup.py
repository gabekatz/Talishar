"""
card_lookup.py — CLI to browse and search the card index.

Usage
-----
    # Look up a specific card
    uv run python -m scripts.card_lookup pouncing_qi_blue

    # Look up multiple cards
    uv run python -m scripts.card_lookup rising_knee_thrust_red blackout_kick_red

    # Full-text search
    uv run python -m scripts.card_lookup --search "dominate"

    # Search card abilities
    uv run python -m scripts.card_lookup --search "Crouching Tiger"

    # Filter by type
    uv run python -m scripts.card_lookup --type AA --limit 20

    # Filter by class
    uv run python -m scripts.card_lookup --class NINJA

    # Filter by keyword
    uv run python -m scripts.card_lookup --keyword combo

    # Combine filters
    uv run python -m scripts.card_lookup --type AA --class NINJA --keyword combo

    # Sort by a field (desc by default)
    uv run python -m scripts.card_lookup --type AA --sort power --limit 10

    # Sort ascending
    uv run python -m scripts.card_lookup --type AA --sort cost --asc --limit 10

    # Show summary stats
    uv run python -m scripts.card_lookup --stats

    # Dump all cards as JSON (pipe to jq, etc.)
    uv run python -m scripts.card_lookup --dump --limit 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from talishar_ai.rag.card_index import CardIndex, CardDocument, _DEFAULT_INDEX_DIR


# ── Formatting helpers ──────────────────────────────────────────────

_TYPE_LABELS = {
    "AA": "Attack Action",
    "A": "Action",
    "AR": "Attack Reaction",
    "DR": "Defense Reaction",
    "I": "Instant",
    "E": "Equipment",
    "W": "Weapon",
    "T": "Token",
    "C": "Resource",
    "M": "Mentor",
}

_PITCH_COLORS = {0: "", 1: "Red", 2: "Yellow", 3: "Blue"}


def _format_card(c: CardDocument, verbose: bool = True) -> str:
    """Pretty-print a single card."""
    type_label = _TYPE_LABELS.get(c.card_type, c.card_type or "???")
    pitch_color = _PITCH_COLORS.get(c.pitch, "")

    header = f"═══ {c.name} ({c.card_id}) ═══"
    lines = [header]

    # Core stats line
    stat_parts = [f"Type: {type_label}"]
    if c.card_class:
        stat_parts.append(f"Class: {c.card_class}")
    if c.card_talent:
        stat_parts.append(f"Talent: {c.card_talent}")
    lines.append("  " + " | ".join(stat_parts))

    stat_parts2 = []
    if c.cost >= 0:
        stat_parts2.append(f"Cost: {c.cost}")
    if c.power > 0:
        stat_parts2.append(f"Power: {c.power}")
    if c.defense >= 0:
        stat_parts2.append(f"Defense: {c.defense}")
    if pitch_color:
        stat_parts2.append(f"Pitch: {c.pitch} ({pitch_color})")
    if stat_parts2:
        lines.append("  " + " | ".join(stat_parts2))

    # Keywords
    if c.keywords:
        lines.append(f"  Keywords: {', '.join(c.keywords)}")

    # Functional text (the actual card rules)
    if c.functional_text:
        # Indent multi-line ability text
        ability_lines = c.functional_text.split("\n")
        lines.append(f"  Rules: {ability_lines[0]}")
        for al in ability_lines[1:]:
            if al.strip():
                lines.append(f"         {al}")

    if verbose:
        # Rate-system valuations
        lines.append(
            f"  Valuations: attack={c.attack_value:.1f}  "
            f"block_will={c.block_willingness:.1f}  "
            f"arsenal={c.arsenal_value:.1f}  "
            f"best_use={c.best_use_value:.1f}"
        )
        lines.append(f"  Go-again: {'yes' if c.has_go_again else 'no'}")

    return "\n".join(lines)


def _print_card_table(cards: list[CardDocument], limit: int) -> None:
    """Print cards in a compact table format."""
    displayed = cards[:limit]
    # Header
    print(
        f"{'Card ID':<45} {'Type':>5} {'Cost':>4} {'Pow':>4} "
        f"{'Def':>4} {'Pitch':>5} {'AtkVal':>6} {'Keywords'}"
    )
    print("─" * 110)
    for c in displayed:
        kw = ", ".join(c.keywords[:3])
        if len(c.keywords) > 3:
            kw += f" +{len(c.keywords)-3}"
        print(
            f"{c.card_id:<45} {c.card_type:>5} {c.cost:>4} {c.power:>4} "
            f"{c.defense:>4} {c.pitch:>5} {c.attack_value:>6.1f} {kw}"
        )
    if len(cards) > limit:
        print(f"\n... and {len(cards) - limit} more (use --limit to show more)")
    print(f"\nTotal: {len(cards)} cards")


def _print_stats(idx: CardIndex) -> None:
    """Print summary statistics about the index."""
    all_cards = idx.all_cards()
    print(f"Total cards in index: {len(all_cards):,}\n")

    # Type distribution
    type_counts: dict[str, int] = {}
    for c in all_cards:
        label = _TYPE_LABELS.get(c.card_type, c.card_type or "(empty)")
        type_counts[label] = type_counts.get(label, 0) + 1
    print("By type:")
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:>20}: {count:>5,}")

    # Class distribution
    class_counts: dict[str, int] = {}
    for c in all_cards:
        cls = c.card_class or "(none)"
        class_counts[cls] = class_counts.get(cls, 0) + 1
    print("\nBy class:")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {cls:>20}: {count:>5,}")

    # Keyword distribution
    kw_counts: dict[str, int] = {}
    for c in all_cards:
        for kw in c.keywords:
            kw_counts[kw] = kw_counts.get(kw, 0) + 1
    print("\nTop 20 keywords:")
    for kw, count in sorted(kw_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {kw:>20}: {count:>5,}")

    # Functional text coverage
    has_text = sum(1 for c in all_cards if c.functional_text)
    print(f"\nCards with functional text: {has_text:,} / {len(all_cards):,} "
          f"({100*has_text/len(all_cards):.0f}%)")

    # Value distribution
    above_rate = sum(1 for c in all_cards if c.best_use_value >= 3.0)
    print(f"Cards at or above rate (≥3.0): {above_rate:,} "
          f"({100*above_rate/len(all_cards):.0f}%)")

    # Pitch distribution
    pitch_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for c in all_cards:
        pitch_counts[c.pitch] = pitch_counts.get(c.pitch, 0) + 1
    print("\nBy pitch:")
    for p in [1, 2, 3, 0]:
        label = _PITCH_COLORS.get(p, str(p)) or "None"
        print(f"  {label:>8}: {pitch_counts.get(p, 0):>5,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Browse and search the card index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  card_lookup pouncing_qi_blue                    # Look up by ID
  card_lookup --search "go again"                 # Full-text search
  card_lookup --type AA --class NINJA             # Filter by type+class
  card_lookup --keyword combo                     # Find combo cards
  card_lookup --type AA --sort power --limit 10   # Top 10 by power
  card_lookup --stats                             # Index statistics
""",
    )
    parser.add_argument(
        "card_ids",
        nargs="*",
        help="Card IDs to look up (e.g. pouncing_qi_blue)",
    )
    parser.add_argument(
        "--search", "-s",
        help="Full-text search across name, description, and abilities",
    )
    parser.add_argument(
        "--type", "-t",
        dest="card_type",
        help="Filter by card type (AA, A, AR, DR, E, W, I, T, C)",
    )
    parser.add_argument(
        "--class", "-c",
        dest="card_class",
        help="Filter by class (NINJA, WARRIOR, BRUTE, etc.)",
    )
    parser.add_argument(
        "--talent",
        help="Filter by talent (SHADOW, ICE, LIGHTNING, etc.)",
    )
    parser.add_argument(
        "--keyword", "-k",
        help="Filter by keyword (combo, dominate, go_again, etc.)",
    )
    parser.add_argument(
        "--sort",
        choices=["power", "cost", "defense", "pitch", "attack_value",
                 "best_use_value", "block_willingness", "arsenal_value", "name"],
        help="Sort results by field (descending by default)",
    )
    parser.add_argument(
        "--asc",
        action="store_true",
        help="Sort ascending instead of descending",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=25,
        help="Maximum results to show (default: 25)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show index summary statistics",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Dump results as JSON (for piping to jq, etc.)",
    )
    parser.add_argument(
        "--brief", "-b",
        action="store_true",
        help="Compact table output instead of detailed cards",
    )
    parser.add_argument(
        "--index-dir",
        default=str(_DEFAULT_INDEX_DIR),
        help=f"Index directory (default: {_DEFAULT_INDEX_DIR})",
    )

    args = parser.parse_args()

    # Load index
    idx = CardIndex(persist_dir=args.index_dir)
    idx._ensure_cache()

    # Stats mode
    if args.stats:
        _print_stats(idx)
        return

    # Direct lookup mode
    if args.card_ids:
        cards = idx.lookup(args.card_ids)
        found_ids = {c.card_id for c in cards}
        for card_id in args.card_ids:
            if card_id not in found_ids:
                # Try fuzzy match
                matches = [
                    c for c in idx._cache.values()
                    if card_id.lower() in c.card_id.lower()
                       or card_id.lower() in c.name.lower()
                ]
                if matches:
                    print(f"'{card_id}' not found. Did you mean:")
                    for m in matches[:10]:
                        print(f"  - {m.card_id}  ({m.name})")
                    print()
                else:
                    print(f"'{card_id}' not found in index.\n")

        if args.dump:
            print(json.dumps([c.to_index_doc() for c in cards], indent=2))
        else:
            for c in cards:
                print(_format_card(c))
                print()
        return

    # Search / filter mode
    all_cards = list(idx._cache.values())

    # Apply filters
    if args.search:
        query = args.search.lower()
        all_cards = [
            c for c in all_cards
            if query in c.name.lower()
            or query in c.description.lower()
            or query in c.functional_text.lower()
            or query in c.card_id.lower()
        ]

    if args.card_type:
        all_cards = [c for c in all_cards if c.card_type == args.card_type.upper()]

    if args.card_class:
        all_cards = [
            c for c in all_cards
            if c.card_class.upper() == args.card_class.upper()
        ]

    if args.talent:
        all_cards = [
            c for c in all_cards
            if c.card_talent.upper() == args.talent.upper()
        ]

    if args.keyword:
        kw = args.keyword.lower().replace(" ", "_")
        all_cards = [c for c in all_cards if kw in c.keywords]

    # Sort
    if args.sort:
        sort_key = args.sort
        if sort_key == "name":
            all_cards.sort(key=lambda c: c.name, reverse=not args.asc)
        else:
            all_cards.sort(
                key=lambda c: getattr(c, sort_key, 0),
                reverse=not args.asc,
            )

    if not all_cards:
        print("No cards match your query.")
        return

    # Output
    if args.dump:
        displayed = all_cards[: args.limit]
        print(json.dumps([c.to_index_doc() for c in displayed], indent=2))
    elif args.brief or (not args.card_ids and not args.search and len(all_cards) > 5):
        _print_card_table(all_cards, args.limit)
    else:
        for c in all_cards[: args.limit]:
            print(_format_card(c))
            print()
        if len(all_cards) > args.limit:
            print(f"... {len(all_cards) - args.limit} more (use --limit to show more)")


if __name__ == "__main__":
    main()
