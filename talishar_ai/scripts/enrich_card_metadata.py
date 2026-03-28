"""
enrich_card_metadata.py — Enrich card_metadata.json with ability text from Fabrary.

Fetches the full card database from Fabrary's public CDN and merges
functionalText (rules text), legalHeroes, typeText, and other fields
into the existing card_metadata.json.

Usage
-----
    uv run python -m scripts.enrich_card_metadata
    uv run python -m scripts.enrich_card_metadata --output enriched_metadata.json
    uv run python -m scripts.enrich_card_metadata --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import requests

_APP_INFO_URL = "https://content.fabrary.net/info/app-info.json"
_CARDS_URL_TEMPLATE = "https://content.fabrary.net/info/cards-{version}.json"

_DEFAULT_METADATA_PATH = Path(__file__).resolve().parent.parent / "card_metadata.json"


def _load_metadata(path: Path) -> dict:
    """Load card_metadata.json, fixing known JSON issues."""
    content = path.read_text()
    # Fix stray characters after quoted values (e.g. "LIGHT",r)
    content = re.sub(r'"([^"]*)",([a-zA-Z])', r'"\1",', content)
    # Fix trailing commas before } or ]
    content = re.sub(r",(\s*[}\]])", r"\1", content)
    return json.loads(content)


def _fetch_fabrary_cards() -> dict[str, dict]:
    """Fetch all cards from Fabrary CDN. Returns {talishar_id: card_data}."""
    print("Fetching Fabrary app info...")
    info = requests.get(_APP_INFO_URL, timeout=10).json()
    version = info["latestCardsVersion"]
    print(f"  Cards version: {version}")

    url = _CARDS_URL_TEMPLATE.format(version=version)
    print(f"Fetching card database ({url})...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    cards = data["cards"]
    print(f"  Fetched {len(cards):,} cards from Fabrary")

    # Build lookup by Talishar-style ID (underscores instead of hyphens)
    lookup: dict[str, dict] = {}
    for card in cards:
        talishar_id = card["cardIdentifier"].replace("-", "_")
        lookup[talishar_id] = card
    return lookup


def _strip_markdown_bold(text: str) -> str:
    """Remove markdown bold markers from functional text."""
    return text.replace("**", "")


def enrich(
    metadata: dict,
    fabrary: dict[str, dict],
    *,
    verbose: bool = False,
) -> tuple[dict, dict[str, int]]:
    """Merge Fabrary data into metadata. Returns (enriched_metadata, stats)."""
    stats = {
        "matched": 0,
        "unmatched": 0,
        "added_functional_text": 0,
        "updated_stats": 0,
        "added_legal_heroes": 0,
        "added_type_text": 0,
    }

    for card_id, meta in metadata.items():
        fab = fabrary.get(card_id)
        if fab is None:
            stats["unmatched"] += 1
            if verbose:
                print(f"  MISS: {card_id}")
            continue

        stats["matched"] += 1

        # Add functional text (the key missing piece)
        func_text = fab.get("functionalText", "")
        if func_text:
            meta["functional_text"] = _strip_markdown_bold(func_text)
            stats["added_functional_text"] += 1

        # Add type text (e.g. "Ninja Equipment - Arms")
        type_text = fab.get("typeText", "")
        if type_text:
            meta["type_text"] = type_text
            stats["added_type_text"] += 1

        # Add legal heroes
        legal_heroes = fab.get("legalHeroes", [])
        if legal_heroes:
            meta["legal_heroes"] = legal_heroes
            stats["added_legal_heroes"] += 1

        # Add keywords from fabrary (may be more complete)
        fab_keywords = fab.get("keywords", [])
        if fab_keywords:
            # Merge with existing keywords, preferring fabrary's format
            existing_kw = set(meta.get("keywords", []))
            fab_kw_lower = {k.lower() for k in fab_keywords}
            # Keep fabrary keywords as authoritative
            meta["keywords"] = fab_keywords
            if existing_kw and existing_kw != fab_kw_lower:
                stats["updated_stats"] += 1

        # Correct stats from fabrary if present (they're authoritative)
        for field in ("cost", "power", "defense", "pitch"):
            fab_val = fab.get(field)
            if fab_val is not None:
                old_val = meta.get(field)
                if old_val != fab_val:
                    if verbose and old_val is not None:
                        print(f"  FIX {card_id}.{field}: {old_val} -> {fab_val}")
                    meta[field] = fab_val
                    stats["updated_stats"] += 1

        # Add arcane damage if present
        arcane = fab.get("arcane")
        if arcane is not None:
            meta["arcane"] = arcane

        # Add talents
        talents = fab.get("talents", [])
        if talents:
            meta["talents"] = talents

        # Add subtypes from fabrary
        subtypes = fab.get("subtypes", [])
        if subtypes:
            meta["subtypes_fab"] = subtypes

        # Add classes from fabrary
        classes = fab.get("classes", [])
        if classes:
            meta["classes_fab"] = classes

    return metadata, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich card_metadata.json with Fabrary ability text"
    )
    parser.add_argument(
        "--metadata",
        default=str(_DEFAULT_METADATA_PATH),
        help=f"Path to card_metadata.json (default: {_DEFAULT_METADATA_PATH})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: overwrite input file)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats without writing",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print per-card details",
    )
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    print(f"Loading metadata from {metadata_path}...")
    metadata = _load_metadata(metadata_path)
    print(f"  {len(metadata):,} cards loaded")

    fabrary = _fetch_fabrary_cards()

    print("\nEnriching...")
    enriched, stats = enrich(metadata, fabrary, verbose=args.verbose)

    print(f"\n--- Enrichment Results ---")
    print(f"  Matched:              {stats['matched']:,}")
    print(f"  Unmatched:            {stats['unmatched']:,}")
    print(f"  Added functional_text: {stats['added_functional_text']:,}")
    print(f"  Added type_text:      {stats['added_type_text']:,}")
    print(f"  Added legal_heroes:   {stats['added_legal_heroes']:,}")
    print(f"  Updated stats:        {stats['updated_stats']:,}")

    # Show a sample
    sample_id = "tearing_shuko"
    if sample_id in enriched:
        print(f"\n--- Sample: {sample_id} ---")
        print(json.dumps(enriched[sample_id], indent=2))

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    output_path = Path(args.output) if args.output else metadata_path
    print(f"\nWriting enriched metadata to {output_path}...")
    output_path.write_text(json.dumps(enriched, indent=2) + "\n")
    print(f"  Done! ({output_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
