"""
fix_metadata_values.py — Compute missing on_hit_value, effect_value,
equipment_utility, and block_willingness from functional_text.

Parses the card's rules text to detect on-hit effects, go-again,
dominate, and equipment abilities, then fills in missing numeric
fields that the observation encoder relies on.

Usage
-----
    uv run python -m scripts.fix_metadata_values
    uv run python -m scripts.fix_metadata_values --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_METADATA_PATH = Path(__file__).resolve().parent.parent / "card_metadata.json"


def _compute_on_hit_value(text: str, card: dict) -> int:
    """Estimate on-hit value from functional text (0-5 scale)."""
    if not text:
        return 0
    t = text.lower()

    # No on-hit trigger
    if "when this hits" not in t and "if this hits" not in t and "on hit" not in t:
        return 0

    val = 0

    # Destroy effects (arsenal, aura, item) = high value
    if "destroy" in t:
        val = max(val, 4)
    # Discard effects
    if "discard" in t and ("random" in t or "their hand" in t):
        val = max(val, 3)
    # Draw effects
    if "draw" in t:
        val = max(val, 2)
    # Name/restrict effects (Censor-style)
    if "name a card" in t or "can't play" in t:
        val = max(val, 3)
    # Banish from deck (action advantage)
    if "banish" in t and "deck" in t:
        val = max(val, 2)
    # Create token
    if "create" in t:
        val = max(val, 2)
    # Gain resource / action point
    if "gain" in t and ("action point" in t or "resource" in t):
        val = max(val, 2)
    # Deal arcane damage
    if "arcane damage" in t:
        val = max(val, 2)
    # Generic on-hit with no recognized effect
    if val == 0:
        val = 1

    return val


def _compute_effect_value(text: str, card: dict) -> float:
    """Estimate effect_value for non-attack cards (equipment, auras, etc.)."""
    if not text:
        return 0.0
    t = text.lower()
    card_type = (card.get("type") or card.get("subtype", "")).upper()
    val = 0.0

    # Equipment abilities
    if card_type in ("E", "EQUIPMENT", "W", "WEAPON"):
        # Resource generation (Fyendal's Spring Tunic — "Gain {r}")
        if ("gain" in t and ("{r}" in t or "resource" in t)) or "generate" in t:
            val = max(val, 6.0)  # Recurring resource = very high
        # Draw cards (Mask of Momentum — recurring draw is strongest equipment effect)
        if "draw a card" in t or "draw 2" in t:
            val = max(val, 7.0)
        # Buff next attack (Tearing Shuko)
        if "gets" in t and ("+1" in t or "+2" in t or "+3" in t):
            val = max(val, 3.0)
        # Create token (Pouncing Paws → Crouching Tiger)
        if "crouching tiger" in t or "create" in t:
            val = max(val, 3.0)
        # Go again on weapon
        if "go again" in t:
            val = max(val, 2.0)
        # Generic equipment with an ability
        if val == 0 and len(t) > 20:
            val = 2.0

    # Auras with ongoing effects
    elif "aura" in card_type.lower() or card.get("subtype", "").lower() == "aura":
        if "ward" in t:
            val = max(val, float(card.get("wardAmount", 3)))
        elif len(t) > 20:
            val = max(val, 3.0)

    return val


def _fix_block_willingness(card: dict, text: str) -> int | None:
    """Fix block_willingness for equipment that's missing or wrong."""
    cid = card.get("_id", "")
    card_type = (card.get("type") or "").upper()

    # Only fix equipment
    if card_type not in ("E", "EQUIPMENT", "W", "WEAPON"):
        return None

    # Weapons should never block (defense = -1 typically)
    defense = int(card.get("defense", 0) or 0)
    if defense <= 0:
        return 0

    t = (text or "").lower()
    utility = int(card.get("equipment_utility", 0) or 0)

    # High-utility equipment: very reluctant to block
    if utility >= 8:
        return 1  # Fyendal's, Mask of Momentum — almost never block

    # Medium-utility: somewhat reluctant
    if utility >= 5:
        return 3

    # Low-utility or generic armor: willing to block
    if utility <= 2:
        return 7

    return None  # Don't change


def fix_metadata(meta: dict, dry_run: bool = False) -> dict:
    """Fix missing values in card_metadata.json."""
    stats = {
        "on_hit_fixed": 0,
        "effect_value_fixed": 0,
        "block_willingness_fixed": 0,
        "keywords_fixed": 0,
    }

    for cid, card in meta.items():
        text = card.get("functional_text", "")

        # Fix on_hit_value
        if "on_hit_value" not in card or card.get("on_hit_value") is None:
            val = _compute_on_hit_value(text, card)
            if val > 0:
                card["on_hit_value"] = val
                stats["on_hit_fixed"] += 1

        # Fix effect_value
        if "effect_value" not in card or card.get("effect_value") is None:
            val = _compute_effect_value(text, card)
            if val > 0:
                card["effect_value"] = val
                stats["effect_value_fixed"] += 1

        # Fix block_willingness
        bw = _fix_block_willingness(card, text)
        if bw is not None:
            old = card.get("block_willingness")
            if old is None or old == 0 or old == "MISSING":
                card["block_willingness"] = bw
                stats["block_willingness_fixed"] += 1

        # Fix keywords — ensure go-again is captured from text
        keywords = card.get("keywords") or []
        if isinstance(keywords, str):
            keywords = [keywords]
        t = (text or "").lower()
        if "go again" in t and "Go again" not in keywords and "goAgain" not in keywords:
            keywords.append("Go again")
            card["keywords"] = keywords
            stats["keywords_fixed"] += 1

    print(f"Fixed: {stats}")

    # Show key equipment values after fix
    for cid in ['fyendals_spring_tunic', 'mask_of_momentum', 'harmonized_kodachi',
                'pouncing_paws', 'tearing_shuko', 'command_and_conquer_red', 'censor_red']:
        c = meta.get(cid, {})
        print(f"  {cid}: on_hit={c.get('on_hit_value', '-')} effect={c.get('effect_value', '-')} "
              f"equip_util={c.get('equipment_utility', '-')} block_will={c.get('block_willingness', '-')}")

    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--input", default=str(_METADATA_PATH))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    path = Path(args.input)
    # Load with same fixer as enrich script
    content = path.read_text()
    content = re.sub(r'"([^"]*)",([a-zA-Z])', r'"\1",', content)
    content = re.sub(r",(\s*[}\]])", r"\1", content)
    meta = json.loads(content)

    print(f"Loaded {len(meta)} cards from {path}")
    meta = fix_metadata(meta, dry_run=args.dry_run)

    if not args.dry_run:
        out_path = Path(args.output) if args.output else path
        out_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        print(f"Wrote {out_path}")
    else:
        print("[DRY RUN] No files modified")


if __name__ == "__main__":
    main()
