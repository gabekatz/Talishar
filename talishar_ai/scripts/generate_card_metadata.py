"""
scripts/generate_card_metadata.py — Generate strategic card metadata.

Stage 1: Parses GeneratedCardDictionaries.php to extract card stats, keywords,
         and equipment durability properties (battleworn, temper, bladebreak,
         guardwell, etc.).
Stage 2 (optional): Uses Claude API to assign strategic scores that capture
         game knowledge beyond raw stats — equipment utility, block willingness,
         on-hit values, offensive/defensive keyword value, token generation, and
         conditional cost awareness.

Usage
-----
# Stage 1 only (no API key needed)
python -m talishar_ai.scripts.generate_card_metadata

# Stage 1 + Stage 2 (requires ANTHROPIC_API_KEY env var)
python -m talishar_ai.scripts.generate_card_metadata --enrich

Output: talishar_ai/card_metadata.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_PHP_PATH = (
    Path(__file__).resolve().parents[2]
    / "GeneratedCode"
    / "GeneratedCardDictionaries.php"
)
_OUT_PATH = Path(__file__).resolve().parents[1] / "card_metadata.json"

# Regex to extract match() entries: "card_id" => "value",
# Handles both quoted values (including commas) and unquoted values.
_MATCH_QUOTED_RE = re.compile(r'^\s*"([^"]+)"\s*=>\s*"([^"]*)"')
_MATCH_UNQUOTED_RE = re.compile(r'^\s*"([^"]+)"\s*=>\s*([^,\s"]+)')


def _parse_match_block(php_text: str, func_name: str) -> dict[str, str]:
    """Extract key→value pairs from a PHP match() block inside a function."""
    mapping: dict[str, str] = {}
    in_func = False
    for line in php_text.splitlines():
        if f"function {func_name}" in line:
            in_func = True
            continue
        if in_func:
            if line.strip().startswith("};") or line.strip().startswith("default =>"):
                break
            # Try quoted first (handles commas in values like "Ira, Crimson Haze")
            m = _MATCH_QUOTED_RE.match(line)
            if not m:
                m = _MATCH_UNQUOTED_RE.match(line)
            if m:
                mapping[m.group(1)] = m.group(2)
    return mapping


def _parse_bool_match_block(php_text: str, func_name: str) -> set[str]:
    """Extract card IDs that return true from a PHP match() block."""
    ids: set[str] = set()
    in_func = False
    for line in php_text.splitlines():
        if f"function {func_name}" in line:
            in_func = True
            continue
        if in_func:
            if line.strip().startswith("};") or line.strip().startswith("default =>"):
                break
            if "=> true" in line.lower() or "=> 1" in line:
                for card_match in re.finditer(r'"([a-z][a-z0-9_]*)"', line):
                    ids.add(card_match.group(1))
    return ids


# ── Keyword definitions ─────────────────────────────────────────────────────
# Boolean keywords: function name → human-readable keyword name
_BOOL_KEYWORD_FUNCS: dict[str, str] = {
    # Combat keywords
    "GeneratedGoAgain":              "goAgain",
    "GeneratedHasAmbush":            "ambush",
    "GeneratedHasDominate":          "dominate",
    "GeneratedHasOverpower":         "overpower",
    "GeneratedHasReprise":           "reprise",
    "GeneratedHasCombo":             "combo",
    "GeneratedHasIntimidate":        "intimidate",
    "GeneratedHasChannel":           "channel",
    "GeneratedHasCrush":             "crush",
    "GeneratedHasPiercing":          "piercing",
    "GeneratedHasPhantasm":          "phantasm",
    "GeneratedHasSurge":             "surge",
    # Equipment durability
    "GeneratedHasBattleworn":        "battleworn",
    "GeneratedHasTemper":            "temper",
    "GeneratedHasBladeBreak":        "bladebreak",
    "GeneratedHasGuardwell":         "guardwell",
    # Talent / element
    "GeneratedHasSpectra":           "spectra",
    "GeneratedHasBloodDebt":         "bloodDebt",
    # Costs & conditions
    "GeneratedHasBoost":             "boost",
    "GeneratedHasCharge":            "charge",
    "GeneratedHasHeave":             "heave",
    # Defense / utility
    "GeneratedHasArcaneBarrier":     "arcaneBarrier",
    "GeneratedHasSpellvoid":         "spellvoid",
    "GeneratedHasWard":              "ward",
    # Token / resource generation
    "GeneratedHasRuneGate":          "runeGate",
    "GeneratedHasGalvanize":         "galvanize",
    # Misc strategic
    "GeneratedHasEphemeral":         "ephemeral",
    "GeneratedHasLegendary":         "legendary",
    "GeneratedHasContract":          "contract",
    "GeneratedHasClash":             "clash",
    "GeneratedHasFreeze":            "freeze",
    "GeneratedHasNegate":            "negate",
    "GeneratedHasStealth":           "stealth",
    "GeneratedHasSteal":             "steal",
    "GeneratedHasCloaked":           "cloaked",
    "GeneratedHasDecompose":         "decompose",
    "GeneratedHasMaterial":          "material",
    "GeneratedHasTranscend":         "transcend",
    "GeneratedHasRupture":           "rupture",
    "GeneratedHasSharpen":           "sharpen",
}

# Amount keywords: function name → (human-readable name, divisor for normalisation)
_AMOUNT_KEYWORD_FUNCS: dict[str, str] = {
    "GeneratedArcaneBarrierAmount":  "arcaneBarrierAmount",
    "GeneratedSpellvoidAmount":      "spellvoidAmount",
    "GeneratedQuellAmount":          "quellAmount",
    "GeneratedWardAmount":           "wardAmount",
    "GeneratedHeaveAmount":          "heaveAmount",
    "GeneratedOptAmount":            "optAmount",
    "GeneratedAmpAmount":            "ampAmount",
}


def parse_card_stats(php_path: Path) -> dict[str, dict]:
    """Parse card stats and keywords from GeneratedCardDictionaries.php."""
    print(f"[metadata] Reading {php_path}...")
    php_text = php_path.read_text(errors="replace")

    # Extract stat functions
    names    = _parse_match_block(php_text, "GeneratedCardName")
    types    = _parse_match_block(php_text, "GeneratedCardType")
    subtypes = _parse_match_block(php_text, "GeneratedCardSubtype")
    costs    = _parse_match_block(php_text, "GeneratedCardCost")
    powers   = _parse_match_block(php_text, "GeneratedPowerValue")
    defenses = _parse_match_block(php_text, "GeneratedBlockValue")
    pitches  = _parse_match_block(php_text, "GeneratedPitchValue")
    classes  = _parse_match_block(php_text, "GeneratedCardClass")
    talents  = _parse_match_block(php_text, "GeneratedCardTalent")

    # Boolean keywords
    keyword_sets: dict[str, set[str]] = {}
    for func_name, kw_name in _BOOL_KEYWORD_FUNCS.items():
        keyword_sets[kw_name] = _parse_bool_match_block(php_text, func_name)

    # Amount keywords
    amount_maps: dict[str, dict[str, str]] = {}
    for func_name, kw_name in _AMOUNT_KEYWORD_FUNCS.items():
        amount_maps[kw_name] = _parse_match_block(php_text, func_name)

    # Build card database
    all_ids = set(names.keys())
    cards: dict[str, dict] = {}

    for cid in sorted(all_ids):
        # GeneratedCardType defaults to "AA" for cards not explicitly listed
        card_type = types.get(cid, "AA")
        card: dict = {
            "name": names.get(cid, cid),
            "type": card_type,
            "subtype": subtypes.get(cid, ""),
            "class": classes.get(cid, ""),
            "talent": talents.get(cid, ""),
            "cost": _safe_int(costs.get(cid, "")),
            "power": _safe_int(powers.get(cid, "")),
            "defense": _safe_int(defenses.get(cid, "")),
            "pitch": _safe_int(pitches.get(cid, "")),
        }

        # Boolean keywords
        kws = []
        for kw, card_set in keyword_sets.items():
            if cid in card_set:
                kws.append(kw)
        if kws:
            card["keywords"] = kws

        # Amount keywords
        for kw_name, kw_map in amount_maps.items():
            if cid in kw_map:
                card[kw_name] = _safe_int(kw_map[cid])

        # Is this equipment?
        if card_type == "E":
            card["is_equipment"] = True

        cards[cid] = card

    n_equip = sum(1 for c in cards.values() if c.get("is_equipment"))
    n_kw = sum(1 for c in cards.values() if c.get("keywords"))
    print(f"[metadata] Parsed {len(cards):,} cards ({n_equip} equipment, {n_kw} with keywords)")
    return cards


def _safe_int(val: str) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


# ── Deterministic card valuation ────────────────────────────────────────────
#
# In FaB the "rate" for a card is 3: a card that attacks for 3, blocks for 3,
# or pitches for 3 is on-rate.  Anything below is under-rate; above is above-
# rate.  This lets us compute most card value algorithmically from raw stats
# and keywords — no LLM needed.

_RATE = 3  # baseline FaB card value

# Equipment block_willingness defaults by durability keyword.
# These are BASELINE scores; the model uses the live counter feature (slot 12)
# to adjust in-game (e.g. temper with 1 counter left → avoid blocking).
_EQUIP_BLOCK_DEFAULTS: dict[str, int] = {
    "guardwell":  9,   # blocking ADDS a counter — actively good to block with
    "battleworn": 5,   # loses a counter after blocking — moderate caution
    "temper":     4,   # loses a counter when blocking — cautious (esp. last counter)
    "bladebreak": 2,   # destroyed on block — avoid unless desperate
}
_EQUIP_BLOCK_NO_KEYWORD = 7  # standard equipment, no durability penalty


def compute_deterministic_values(cards: dict[str, dict]) -> dict[str, dict]:
    """
    Add deterministic card valuations based on FaB rate system.

    Adds:
      attack_value:      raw offensive value (power, adjusted by keywords)
      block_value:       raw defensive value (defense)
      pitch_value:       resource generation value (pitch)
      best_use_value:    max(attack_value, block_value, pitch_value)
      rate_delta:        best_use_value - 3 (above/below rate)
      block_willingness: equipment baseline (deterministic from keywords)
      arsenal_value:     how useful this card is when stored in arsenal (0-10)
                         Cards that can only block/pitch (resources, gems, pure
                         block cards) get 0 because those abilities require the
                         card to be in hand, not arsenal.
    """
    for cid, card in cards.items():
        power   = card.get("power", 0)
        defense = card.get("defense", 0)
        pitch   = card.get("pitch", 0)
        cost    = card.get("cost", 0)
        kws     = set(card.get("keywords", []))
        ctype   = card.get("type", "")

        # ── Attack value ──
        # Base = power.  Keyword adjustments for deterministic modifiers.
        atk_val = power
        if ctype in ("AA", "A"):
            # Go-again effectively doubles throughput when chaining — but we
            # can't encode that as a simple per-card modifier (it depends on
            # the rest of the hand).  Leave that to the hand planning features.
            # These are small per-card adjustments:
            if "dominate" in kws:
                atk_val += 1   # harder to block → effectively more damage
            if "overpower" in kws:
                atk_val += 1   # excess damage hits hero
            if "intimidate" in kws:
                atk_val += 1   # opponent loses a card before blocking
            if "phantasm" in kws:
                atk_val -= 1   # can be popped by 6+ power from hand
            if "crush" in kws:
                atk_val += 1   # conditional on-hit at 4+ damage
            if "piercing" in kws:
                atk_val += 1   # ignores arcane barrier
            if "bloodDebt" in kws:
                atk_val -= 1   # costs life to play

            # Net cost penalty: a card costing 3 to play needs 1 card pitched
            # (pitch 3 blue), effectively consuming 2 cards for 1 attack.
            # A 0-cost card is self-contained.
            if cost > 0:
                # Rough: each point of cost requires ~1/3 of a card pitched
                atk_val -= cost / _RATE

        card["attack_value"] = round(max(0, atk_val), 1)
        card["block_value"]  = defense
        card["pitch_value"]  = pitch

        best = max(card["attack_value"], defense, pitch)
        card["best_use_value"] = best
        card["rate_delta"]     = round(best - _RATE, 1)

        # ── Equipment block_willingness (deterministic) ──
        if card.get("is_equipment"):
            bw = _EQUIP_BLOCK_NO_KEYWORD
            for kw, default_bw in _EQUIP_BLOCK_DEFAULTS.items():
                if kw in kws:
                    bw = default_bw
                    break  # use the first matching durability keyword

            # Adjust by defense value: equipment with higher defense is more
            # worth blocking with (you get more value from the block action).
            if defense >= 3:
                bw = min(10, bw + 1)
            elif defense == 0:
                bw = max(0, bw - 2)  # 0-defense equipment — don't block with it

            card["block_willingness"] = bw

        # ── Arsenal value (deterministic) ──
        # In FaB you can only pitch and block from hand, NOT from arsenal.
        # A card in arsenal can only be PLAYED (as its type allows).  Cards
        # with no meaningful play effect (resources, gems, pure block cards)
        # are dead weight in arsenal.
        ars_val = 0
        if ctype == "AA":
            # Attack actions are the primary arsenal targets.
            # Scale by attack_value: a 6-power attack is more worth saving
            # than a 3-power one.
            ars_val = min(10, max(1, round(card["attack_value"] * 1.5)))
        elif ctype == "A":
            # Non-attack actions (utility: auras, life gain, card draw, etc.)
            # are always playable from arsenal.  These have varied utility
            # but can't be scored from power alone since most have power 0.
            # Use cost as a proxy: higher-cost actions tend to have more
            # impactful effects.  Floor at 4 since ANY playable card in
            # arsenal beats a dead resource.
            ars_val = min(10, max(4, cost + 3))
        elif ctype == "I":
            # Instants can be played from arsenal at instant speed —
            # valuable for disruption/defense.  Scale by power or flat 4
            # for utility instants.
            ars_val = min(10, max(3, power)) if power > 0 else 4
        elif ctype == "AR":
            # Attack reactions can pump from arsenal, but situational —
            # you need to be attacking and have a legal target.
            ars_val = min(8, max(2, power))
        elif ctype == "DR":
            # Defense reactions are playable from arsenal, but reactive.
            # Moderate value — better than dead but not ideal.
            ars_val = min(6, max(1, defense))
        elif ctype == "R":
            # Resources (gems) — zero arsenal value.  Can only pitch
            # from hand; in arsenal they're completely dead.
            ars_val = 0
        elif ctype == "T":
            # Tokens — not normally arsenaled.
            ars_val = 0
        elif ctype in ("E", "W"):
            # Equipment/weapons go in their own zones, not arsenal.
            ars_val = 0
        else:
            # Unknown types: if it has power or a play effect, some value.
            ars_val = min(5, power) if power > 0 else 0

        card["arsenal_value"] = ars_val

    n_above = sum(1 for c in cards.values() if c.get("rate_delta", 0) > 0)
    n_on    = sum(1 for c in cards.values() if c.get("rate_delta", 0) == 0)
    n_below = sum(1 for c in cards.values() if c.get("rate_delta", 0) < 0)
    print(f"[metadata] Rate analysis: {n_above} above rate, {n_on} on rate, {n_below} below rate")

    return cards


# ── LLM enrichment ──────────────────────────────────────────────────────────

_EQUIPMENT_SYSTEM_PROMPT = """\
You are a Flesh and Blood TCG expert. For each equipment card, rate its
strategic utility on a 0-10 scale based on how valuable it is to PRESERVE
throughout a game.  Focus on the card's ACTIVATED ABILITIES and TRIGGERED
EFFECTS — the basic stats (defense, durability keywords like battleworn/
temper/bladebreak/guardwell) are already handled separately.

Scoring guide:
- 10: Game-defining equipment enabling the hero's core strategy
      (e.g., Mask of Momentum for Ninja — draws cards on every hit)
- 7-9: High-value equipment with strong recurring effects or critical
       defensive utility (e.g., Nullrune Boots for arcane barrier in a
       Wizard-heavy meta, Crown of Seeds for constant life gain)
- 4-6: Solid equipment with meaningful triggered abilities or situational
       value beyond just blocking
- 1-3: Low-value equipment, mainly used for block value with no meaningful
       ongoing effect
- 0: Placeholder or flavor equipment with no strategic impact

Return JSON: {"card_id": {"equipment_utility": N}, ...}
Only include the card_id key and the equipment_utility field."""

_ATTACK_SYSTEM_PROMPT = """\
You are a Flesh and Blood TCG expert. For each attack card, evaluate properties
that CANNOT be derived from raw stats or keywords (those are handled separately).

Provide these fields:

1. **on_hit_value** (0-5): Value of the card's on-hit effect.
   - 0: No on-hit effect
   - 1: Minor (opponent loses 1 resource, minor disruption)
   - 2: Moderate (draw a card, deal 1 arcane damage)
   - 3: Strong (opponent discards a card, create a useful token)
   - 4: Very strong (opponent discards 2 cards, banish from deck)
   - 5: Game-changing (opponent skips next turn, massive disruption)

2. **has_on_hit** (bool): Whether this card has any on-hit trigger at all.

3. **conditional_cost** (string or null): Description of any conditional
   requirements that affect when/how this card can be played.  Focus on
   non-obvious requirements that aren't captured by keywords.
   Examples: "Requires boost tokens to have been used this turn",
   "Needs a contract attack on the chain", "Requires hero to have attacked
   this turn", "Must be played as second action".  null if no special conditions.

4. **token_generation** (0-3): Does this card generate tokens or create
   persistent value beyond its immediate attack?
   - 0: No token generation
   - 1: Minor (generates 1 runechant, creates a minor token)
   - 2: Moderate (generates agility tokens on discard, creates useful auras,
        generates 2-3 runechants)
   - 3: Major (creates multiple valuable tokens, significant persistent
        board state)

Return JSON: {"card_id": {"on_hit_value": N, "has_on_hit": bool,
"conditional_cost": "..." or null, "token_generation": N}, ...}
Only include cards that have at least one non-zero/non-null field.
Omit cards with no on-hit, no conditional cost, and no token generation."""

_WEAPON_SYSTEM_PROMPT = """\
You are a Flesh and Blood TCG expert. For each weapon card, rate its
strategic utility on a 0-10 scale based on how valuable its activated
ability is for the hero's game plan.

Scoring guide:
- 10: Core weapon that defines the hero's attack strategy and generates
      consistent value every turn (e.g., Harmonized Kodachi for Katsu —
      enables 2-weapon multi-attack turns)
- 7-9: Strong weapon with powerful triggered/activated abilities or
       excellent synergy with the hero's class (e.g., Dawnblade gaining
       +1 permanently on hit)
- 4-6: Solid weapon with meaningful but conditional value
- 1-3: Filler weapon, used mainly for chip damage with minimal special
       effects
- 0: Token/placeholder weapon

Return JSON: {"card_id": {"equipment_utility": N}, ...}
Only include the card_id key and the equipment_utility field."""

_REACTION_SYSTEM_PROMPT = """\
You are a Flesh and Blood TCG expert. For each attack reaction or defense
reaction card, evaluate properties that CANNOT be derived from raw stats
or keywords (those are handled separately).

Provide these fields:

1. **conditional_cost** (string or null): Description of any conditional
   requirements for playing this reaction. Focus on non-obvious
   requirements beyond the standard "must be reacting to an attack" rule.
   Examples: "Requires the attack to have hit", "Only if defending hero
   has less life", "Requires a Ninja attack on the chain",
   "Must have combo'd this combat chain".  null if no special conditions.

2. **token_generation** (0-3): Does this card generate tokens or persistent
   value beyond its immediate effect?
   - 0: No token generation
   - 1: Minor (1 runechant, minor token)
   - 2: Moderate (agility token, aura, 2-3 runechants)
   - 3: Major (multiple valuable tokens, significant persistent state)

3. **pump_value** (0-5): For attack reactions, how much additional power
   does this card grant beyond what's printed on the card? Include
   conditional bonuses at their typical expected value.
   - 0: No pump effect (card only blocks / has other utility)
   - 1-2: Minor conditional pump
   - 3-4: Solid pump with good conditions
   - 5: Exceptional pump (e.g., +5 power, or +3 with easy conditions)
   For defense reactions, set to 0.

Return JSON: {"card_id": {"conditional_cost": "..." or null,
"token_generation": N, "pump_value": N}, ...}
Only include cards that have at least one non-zero/non-null field."""

_ACTION_SYSTEM_PROMPT = """\
You are a Flesh and Blood TCG expert. For each NON-ATTACK action card
(type "A"), evaluate properties that CANNOT be derived from raw stats or
keywords.  These cards are utility/setup cards — they create auras, pump
subsequent attacks, generate tokens, draw cards, gain life, etc.

Provide these fields:

1. **conditional_cost** (string or null): Description of any conditional
   requirements that affect when/how this card can be played effectively.
   Examples: "Requires an aura in play", "Only useful if you attacked
   this turn", "Needs specific hero or class", "Requires pitch zone
   cards".  null if no special conditions.

2. **token_generation** (0-3): Does this card generate tokens, auras,
   or persistent board state?
   - 0: No persistent value (one-shot effect like life gain or draw)
   - 1: Minor (1 runechant, 1 minor aura, 1 item token)
   - 2: Moderate (creates a lasting aura with recurring effect, 2-3
        runechants, agility token, useful item)
   - 3: Major (creates multiple auras/tokens, transforms board state,
        major persistent advantage)

3. **pump_value** (0-5): Does this card buff subsequent attacks this turn?
   Include attack power increases, dominate/go-again grants, and cost
   reductions for attacks.
   - 0: No pump effect
   - 1-2: Minor pump (+1-2 power or minor keyword to next attack)
   - 3-4: Strong pump (+3-4 or valuable keyword like dominate)
   - 5: Exceptional pump (+5 or multiple strong keywords)

4. **disruption_value** (0-5): Does this card disrupt the opponent?
   - 0: No disruption (self-beneficial only)
   - 1: Minor (opponent loses 1 resource)
   - 2: Moderate (taxes opponent, minor hand disruption)
   - 3: Strong (opponent discards, destroys a permanent)
   - 4-5: Very strong to game-changing disruption

5. **effect_value** (0-10): The card's effective "play value" measured in
   the FaB rate system where 3 = on-rate.  This captures the total value
   generated when the card is PLAYED, considering all effects — token
   generation, hero-specific resource creation, pumps, life gain, card
   draw, etc.  Think of it as: "how many points of rate does playing
   this card generate?"
   Examples:
   - Deadwood Dirge (red): 3 (creates 3 runechants = 3 arcane damage)
   - A card that draws 2: 4-5 (card advantage is worth ~2 each)
   - Sigil of Solace (red, gain 3 life): 3 (3 life = on-rate)
   - A card creating 1 ash for Dromai: 2-3 (enables a dragon later)
   - A card adding soul for Light heroes: 2-4 (enables soul abilities)
   - A card with go-again that creates 2 runechants: 4 (2 from tokens
     + 2 from go-again enabling the next attack)
   - A pure pitch/block card with no play effect: 0

Return JSON: {"card_id": {"conditional_cost": "..." or null,
"token_generation": N, "pump_value": N, "disruption_value": N,
"effect_value": N}, ...}
Only include cards that have at least one non-zero/non-null field."""

_INSTANT_SYSTEM_PROMPT = """\
You are a Flesh and Blood TCG expert. For each instant card, evaluate
properties that CANNOT be derived from raw stats or keywords.

Provide these fields:

1. **conditional_cost** (string or null): Description of any conditional
   requirements for playing this instant effectively.
   Examples: "Requires runechants in play", "Only effective against
   arcane damage", "Needs allies in play", "Requires soul to function".
   null if no special conditions.

2. **token_generation** (0-3): Does this card generate tokens or
   persistent value?
   - 0: No token generation
   - 1: Minor (1 runechant, 1 token)
   - 2: Moderate (2-3 runechants, useful aura)
   - 3: Major (multiple tokens, significant board impact)

3. **disruption_value** (0-5): How much does this card disrupt the
   opponent's game plan when played effectively?
   - 0: No disruption (purely self-beneficial)
   - 1: Minor disruption (opponent loses 1 resource)
   - 2: Moderate (prevents 1 damage, minor tempo gain)
   - 3: Strong (negates an action, forces opponent to re-plan)
   - 4: Very strong (destroys opponent's permanent, prevents key ability)
   - 5: Game-changing (completely negates opponent's turn)

4. **effect_value** (0-10): The instant's effective "play value" measured
   in the FaB rate system (3 = on-rate).  Captures total value from all
   effects — arcane damage, life gain, card draw, token creation, etc.
   Examples:
   - Instant dealing 2 arcane damage: 2
   - Instant preventing 3 damage: 3
   - Instant drawing 1 card: 2-3
   - Instant with no meaningful effect beyond stats: 0

Return JSON: {"card_id": {"conditional_cost": "..." or null,
"token_generation": N, "disruption_value": N, "effect_value": N}, ...}
Only include cards that have at least one non-zero/non-null field."""


def enrich_with_llm(cards: dict[str, dict], batch_size: int = 50) -> dict[str, dict]:
    """
    Use Claude API to assign strategic metadata that can't be derived from
    raw stats or keywords.

    Adds to equipment (E):
      equipment_utility: 0-10 strategic preservation value (activated abilities)
    Adds to weapons (W):
      equipment_utility: 0-10 strategic value of the weapon's ability
    Adds to attack actions (AA):
      on_hit_value: 0-5 estimated value of on-hit effects
      has_on_hit: boolean
      conditional_cost: string description of conditional requirements
      token_generation: 0-3 token/persistent value generation
    Adds to non-attack actions (A):
      conditional_cost: string description of play conditions
      token_generation: 0-3 token/aura/persistent value generation
      pump_value: 0-5 buff to subsequent attacks
      disruption_value: 0-5 opponent disruption level
      effect_value: 0-10 total play value in FaB rate system (updates best_use_value)
    Adds to reactions (AR, DR):
      conditional_cost: string description of play conditions
      token_generation: 0-3 token/persistent value generation
      pump_value: 0-5 additional power granted (AR only)
    Adds to instants (I):
      conditional_cost: string description of play conditions
      token_generation: 0-3 token/persistent value generation
      disruption_value: 0-5 opponent disruption level

    Note: block_willingness, attack_value, and keyword modifiers are computed
    deterministically by compute_deterministic_values() — the LLM is only
    used for properties that require knowledge of card text/effects.
    """
    try:
        import anthropic
    except ImportError:
        print("[metadata] anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic()

    # Type-specific keys that indicate a card has already been LLM-enriched.
    # If a card has ANY of its type's keys, we skip it.
    _ENRICHED_KEYS = {
        "E":  ("equipment_utility",),
        "W":  ("equipment_utility",),
        "AA": ("has_on_hit", "on_hit_value", "conditional_cost", "token_generation"),
        "A":  ("effect_value", "pump_value", "disruption_value", "conditional_cost", "token_generation"),
        "AR": ("pump_value", "conditional_cost", "token_generation"),
        "DR": ("conditional_cost", "token_generation"),
        "I":  ("effect_value", "disruption_value", "conditional_cost", "token_generation"),
    }

    def _needs_enrichment(c: dict, ctype: str) -> bool:
        keys = _ENRICHED_KEYS.get(ctype, ())
        return not any(k in c for k in keys)

    equipment_cards = {cid: c for cid, c in cards.items()
                       if c.get("is_equipment") and _needs_enrichment(c, "E")}
    attack_cards    = {cid: c for cid, c in cards.items()
                       if c.get("type") == "AA" and _needs_enrichment(c, "AA")}
    action_cards    = {cid: c for cid, c in cards.items()
                       if c.get("type") == "A" and _needs_enrichment(c, "A")}
    weapon_cards    = {cid: c for cid, c in cards.items()
                       if c.get("type") == "W" and _needs_enrichment(c, "W")}
    reaction_cards  = {cid: c for cid, c in cards.items()
                       if c.get("type") in ("AR", "DR") and _needs_enrichment(c, c["type"])}
    instant_cards   = {cid: c for cid, c in cards.items()
                       if c.get("type") == "I" and _needs_enrichment(c, "I")}

    # Count already-enriched for reporting
    n_already = sum(
        1 for c in cards.values()
        if any(k in c for k in ("equipment_utility", "has_on_hit", "effect_value",
                                "pump_value", "disruption_value"))
    )
    if n_already:
        print(f"[metadata] Skipping {n_already} already-enriched cards")

    print(
        f"[metadata] Enriching via Claude API: "
        f"{len(equipment_cards)} equipment, {len(attack_cards)} attack actions, "
        f"{len(action_cards)} non-attack actions, {len(weapon_cards)} weapons, "
        f"{len(reaction_cards)} reactions, {len(instant_cards)} instants..."
    )

    # ── Equipment: utility only (block_willingness is deterministic) ──
    equipment_enriched = _enrich_batch(
        client, equipment_cards, _EQUIPMENT_SYSTEM_PROMPT,
        batch_size=batch_size,
    )
    for cid, data in equipment_enriched.items():
        if cid in cards:
            cards[cid]["equipment_utility"] = data.get("equipment_utility", 5)

    # ── Weapons: utility score (like equipment, they persist) ──
    weapon_enriched = _enrich_batch(
        client, weapon_cards, _WEAPON_SYSTEM_PROMPT,
        batch_size=batch_size,
    )
    for cid, data in weapon_enriched.items():
        if cid in cards:
            cards[cid]["equipment_utility"] = data.get("equipment_utility", 5)

    # ── Attack actions (AA): on-hit, conditions, tokens ──
    attack_enriched = _enrich_batch(
        client, attack_cards, _ATTACK_SYSTEM_PROMPT,
        batch_size=batch_size,
    )
    for cid, data in attack_enriched.items():
        if cid in cards:
            if data.get("has_on_hit"):
                cards[cid]["on_hit_value"] = data.get("on_hit_value", 0)
                cards[cid]["has_on_hit"]   = True
            if data.get("conditional_cost"):
                cards[cid]["conditional_cost"] = data["conditional_cost"]
            if data.get("token_generation", 0) > 0:
                cards[cid]["token_generation"] = data["token_generation"]

    # ── Non-attack actions (A): conditions, tokens, pumps, disruption, effect value ──
    action_enriched = _enrich_batch(
        client, action_cards, _ACTION_SYSTEM_PROMPT,
        batch_size=batch_size,
    )
    for cid, data in action_enriched.items():
        if cid in cards:
            if data.get("conditional_cost"):
                cards[cid]["conditional_cost"] = data["conditional_cost"]
            if data.get("token_generation", 0) > 0:
                cards[cid]["token_generation"] = data["token_generation"]
            if data.get("pump_value", 0) > 0:
                cards[cid]["pump_value"] = data["pump_value"]
            if data.get("disruption_value", 0) > 0:
                cards[cid]["disruption_value"] = data["disruption_value"]
            if data.get("effect_value", 0) > 0:
                ev = data["effect_value"]
                cards[cid]["effect_value"] = ev
                # Update best_use_value and rate_delta to reflect actual
                # play value — e.g. Deadwood Dirge red creates 3 runechants
                # so its effect_value=3 makes it on-rate, not below-rate.
                old_best = cards[cid].get("best_use_value", 0)
                if ev > old_best:
                    cards[cid]["best_use_value"] = ev
                    cards[cid]["rate_delta"] = round(ev - _RATE, 1)

    # ── Reactions (AR + DR): conditions, tokens, pump values ──
    reaction_enriched = _enrich_batch(
        client, reaction_cards, _REACTION_SYSTEM_PROMPT,
        batch_size=batch_size,
    )
    for cid, data in reaction_enriched.items():
        if cid in cards:
            if data.get("conditional_cost"):
                cards[cid]["conditional_cost"] = data["conditional_cost"]
            if data.get("token_generation", 0) > 0:
                cards[cid]["token_generation"] = data["token_generation"]
            if data.get("pump_value", 0) > 0:
                cards[cid]["pump_value"] = data["pump_value"]

    # ── Instants: conditions, tokens, disruption ──
    instant_enriched = _enrich_batch(
        client, instant_cards, _INSTANT_SYSTEM_PROMPT,
        batch_size=batch_size,
    )
    for cid, data in instant_enriched.items():
        if cid in cards:
            if data.get("conditional_cost"):
                cards[cid]["conditional_cost"] = data["conditional_cost"]
            if data.get("token_generation", 0) > 0:
                cards[cid]["token_generation"] = data["token_generation"]
            if data.get("disruption_value", 0) > 0:
                cards[cid]["disruption_value"] = data["disruption_value"]
            if data.get("effect_value", 0) > 0:
                ev = data["effect_value"]
                cards[cid]["effect_value"] = ev
                old_best = cards[cid].get("best_use_value", 0)
                if ev > old_best:
                    cards[cid]["best_use_value"] = ev
                    cards[cid]["rate_delta"] = round(ev - _RATE, 1)

    n_equip = sum(1 for c in cards.values() if "equipment_utility" in c)
    n_onhit = sum(1 for c in cards.values() if c.get("has_on_hit"))
    n_cond  = sum(1 for c in cards.values() if c.get("conditional_cost"))
    n_token = sum(1 for c in cards.values() if c.get("token_generation", 0) > 0)
    n_pump  = sum(1 for c in cards.values() if c.get("pump_value", 0) > 0)
    n_disrupt = sum(1 for c in cards.values() if c.get("disruption_value", 0) > 0)
    n_effect = sum(1 for c in cards.values() if c.get("effect_value", 0) > 0)
    print(
        f"[metadata] LLM enriched: {n_equip} equipment/weapon utility, "
        f"{n_onhit} on-hit, {n_cond} conditional costs, {n_token} token generators, "
        f"{n_pump} pumps, {n_disrupt} disruptive, {n_effect} with effect_value"
    )

    return cards


def _enrich_batch(
    client,
    card_subset: dict[str, dict],
    system_prompt: str,
    batch_size: int = 50,
) -> dict[str, dict]:
    """Send cards to Claude in batches and collect structured metadata."""
    all_results: dict[str, dict] = {}
    card_list = list(card_subset.items())

    for i in range(0, len(card_list), batch_size):
        batch = card_list[i : i + batch_size]
        batch_desc = {
            cid: {
                "name": c["name"],
                "type": c.get("type", ""),
                "subtype": c.get("subtype", ""),
                "class": c.get("class", ""),
                "talent": c.get("talent", ""),
                "cost": c.get("cost", 0),
                "power": c.get("power", 0),
                "defense": c.get("defense", 0),
                "pitch": c.get("pitch", 0),
                "keywords": c.get("keywords", []),
                **{k: c[k] for k in _AMOUNT_KEYWORD_FUNCS.values() if k in c},
            }
            for cid, c in batch
        }

        batch_num = i // batch_size + 1
        total_batches = (len(card_list) + batch_size - 1) // batch_size
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} cards)...", flush=True)

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": f"Analyze these Flesh and Blood cards:\n\n{json.dumps(batch_desc, indent=2)}",
                    }
                ],
                system=system_prompt,
            )

            # Extract JSON from response
            text = response.content[0].text
            json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            if json_match:
                text = json_match.group(1)
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                parsed = json.loads(text[json_start:json_end])
                all_results.update(parsed)
            else:
                print(f"    Warning: Could not parse JSON from response")

        except Exception as exc:
            print(f"    Error in batch {batch_num}: {exc}")

    return all_results


# ── Hero-conditioned metadata ─────────────────────────────────────────────

_HERO_ABILITIES_PATH = Path(__file__).resolve().parents[1] / "hero_abilities.json"

_HERO_ABILITY_PROMPT = """\
You are a Flesh and Blood TCG expert. For each hero card below, write their
hero ability text exactly as it appears on the card. Include the full rules
text — action costs, trigger conditions, and effects.

Use this format for each hero:
- Start with the timing/cost (e.g., "Once per Turn Action - 0:", "Once per Turn Instant - 0:", "Essence of Earth and Ice -")
- Then the full ability text
- If the hero has an innate passive (like Ira's weapon go-again), include it in parentheses

For heroes you are unsure about, write "UNKNOWN" so they can be manually filled in.

Return JSON: {"hero_id": {"ability": "full ability text", "intellect": N, "life": N}, ...}
- intellect: the hero's intellect value (hand size, typically 4)
- life: the hero's starting life total (young heroes typically 20, adult heroes vary)
"""


def generate_hero_abilities(
    cards: dict[str, dict],
    existing: dict[str, dict],
    batch_size: int = 30,
) -> dict[str, dict]:
    """
    Use Claude API to generate ability descriptions for all hero cards
    not already in hero_abilities.json.
    """
    try:
        import anthropic
    except ImportError:
        print("[metadata] anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic()

    # Find all hero cards missing from existing abilities
    heroes = {cid: c for cid, c in cards.items() if c.get("type") == "C"}
    missing = {cid: c for cid, c in heroes.items()
               if cid not in existing or not existing[cid].get("ability")}

    if not missing:
        print("[metadata] All heroes already have ability descriptions")
        return existing

    print(f"[metadata] Generating ability descriptions for {len(missing)} heroes...")

    card_list = list(missing.items())
    all_results: dict[str, dict] = {}

    for i in range(0, len(card_list), batch_size):
        batch = card_list[i : i + batch_size]
        batch_desc = {
            cid: {
                "name": c["name"],
                "class": c.get("class", ""),
                "talent": c.get("talent", ""),
                "subtype": c.get("subtype", ""),
            }
            for cid, c in batch
        }

        batch_num = i // batch_size + 1
        total_batches = (len(card_list) + batch_size - 1) // batch_size
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} heroes)...", flush=True)

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8192,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Write the hero ability text for each of these "
                            "Flesh and Blood heroes:\n\n"
                            + json.dumps(batch_desc, indent=2)
                        ),
                    }
                ],
                system=_HERO_ABILITY_PROMPT,
            )

            text = response.content[0].text
            json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            if json_match:
                text = json_match.group(1)
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                parsed = json.loads(text[json_start:json_end])
                all_results.update(parsed)
            else:
                print(f"    Warning: Could not parse JSON from response")

        except Exception as exc:
            print(f"    Error in batch {batch_num}: {exc}")

    # Merge with existing, preserving manually-written entries
    merged = dict(existing)
    n_new = 0
    n_unknown = 0
    for hero_id, data in all_results.items():
        if hero_id in merged and merged[hero_id].get("ability"):
            continue  # don't overwrite manual entries
        ability = data.get("ability", "UNKNOWN")
        if ability == "UNKNOWN":
            n_unknown += 1
        merged[hero_id] = data
        n_new += 1

    print(f"[metadata] Generated {n_new} hero abilities ({n_unknown} marked UNKNOWN)")
    return merged


_HERO_SYNERGY_PROMPT = """\
You are a Flesh and Blood TCG expert. You are evaluating cards specifically
for the hero **{hero_name}** ({hero_class}{hero_talent_str}).

Hero ability: {hero_ability}
{hero_extra}

For each card, rate its **hero_synergy** on a 0-10 scale. This measures how
much THIS SPECIFIC HERO's abilities change the card's value compared to a
generic hero of the same class.

Scoring guide:
- 0: No special synergy — card works at its base value for any hero.
- 1-3: Minor synergy — hero ability provides a small bonus (e.g., the card
  happens to trigger a minor hero effect, or benefits slightly from the
  hero's playstyle).
- 4-6: Significant synergy — this card meaningfully interacts with the
  hero's ability (e.g., a Combo card for Katsu who searches for Combos,
  a Runeblade non-attack action for Viserai who creates Runechants from
  them, a card with go-again for a hero that rewards multi-attack turns).
- 7-9: Core synergy — this card is a key part of the hero's strategy
  and gains substantial extra value from the hero's ability (e.g., weapon
  attacks for Dorinthea who chains weapon hits, Draconic cards for Fai
  who deals arcane damage per Draconic chain link).
- 10: Defining synergy — this card is essentially built for this hero
  and is dramatically more valuable here than anywhere else.

Return JSON: {{"card_id": {{"hero_synergy": N}}, ...}}
IMPORTANT: Only include cards with hero_synergy >= 1 (skip cards with 0).
Most generic cards will have 0 synergy — do NOT list them."""


def _compatible_cards(
    cards: dict[str, dict],
    hero_class: str,
    hero_talent: str,
) -> dict[str, dict]:
    """Filter cards to those playable by a hero with the given class/talent,
    AND that have mechanical properties which could interact with a hero ability.

    Cards that are purely stat-based (no keywords, no effects, no on-hit, no
    token generation, no special properties) get hero_synergy=0 by default —
    a vanilla 3/3 attack with no keywords plays identically for every hero.

    This dramatically reduces API costs for hero synergy scoring. Non-generic
    class/talent cards are always included since they were designed for specific
    heroes. Only generic "stat sticks" are filtered out.

    Handles multi-class heroes (e.g., "PIRATE,RANGER") by matching cards
    from ANY of the hero's classes.
    """
    playable_types = {"AA", "A", "AR", "DR", "I", "E", "W"}
    # Split multi-class into a set for matching
    hero_classes = set(c.strip() for c in hero_class.split(",") if c.strip())
    result: dict[str, dict] = {}
    for cid, c in cards.items():
        if c.get("type") not in playable_types:
            continue
        cclass = c.get("class", "")
        ctalent = c.get("talent", "")
        # Card is compatible if: class is generic, OR any of the card's
        # classes overlap with any of the hero's classes
        card_classes = set(cc.strip() for cc in cclass.split(",") if cc.strip())
        class_ok = (not card_classes or "GENERIC" in card_classes
                    or bool(card_classes & hero_classes))
        talent_ok = (ctalent == "" or ctalent == hero_talent)
        if not (class_ok and talent_ok):
            continue

        # Non-generic cards always included — they're designed for specific
        # heroes and likely have synergy worth evaluating.
        is_generic = (not card_classes or "GENERIC" in card_classes)
        if not is_generic:
            result[cid] = c
            continue

        # For generic cards, only include ones with mechanical properties
        # that could interact with a hero ability. Pure stat sticks
        # (attack for X, block for Y, no keywords/effects) play the same
        # for every hero.
        has_keywords = bool(c.get("keywords"))
        has_on_hit = c.get("has_on_hit", False)
        has_effects = c.get("effect_value", 0) > 0
        has_tokens = c.get("token_generation", 0) > 0
        has_pump = c.get("pump_value", 0) > 0
        has_disruption = c.get("disruption_value", 0) > 0
        has_conditional = bool(c.get("conditional_cost"))
        has_utility = c.get("equipment_utility", 5) != 5  # non-default utility
        has_go_again = c.get("goAgain", False)

        if any((has_keywords, has_on_hit, has_effects, has_tokens,
                has_pump, has_disruption, has_conditional, has_utility,
                has_go_again)):
            result[cid] = c

    return result


def enrich_hero_synergy(
    cards: dict[str, dict],
    hero_ids: list[str],
    batch_size: int = 80,
) -> dict[str, dict]:
    """
    Use Claude API to score hero-specific card synergies.

    For each hero in hero_ids, filters to compatible cards, sends them
    to the LLM with the hero's ability description, and stores results
    as cards[cid]["hero_scores"][hero_id] = {"hero_synergy": N}.

    Deduplication: heroes with identical (ability, class, talent) are
    evaluated only once.  This avoids redundant API calls for young/adult
    variants of the same hero that share an ability, while correctly
    separating heroes that share a name but have different abilities
    (e.g., Arakni, Solitary Confinement vs Arakni, Huntsman).

    Only cards compatible with the hero's class/talent are evaluated
    (e.g., no Pirate Ranger cards for Lexi).

    Requires hero_abilities.json to exist with ability descriptions.
    """
    try:
        import anthropic
    except ImportError:
        print("[metadata] anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)

    if not _HERO_ABILITIES_PATH.exists():
        print(f"[metadata] Hero abilities file not found: {_HERO_ABILITIES_PATH}")
        sys.exit(1)

    hero_abilities = json.loads(_HERO_ABILITIES_PATH.read_text())
    client = anthropic.Anthropic()

    # ── Dedup: group heroes by (ability_text, class, talent) ──
    # Heroes with identical ability + card pool share synergy scores.
    # This catches young/adult variants with the same ability.
    dedup_groups: dict[tuple, list[str]] = {}  # key → [hero_ids]
    dedup_info: dict[tuple, dict] = {}         # key → first hero's info

    for hero_id in hero_ids:
        if hero_id not in hero_abilities:
            print(f"[metadata] Warning: {hero_id} not in hero_abilities.json, skipping")
            continue

        hero_info = hero_abilities[hero_id]
        hero_card = cards.get(hero_id, {})
        hero_class = hero_card.get("class", "")
        hero_talent = hero_card.get("talent", "")
        ability_text = hero_info.get("ability", "").strip()

        dedup_key = (ability_text, hero_class, hero_talent)

        if dedup_key not in dedup_groups:
            dedup_groups[dedup_key] = []
            dedup_info[dedup_key] = {
                "hero_info": hero_info,
                "hero_class": hero_class,
                "hero_talent": hero_talent,
                "hero_name": hero_card.get("name", hero_id),
                "primary_id": hero_id,
            }
        dedup_groups[dedup_key].append(hero_id)

    # Report dedup savings
    total_requested = sum(len(g) for g in dedup_groups.values())
    n_unique = len(dedup_groups)
    if total_requested > n_unique:
        print(
            f"[metadata] Hero dedup: {total_requested} requested → "
            f"{n_unique} unique abilities (saved {total_requested - n_unique} API calls)"
        )

    # ── Evaluate each unique ability group, skipping already-scored ──
    n_skipped = 0
    for dedup_key, group_hero_ids in dedup_groups.items():
        info = dedup_info[dedup_key]
        hero_info = info["hero_info"]
        hero_class = info["hero_class"]
        hero_talent = info["hero_talent"]
        hero_name = info["hero_name"]
        primary_id = info["primary_id"]

        # Check if this hero group already has synergy scores in the data.
        # We check the primary hero — if it has hero_scores entries on any
        # cards, we consider this group already done.
        sample_scored = sum(
            1 for c in cards.values()
            if c.get("hero_scores", {}).get(primary_id)
        )
        if sample_scored > 0:
            n_skipped += 1
            continue

        # Show which heroes share this ability
        if len(group_hero_ids) > 1:
            print(
                f"[metadata] Hero: {hero_name} ({hero_class}"
                f"{'/' + hero_talent if hero_talent else ''}) "
                f"— shared by: {', '.join(group_hero_ids)}"
            )
        else:
            print(
                f"[metadata] Hero: {hero_name} ({hero_class}"
                f"{'/' + hero_talent if hero_talent else ''})"
            )

        compatible = _compatible_cards(cards, hero_class, hero_talent)
        print(f"[metadata]   {len(compatible)} compatible cards")

        hero_talent_str = f"/{hero_talent}" if hero_talent else ""
        hero_extra_parts = []
        if hero_info.get("intellect"):
            hero_extra_parts.append(f"Intellect (hand size): {hero_info['intellect']}")
        if hero_info.get("life"):
            hero_extra_parts.append(f"Life: {hero_info['life']}")
        hero_extra = "\n".join(hero_extra_parts)

        prompt = _HERO_SYNERGY_PROMPT.format(
            hero_name=hero_name,
            hero_class=hero_class,
            hero_talent_str=hero_talent_str,
            hero_ability=hero_info["ability"],
            hero_extra=hero_extra,
        )

        results = _enrich_batch(
            client, compatible, prompt,
            batch_size=batch_size,
        )

        # Store synergy scores for ALL heroes in this dedup group
        n_scored = 0
        for cid, data in results.items():
            if cid in cards:
                syn = data.get("hero_synergy", 0)
                if syn > 0:
                    if "hero_scores" not in cards[cid]:
                        cards[cid]["hero_scores"] = {}
                    for hid in group_hero_ids:
                        cards[cid]["hero_scores"][hid] = {"hero_synergy": syn}
                    n_scored += 1

        print(f"[metadata]   → {n_scored} cards with synergy >= 1")

    if n_skipped:
        print(f"[metadata] Skipped {n_skipped} hero groups (already scored)")

    return cards


def main():
    parser = argparse.ArgumentParser(
        description="Generate card_metadata.json for AI training."
    )
    parser.add_argument(
        "--php-path", default=str(_PHP_PATH),
        help="Path to GeneratedCardDictionaries.php",
    )
    parser.add_argument(
        "--out-path", default=str(_OUT_PATH),
        help="Output path for card_metadata.json",
    )
    parser.add_argument(
        "--enrich", action="store_true",
        help="Use Claude API to add strategic utility scores and on-hit values. "
             "Requires ANTHROPIC_API_KEY environment variable.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="Cards per API batch (default: 50).",
    )
    parser.add_argument(
        "--generate-hero-abilities", action="store_true",
        help="Use Claude API to auto-generate ability descriptions for all hero "
             "cards not yet in hero_abilities.json. Run this first, review the "
             "output, then use --heroes.",
    )
    parser.add_argument(
        "--heroes", type=str, default="",
        help="Hero-conditioned synergy scoring. Use 'all' to process every hero "
             "in hero_abilities.json, or a comma-separated list of hero IDs. "
             "Requires --enrich. Example: --heroes all  OR  --heroes ira_crimson_haze,katsu",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-enrich all cards even if they already have LLM metadata. "
             "Without this flag, already-enriched cards and heroes are skipped.",
    )
    args = parser.parse_args()

    # Load existing metadata so we can preserve LLM-enriched fields
    out = Path(args.out_path)
    existing_metadata: dict[str, dict] = {}
    if out.exists():
        existing_metadata = json.loads(out.read_text())
        print(f"[metadata] Loaded {len(existing_metadata):,} existing cards from {out}")

    # Stage 1: Parse PHP stats and keywords
    cards = parse_card_stats(Path(args.php_path))

    # Stage 2: Deterministic valuation (always runs — rate system + keywords)
    cards = compute_deterministic_values(cards)

    # Merge: layer LLM-enriched fields from existing metadata onto freshly
    # parsed cards. Deterministic fields are always recomputed (they're cheap
    # and PHP source may have changed), but LLM fields are preserved.
    # With --force, skip this to re-enrich everything from scratch.
    if not args.force and existing_metadata:
        _LLM_FIELDS = {
            "equipment_utility", "on_hit_value", "has_on_hit",
            "conditional_cost", "token_generation", "pump_value",
            "disruption_value", "effect_value", "hero_scores",
        }
        n_preserved = 0
        for cid, old in existing_metadata.items():
            if cid not in cards:
                continue
            for key in _LLM_FIELDS:
                if key in old and key not in cards[cid]:
                    cards[cid][key] = old[key]
                    n_preserved += 1
            # Restore effect_value → best_use_value feedback if applicable
            if "effect_value" in cards[cid]:
                ev = cards[cid]["effect_value"]
                if ev > cards[cid].get("best_use_value", 0):
                    cards[cid]["best_use_value"] = ev
                    cards[cid]["rate_delta"] = round(ev - _RATE, 1)
        if n_preserved:
            print(f"[metadata] Preserved {n_preserved} LLM-enriched fields from existing data")
    elif args.force and existing_metadata:
        print("[metadata] --force: ignoring existing LLM data, will re-enrich all cards")

    # Stage 2.5: Optional — auto-generate hero ability descriptions
    if args.generate_hero_abilities:
        existing_abilities: dict = {}
        if _HERO_ABILITIES_PATH.exists():
            existing_abilities = json.loads(_HERO_ABILITIES_PATH.read_text())
        updated = generate_hero_abilities(cards, existing_abilities, batch_size=30)
        # Remove _comment key if present before counting
        save_data = {k: v for k, v in updated.items() if not k.startswith("_")}
        save_data["_comment"] = (
            "Hero ability descriptions for hero-conditioned card metadata. "
            "Auto-generated entries can be reviewed and corrected manually."
        )
        with open(_HERO_ABILITIES_PATH, "w") as f:
            json.dump(save_data, f, indent=2)
        n_with_ability = sum(1 for k, v in save_data.items()
                            if not k.startswith("_") and v.get("ability", "UNKNOWN") != "UNKNOWN")
        print(f"[metadata] Wrote {n_with_ability} hero abilities to {_HERO_ABILITIES_PATH}")

    # Stage 3: Optional LLM enrichment (only for non-derivable properties)
    if args.enrich:
        cards = enrich_with_llm(cards, batch_size=args.batch_size)

    # Stage 4: Optional hero-conditioned synergy scoring
    if args.heroes:
        if not args.enrich:
            print("[metadata] Warning: --heroes requires --enrich (needs API key)")
            sys.exit(1)
        # Resolve hero list: "all" → every hero in hero_abilities.json
        if args.heroes.strip().lower() == "all":
            hero_abilities = json.loads(_HERO_ABILITIES_PATH.read_text())
            hero_ids = [k for k in hero_abilities.keys() if not k.startswith("_")]
            print(f"[metadata] --heroes all: {len(hero_ids)} heroes from hero_abilities.json")
        else:
            hero_ids = [h.strip() for h in args.heroes.split(",") if h.strip()]
        cards = enrich_hero_synergy(cards, hero_ids, batch_size=args.batch_size)

    # Write output
    with open(out, "w") as f:
        json.dump(cards, f, indent=2)
    print(f"[metadata] Wrote {len(cards):,} cards to {out}")


if __name__ == "__main__":
    main()
