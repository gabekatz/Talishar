"""
features.py — Convert a GetAIState JSON response into a fixed-size numpy vector.

Observation layout (OBS_DIM = 535 floats, all normalised to roughly [0, 1]):

  Slots   0–125  hand           7 cards × 18 features
  Slots 126–161  arsenal        2 cards × 18
  Slots 162–251  equipment      5 cards × 18
  Slots 252–341  auras          5 cards × 18
  Slots 342–431  items          5 cards × 18
  Slots 432–485  allies         3 cards × 18
  Slots 486–492  global scalars (7)
  Slots 493–500  phase one-hot  (8)
  Slots 501–508  combat chain   (8)
  Slot  509      stack size     (1)
  Slots 510–513  opponent scalars: health, deckCount, handCount, soulCount (4)
  Slots 514–520  aggregated hand stats (7):
    [514] total hand power   / 50   — sum of all hand card power values
    [515] total hand defense / 35   — sum of all hand card defense values
    [516] total hand pitch   / 21   — sum of all hand card pitch values
    [517] can_threaten_lethal       — 1 if total power ≥ opponent's remaining health
    [518] incoming_lethal_if_no_block — 1 if combat chain power ≥ my remaining health
    [519] survive_after_full_block  — 1 if I survive even after blocking with all hand cards
    [520] can_prevent_on_hit        — 1 if total hand defense ≥ chain power (when on-hit active)
  Slots 521–527  hand attack plan (7):
    [521] best_attack_line / 20    — max damage from optimal attack sequence (costs + go-again)
    [522] attack_actions / 7       — count of AA-type cards in hand
    [523] go_again_sources / 3     — go-again from hand cards + aura tokens
    [524] surplus_cards / 7        — cards left over after best attack plan (free to block)
    [525] surplus_block_value / 20 — total defense of surplus cards
    [526] attack_reactions / 3     — instant/reaction pumps in hand
    [527] can_multi_attack         — 1 if can play 2+ attacks (has go-again + 2+ AAs + pitch)
  Slots 528–534  equipment & tempo (7):
    [528] turn number / 30          — game progression (capped at 30)
    [529] my equipment count / 5    — how many equipment pieces I still have
    [530] opp equipment count / 5   — how many equipment pieces opponent has
    [531] my equipment defense / 20 — total block value of remaining equipment
    [532] is_first_turn             — 1.0 on turn 0 (both players redraw to intellect)
    [533] is_defending              — 1.0 when in defense phase
    [534] hand_is_free              — 1.0 when turn 0 AND defending (hand cards are
                                      "free" to block with since both players redraw)
  Slots 535–542  pitch stack (8):
    [535] is_pdeck                  — 1.0 when in PDECK phase (pitch ordering decision)
    [536] deck_progress             — how far through the deck (1.0 = near second cycle)
    [537] n_stacked / 40            — total cards stacked at deck bottom this game
    [538] blue_ratio                — fraction of blues (pitch 3) in full stack
    [539] red_ratio                 — fraction of reds (pitch 1) in full stack
    [540] last_was_blue             — 1.0 if most recent stacked card was blue
    [541] last_was_red              — 1.0 if most recent stacked card was red
    [542] group_imbalance           — within current intellect-sized group, how unbalanced
                                      is blue vs red (0 = balanced, 1 = all same color)
  ... total = 543

Card feature vector (18 floats):
  [0]  cost     / 10
  [1]  power    / 10
  [2]  defense  / 10
  [3]  pitch    / 3
  [4–11] type one-hot: AA, I, A, E, DR, R, T, C  (unknown → all-zeros)
  [12] counters / 5
  [13] tapped   (0 or 1)
  [14] equipment_utility / 10  — strategic value of equipment (from card metadata)
  [15] block_willingness / 10  — how willing AI should be to block with this card
       (guardwell=high, battleworn=medium, bladebreak=low, temper=conditional)
  [16] arsenal_value / 10      — how useful this card is in arsenal (0 = dead weight,
       e.g. resources/gems can only pitch from hand; 10 = high-value attack to save)
  [17] hero_synergy / 10       — how much the active hero's ability enhances this card
       (0 = no special synergy; 10 = card defines the hero's strategy)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .card_vocab import CardVocab

_METADATA_PATH = Path(__file__).parent / "card_metadata.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CARD_DIM = 18
MAX_HAND      = 7
MAX_ARSENAL   = 2
MAX_EQUIPMENT = 5
MAX_AURAS     = 5
MAX_ITEMS     = 5
MAX_ALLIES    = 3

ZONE_SIZES    = [MAX_HAND, MAX_ARSENAL, MAX_EQUIPMENT, MAX_AURAS, MAX_ITEMS, MAX_ALLIES]
ZONE_DIM      = sum(ZONE_SIZES) * CARD_DIM  # 486
N_CARD_SLOTS  = sum(ZONE_SIZES)             # 27 — one slot per card position across all zones

GLOBAL_DIM  = 7   # my health, their health, resources, ap, deck, their_deck, their_hand
PHASE_DIM   = 8   # one-hot phase
CC_DIM      = 8   # combat chain scalars + on-hit features
STACK_DIM   = 1
OPP_DIM     = 4   # opponent scalars (health, deck, hand, soul)
HAND_AGG_DIM = 7  # aggregated hand totals + lethal flags + on-hit prevention
HAND_PLAN_DIM = 7  # attack line planning: damage, surplus, go-again, reactions
EQUIP_TEMPO_DIM = 7  # turn number, equipment counts, equipment defense, first turn, defending, hand free
PITCH_STACK_DIM = 8  # pitch stacking features for second-cycle planning

OBS_DIM = ZONE_DIM + GLOBAL_DIM + PHASE_DIM + CC_DIM + STACK_DIM + OPP_DIM + HAND_AGG_DIM + HAND_PLAN_DIM + EQUIP_TEMPO_DIM + PITCH_STACK_DIM  # 543

MAX_ACTIONS = 64

# Type string → one-hot index (index 7 = catch-all / unknown)
_TYPE_INDEX: dict[str, int] = {
    "AA": 0,
    "I":  1,
    "A":  2,
    "E":  3,
    "DR": 4,
    "R":  5,
    "T":  6,
    "C":  7,
}

_PHASES = ["M", "A", "D", "B", "P", "ARS", "OVER", "OTHER"]
_PHASE_INDEX: dict[str, int] = {p: i for i, p in enumerate(_PHASES)}


# ---------------------------------------------------------------------------
# Card encoding
# ---------------------------------------------------------------------------

def card_to_vec(
    card: dict[str, Any],
    metadata: dict[str, dict] | None = None,
    hero_id: str = "",
) -> np.ndarray:
    """
    Encode a single card dict (from GetAIState) into an 18-float vector.

    Cards with no stats (unknown IDs, blanks) return an all-zero vector.
    """
    vec = np.zeros(CARD_DIM, dtype=np.float32)
    stats = card.get("stats") or {}
    if not stats:
        return vec

    vec[0] = min(int(stats.get("cost",    0) or 0), 10) / 10.0
    vec[1] = min(int(stats.get("power",   0) or 0), 10) / 10.0
    vec[2] = min(int(stats.get("defense", 0) or 0), 10) / 10.0
    vec[3] = min(int(stats.get("pitch",   0) or 0),  3) /  3.0

    # Type one-hot — look at comma-separated primary type
    raw_type = (stats.get("type") or "").split(",")[0].strip()
    type_idx = _TYPE_INDEX.get(raw_type, 7)
    vec[4 + type_idx] = 1.0

    vec[12] = min(int(card.get("counters", 0) or 0), 5) / 5.0
    vec[13] = 1.0 if card.get("tapped") else 0.0

    # Strategic metadata from card_metadata.json (deterministic + LLM)
    if metadata:
        cid = card.get("cardID", "")
        meta = metadata.get(cid, {})
        vec[14] = min(int(meta.get("equipment_utility", 0)), 10) / 10.0
        vec[15] = min(int(meta.get("block_willingness", 0)), 10) / 10.0
        vec[16] = min(int(meta.get("arsenal_value", 0)), 10) / 10.0
        # Hero-conditioned synergy: how much the active hero's ability
        # enhances this card.  Looked up via hero_scores[hero_id].
        if hero_id:
            hero_scores = meta.get("hero_scores", {}).get(hero_id, {})
            vec[17] = min(int(hero_scores.get("hero_synergy", 0)), 10) / 10.0

    return vec


def zone_to_block(
    cards: list[dict],
    max_slots: int,
    metadata: dict[str, dict] | None = None,
    hero_id: str = "",
) -> np.ndarray:
    """
    Encode a zone list into a (max_slots * CARD_DIM,) vector, zero-padded.
    """
    block = np.zeros(max_slots * CARD_DIM, dtype=np.float32)
    for i, card in enumerate(cards[:max_slots]):
        block[i * CARD_DIM : (i + 1) * CARD_DIM] = card_to_vec(card, metadata, hero_id)
    return block


# ---------------------------------------------------------------------------
# Hand attack planning
# ---------------------------------------------------------------------------

def _compute_attack_plan(
    hand_cards: list[dict],
    auras: list[dict],
    resources: int,
    metadata: dict[str, dict] | None,
) -> dict[str, float]:
    """
    Compute the best attack line from the current hand.

    Greedy algorithm:
      1. Identify AA cards (attack actions) and their go-again status via metadata.
      2. Chain: play go-again AAs by power (descending), then the best
         non-go-again AA as a finisher.  External go-again (agility tokens)
         lets one extra non-ga attack chain.
      3. Deduct costs using current resources + pitch from remaining hand cards
         (best pitchers first).  If a card can't be afforded, drop it from
         the chain.
      4. Surplus = hand cards not consumed by the plan (available to block).

    Returns dict with: line_power, n_attacks, go_again_sources,
    surplus_cards, surplus_block, n_reactions, can_multi.
    """
    meta = metadata or {}

    # ── Parse hand cards ──
    # Each entry: (power, cost, defense, pitch, has_go_again, is_reaction, idx)
    attacks: list[tuple] = []      # AA cards
    reactions: list[tuple] = []    # Instant pumps (type I with power > 0)
    other_pitches: list[tuple] = []  # Non-attack cards (pitch, defense, idx)

    for i, card in enumerate(hand_cards):
        stats = card.get("stats") or {}
        cid = card.get("cardID", "") or ""
        cm = meta.get(cid, {})
        ctype = (stats.get("type") or "").split(",")[0].strip()
        pwr   = max(0, int(stats.get("power",   0) or 0))
        cost  = max(0, int(stats.get("cost",    0) or 0))
        defn  = max(0, int(stats.get("defense", 0) or 0))
        pitch = max(0, int(stats.get("pitch",   0) or 0))
        has_ga = "goAgain" in (cm.get("keywords") or [])

        if ctype == "AA":
            attacks.append((pwr, cost, defn, pitch, has_ga, i))
        elif ctype == "I" and pwr > 0:
            reactions.append((pwr, cost, defn, pitch, i))
        else:
            other_pitches.append((pitch, defn, i))

    # ── External go-again sources (agility tokens in auras) ──
    ext_ga = sum(
        1 for a in auras
        if "agility" in (a.get("cardID") or "").lower()
    )

    n_ga_hand = sum(1 for a in attacks if a[4])
    go_again_total = n_ga_hand + ext_ga

    # ── Build attack chain ──
    # Go-again AAs first (sorted by power desc), then best non-ga finisher.
    ga_attacks  = sorted([a for a in attacks if a[4]],     key=lambda x: -x[0])
    nga_attacks = sorted([a for a in attacks if not a[4]], key=lambda x: -x[0])

    chain: list[tuple] = list(ga_attacks)

    # After the go-again chain, we can add finishers:
    # - Each ga_attack grants go-again for the next attack
    # - External go-again (agility) grants one additional go-again
    finisher_slots = 0
    if chain:
        finisher_slots = 1  # last ga-attack grants go-again for one more
    if ext_ga > 0 and not chain:
        # Agility lets us play one non-ga attack, get go-again, then another
        if len(nga_attacks) >= 2:
            chain.append(nga_attacks[0])
            nga_attacks = nga_attacks[1:]
            finisher_slots = 1  # agility covers first, finisher follows
        elif nga_attacks:
            chain.append(nga_attacks[0])
            nga_attacks = []
            finisher_slots = 0
    elif ext_ga > 0 and chain:
        finisher_slots += 1  # agility gives one more on top of chain

    # Add finisher(s) from non-ga attacks
    for _ in range(finisher_slots):
        if nga_attacks:
            chain.append(nga_attacks[0])
            nga_attacks = nga_attacks[1:]

    # Fallback: if chain is still empty, play the best single attack
    if not chain and nga_attacks:
        chain.append(nga_attacks[0])
        nga_attacks = nga_attacks[1:]

    # ── Compute costs and affordability ──
    # Pitch pool: other_pitches + unused attacks + reactions
    used_indices = {a[5] for a in chain}  # indices of cards in the chain
    pitch_pool: list[tuple] = []  # (pitch_val, defense, idx)

    for pitch, defn, idx in other_pitches:
        pitch_pool.append((pitch, defn, idx))
    for a in attacks:
        if a[5] not in used_indices:
            pitch_pool.append((a[3], a[2], a[5]))  # pitch, defense, idx
    for r in reactions:
        pitch_pool.append((r[3], r[2], r[4]))  # pitch, defense, idx

    # Sort pitch pool: highest pitch first (use best pitchers)
    pitch_pool.sort(key=lambda x: -x[0])

    # Try to afford the full chain; drop from the end if we can't
    while chain:
        total_cost = sum(a[1] for a in chain)
        budget = resources
        pitched_indices: set[int] = set()

        for pv, pd, pidx in pitch_pool:
            if budget >= total_cost:
                break
            if pv > 0:
                budget += pv
                pitched_indices.add(pidx)

        if budget >= total_cost:
            break  # chain is affordable
        # Can't afford — drop the last (weakest) attack
        removed = chain.pop()
        # Return it to pitch pool
        pitch_pool.append((removed[3], removed[2], removed[5]))
        pitch_pool.sort(key=lambda x: -x[0])
        used_indices.discard(removed[5])

    # ── Recompute pitched cards for final chain ──
    total_cost = sum(a[1] for a in chain)
    budget = resources
    pitched_indices = set()
    for pv, pd, pidx in pitch_pool:
        if budget >= total_cost:
            break
        if pv > 0:
            budget += pv
            pitched_indices.add(pidx)

    # ── Results ──
    line_power = sum(a[0] for a in chain)
    cards_consumed = used_indices | pitched_indices
    surplus = max(0, len(hand_cards) - len(cards_consumed))

    # Surplus block value: defense of cards NOT consumed
    surplus_block = sum(
        max(0, int((hand_cards[i].get("stats") or {}).get("defense", 0) or 0))
        for i in range(len(hand_cards))
        if i not in cards_consumed
    )

    can_multi = 1.0 if len(chain) >= 2 else 0.0

    return {
        "line_power":       line_power,
        "n_attacks":        len(attacks),
        "go_again_sources": go_again_total,
        "surplus_cards":    surplus,
        "surplus_block":    surplus_block,
        "n_reactions":      len(reactions),
        "can_multi":        can_multi,
    }


# ---------------------------------------------------------------------------
# State encoder
# ---------------------------------------------------------------------------

_ZONE_KEYS = ["hand", "arsenal", "equipment", "auras", "items", "allies"]


class StateEncoder:
    """
    Converts a GetAIState dict to numpy arrays for the model.

    Parameters
    ----------
    vocab:
        Optional CardVocab for card-identity encoding.  When provided,
        ``card_ids()`` returns meaningful integer indices.  When None,
        ``card_ids()`` returns an all-zero array (PAD).
    """

    def __init__(self, vocab: "CardVocab | None" = None, hero_id: str = "") -> None:
        self._vocab = vocab
        self._hero_id = hero_id
        self._metadata: dict[str, dict] = {}
        if _METADATA_PATH.exists():
            try:
                self._metadata = json.loads(_METADATA_PATH.read_text())
            except Exception:
                pass

    def encode(self, state: dict[str, Any]) -> np.ndarray:
        """Return a float32 array of shape (OBS_DIM,)."""
        my  = state.get("myState",    {})
        opp = state.get("theirState", {})

        parts: list[np.ndarray] = []

        # -- Zones -----------------------------------------------------------
        meta = self._metadata or None
        # Resolve hero_id: use constructor value, or detect from character zone
        hero_id = self._hero_id
        if not hero_id:
            char_zone = my.get("character", [])
            if char_zone:
                hero_id = char_zone[0].get("cardID", "") or ""
        for zone_key, max_slots in zip(_ZONE_KEYS, ZONE_SIZES):
            parts.append(zone_to_block(my.get(zone_key, []), max_slots, meta, hero_id))

        # -- Global scalars --------------------------------------------------
        def _i(v): return int(v or 0)
        parts.append(np.array([
            min(_i(my.get("health",    0)),  40) / 40.0,
            min(_i(opp.get("health",   0)),  40) / 40.0,
            min(_i(my.get("resources", 0)),  10) / 10.0,
            min(_i(my.get("ap",        0)),   3) /  3.0,
            min(_i(my.get("deckCount", 0)),  80) / 80.0,
            min(_i(opp.get("deckCount",0)),  80) / 80.0,
            min(_i(opp.get("handCount",0)),   7) /  7.0,
        ], dtype=np.float32))

        # -- Phase one-hot ---------------------------------------------------
        phase_vec = np.zeros(PHASE_DIM, dtype=np.float32)
        raw_phase = (state.get("phase") or {}).get("turnPhase", "")
        phase_idx = _PHASE_INDEX.get(raw_phase, _PHASE_INDEX["OTHER"])
        phase_vec[phase_idx] = 1.0
        parts.append(phase_vec)

        # -- Combat chain ----------------------------------------------------
        cc = state.get("combatChain") or {}
        cc_power = _i(cc.get("totalPower", 0))

        # On-hit awareness: does the attacking card have an on-hit trigger,
        # and how valuable is it?  Uses both the live activeOnHits flag from
        # the game engine and the pre-computed on_hit_value from metadata.
        active_on_hits = 1.0 if cc.get("activeOnHits") else 0.0
        attacking_card_id = cc.get("attackingCard", "") or ""
        atk_meta = (self._metadata or {}).get(attacking_card_id, {})
        on_hit_val = int(atk_meta.get("on_hit_value", 0))
        # If metadata says there's an on-hit but engine flag is missing, trust metadata
        if on_hit_val > 0 and active_on_hits == 0.0:
            active_on_hits = 1.0

        parts.append(np.array([
            min(cc_power, 15) / 15.0,
            min(_i(cc.get("totalDefense", 0)), 15) / 15.0,
            1.0 if cc.get("goAgain")   else 0.0,
            1.0 if cc.get("dominate")  else 0.0,
            1.0 if cc.get("piercing")  else 0.0,
            active_on_hits,
            min(on_hit_val, 5) / 5.0,
            # Effective attack value: raw damage + on-hit value, normalised.
            # Tells the model the TRUE cost of letting this attack through.
            min(cc_power + on_hit_val, 20) / 20.0,
        ], dtype=np.float32))

        # -- Stack size ------------------------------------------------------
        stack = state.get("stack") or {}
        stack_size = len(stack.get("contents", [])) if stack else 0
        parts.append(np.array([min(stack_size, 5) / 5.0], dtype=np.float32))

        # -- Opponent extra scalars ------------------------------------------
        parts.append(np.array([
            min(_i(opp.get("health",    0)), 40) / 40.0,
            min(_i(opp.get("deckCount", 0)), 80) / 80.0,
            min(_i(opp.get("handCount", 0)),  7) /  7.0,
            min(_i(opp.get("soulCount", 0)),  5) /  5.0,
        ], dtype=np.float32))

        # -- Aggregated hand statistics --------------------------------------
        #
        # Seven numbers giving the model pre-computed summaries of the hand
        # so it doesn't have to learn to sum across 7 individual card vectors.
        #
        hand_cards = my.get("hand", [])

        def _hand_stat(card: dict, field: str) -> int:
            return int((card.get("stats") or {}).get(field, 0) or 0)

        total_hand_power   = sum(_hand_stat(c, "power")   for c in hand_cards)
        total_hand_defense = sum(_hand_stat(c, "defense") for c in hand_cards)
        total_hand_pitch   = sum(_hand_stat(c, "pitch")   for c in hand_cards)

        my_health  = _i(my.get("health",  0))
        opp_health = _i(opp.get("health", 0))

        can_threaten_lethal = 1.0 if (opp_health > 0 and total_hand_power >= opp_health) else 0.0
        incoming_lethal_if_no_block = 1.0 if (my_health > 0 and cc_power >= my_health) else 0.0

        damage_through_full_block = max(0, cc_power - total_hand_defense)
        survive_after_full_block  = 1.0 if my_health - damage_through_full_block > 0 else 0.0

        # 1.0 if my hand can fully block the incoming attack when there's an
        # active on-hit effect — signals "you CAN prevent the on-hit trigger
        # if you commit enough defense".  0.0 when no on-hit or not enough.
        can_prevent_on_hit = 0.0
        if active_on_hits > 0 and cc_power > 0:
            can_prevent_on_hit = 1.0 if total_hand_defense >= cc_power else 0.0

        parts.append(np.array([
            min(total_hand_power,   50) / 50.0,
            min(total_hand_defense, 35) / 35.0,
            min(total_hand_pitch,   21) / 21.0,
            can_threaten_lethal,
            incoming_lethal_if_no_block,
            survive_after_full_block,
            can_prevent_on_hit,
        ], dtype=np.float32))

        # -- Hand attack plan --------------------------------------------------
        #
        # Pre-computes the best attack sequence from the current hand,
        # accounting for go-again chaining, pitch costs, and external
        # go-again sources (agility tokens in auras).  This gives the model
        # a "hand report" so it knows how much damage it can threaten and
        # how many cards are surplus (free to block or arsenal).
        #
        plan = _compute_attack_plan(
            hand_cards, my.get("auras", []),
            int(my.get("resources", 0) or 0),
            self._metadata,
        )
        parts.append(np.array([
            min(plan["line_power"], 20)     / 20.0,
            min(plan["n_attacks"], 7)       /  7.0,
            min(plan["go_again_sources"], 3)/  3.0,
            min(plan["surplus_cards"], 7)   /  7.0,
            min(plan["surplus_block"], 20)  / 20.0,
            min(plan["n_reactions"], 3)     /  3.0,
            plan["can_multi"],
        ], dtype=np.float32))

        # -- Equipment & tempo --------------------------------------------------
        #
        # Turn number lets the model learn temporal strategies like "don't break
        # equipment on turn 0". Equipment counts let it value preservation —
        # the reward signal penalises losing equipment, and these features tell
        # the model how much is left.  Total equipment defense gives a sense of
        # how much block value equipment still provides.
        #
        turn_no = _i(state.get("turnNumber", 0))
        my_equip = my.get("equipment", [])
        opp_equip = opp.get("equipment", [])
        my_equip_count = len(my_equip)
        opp_equip_count = len(opp_equip)
        my_equip_defense = sum(
            int((c.get("stats") or {}).get("defense", 0) or 0)
            for c in my_equip
        )

        # Binary strategic signals:
        # - is_first_turn: turn 0 is unique — both players draw back to intellect,
        #   so hand cards used to block are "free" (will be replaced).
        # - is_defending: in defense phase, the key decision is what to block with.
        # - hand_is_free: conjunction of the above — a strong hint that hand cards
        #   should be preferred over equipment for blocking on turn 0.
        is_first_turn = 1.0 if turn_no == 0 else 0.0
        is_defending  = 1.0 if raw_phase == "D" else 0.0
        hand_is_free  = 1.0 if (turn_no == 0 and raw_phase == "D") else 0.0

        parts.append(np.array([
            min(turn_no, 30) / 30.0,
            min(my_equip_count,  5) / 5.0,
            min(opp_equip_count, 5) / 5.0,
            min(my_equip_defense, 20) / 20.0,
            is_first_turn,
            is_defending,
            hand_is_free,
        ], dtype=np.float32))

        # -- Pitch stack features -----------------------------------------------
        #
        # During PDECK phase, the model must choose which pitched card to place
        # on the deck bottom first.  For second-cycle planning, it needs to know
        # the recent color pattern of previously stacked cards.  The goal is to
        # interleave blues (pitch fuel) and reds (threats) so that second-cycle
        # hands are balanced — not 4 blues (no threats) or 4 reds (can't pay).
        #
        # Features:
        #   is_pdeck:           1.0 when in PDECK phase (pitch ordering decision)
        #   deck_progress:      how far through the deck (1.0 = approaching 2nd cycle)
        #   n_stacked / 40:     total cards stacked this game
        #   blue_ratio:         fraction of blues in the full pitch stack
        #   red_ratio:          fraction of reds in the full pitch stack
        #   last_was_blue:      previous card stacked was blue (pitch 3)
        #   last_was_red:       previous card stacked was red (pitch 1)
        #   group_imbalance:    within the current intellect-sized group, how
        #                       unbalanced is blue vs red (0 = balanced, 1 = all same)
        #
        pitch_stack = state.get("_pitch_stack", {})
        history = pitch_stack.get("history", [])
        starting_deck = pitch_stack.get("starting_deck_size", 60)
        deck_count = _i(my.get("deckCount", starting_deck))

        is_pdeck = 1.0 if raw_phase == "PDECK" else 0.0
        deck_progress = 1.0 - (min(deck_count, starting_deck) / max(starting_deck, 1))
        n_stacked = len(history)

        if n_stacked > 0:
            n_blue = sum(1 for p in history if p == 3)
            n_red  = sum(1 for p in history if p == 1)
            blue_ratio = n_blue / n_stacked
            red_ratio  = n_red / n_stacked
        else:
            blue_ratio = 0.0
            red_ratio  = 0.0

        last_was_blue = 1.0 if (history and history[-1] == 3) else 0.0
        last_was_red  = 1.0 if (history and history[-1] == 1) else 0.0

        # Group imbalance: within the current intellect-sized group being formed
        # at deck bottom, how skewed is the color ratio?  With intellect 4, we
        # look at (n_stacked % 4) recent cards.  If the group has 3 blues and
        # 0 reds, imbalance is high — the model should stack a red next.
        intellect = 4  # TODO: detect from hero card if varying
        group_pos = n_stacked % intellect  # position within current group (0-3)
        if group_pos > 0:
            recent = history[-group_pos:]
            grp_blue = sum(1 for p in recent if p == 3)
            grp_red  = sum(1 for p in recent if p == 1)
            # Imbalance: |blue - red| / group_pos, so 0 = balanced, 1 = all same
            group_imbalance = abs(grp_blue - grp_red) / group_pos
        else:
            group_imbalance = 0.0

        parts.append(np.array([
            is_pdeck,
            deck_progress,
            min(n_stacked, 40) / 40.0,
            blue_ratio,
            red_ratio,
            last_was_blue,
            last_was_red,
            group_imbalance,
        ], dtype=np.float32))

        obs = np.concatenate(parts)
        assert obs.shape == (OBS_DIM,), f"Expected ({OBS_DIM},), got {obs.shape}"
        return obs

    def card_ids(self, state: dict[str, Any]) -> np.ndarray:
        """
        Return an int64 array of shape (N_CARD_SLOTS,) with the vocabulary
        index for each card slot across all zones (in the same order as
        ``encode()``).

        Empty slots and unknown card IDs are encoded as PAD (0).
        If no CardVocab was provided at construction, returns all-zeros.
        """
        ids = np.zeros(N_CARD_SLOTS, dtype=np.int64)
        if self._vocab is None:
            return ids

        my   = state.get("myState", {})
        ptr  = 0
        for zone_key, max_slots in zip(_ZONE_KEYS, ZONE_SIZES):
            cards = my.get(zone_key, [])
            for i in range(max_slots):
                if i < len(cards):
                    cid = cards[i].get("cardID", "") or ""
                    ids[ptr] = self._vocab.encode(cid)
                ptr += 1
        return ids

    def encode_with_ids(
        self, state: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convenience wrapper: returns ``(obs_float, card_ids_int)``.

        Equivalent to calling ``encode()`` and ``card_ids()`` separately.
        """
        return self.encode(state), self.card_ids(state)

    def action_mask(self, state: dict[str, Any]) -> np.ndarray:
        """
        Return a bool array of shape (MAX_ACTIONS,).

        Index i is True if legalMoves[i] exists (i.e. that action slot is valid).
        """
        mask = np.zeros(MAX_ACTIONS, dtype=bool)
        n = min(len(state.get("legalMoves", [])), MAX_ACTIONS)
        mask[:n] = True
        return mask
