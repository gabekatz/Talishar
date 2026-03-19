"""
features.py — Convert a GetAIState JSON response into a fixed-size numpy vector.

Observation layout (OBS_DIM = 409 floats, all normalised to roughly [0, 1]):

  Slots  0– 97  hand           7 cards × 14 features
  Slots 98–125  arsenal        2 cards × 14
  Slots 126–195 equipment      5 cards × 14
  Slots 196–265 auras          5 cards × 14
  Slots 266–335 items          5 cards × 14
  Slots 336–377 allies         3 cards × 14
  Slots 378–384 global scalars (7)
  Slots 385–392 phase one-hot  (8)
  Slots 393–397 combat chain   (5)
  Slot  398     stack size     (1)
  Slots 399–402 opponent scalars: health, deckCount, handCount, soulCount (4)
  Slots 403–408 aggregated hand stats (6):
    [403] total hand power   / 50   — sum of all hand card power values
    [404] total hand defense / 35   — sum of all hand card defense values
    [405] total hand pitch   / 21   — sum of all hand card pitch values
    [406] can_threaten_lethal       — 1 if total power ≥ opponent's remaining health
    [407] incoming_lethal_if_no_block — 1 if combat chain power ≥ my remaining health
    [408] survive_after_full_block  — 1 if I survive even after blocking with all hand cards
  ... total = 409

Card feature vector (14 floats):
  [0]  cost     / 10
  [1]  power    / 10
  [2]  defense  / 10
  [3]  pitch    / 3
  [4–11] type one-hot: AA, I, A, E, DR, R, T, C  (unknown → all-zeros)
  [12] counters / 5
  [13] tapped   (0 or 1)
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .card_vocab import CardVocab

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CARD_DIM = 14
MAX_HAND      = 7
MAX_ARSENAL   = 2
MAX_EQUIPMENT = 5
MAX_AURAS     = 5
MAX_ITEMS     = 5
MAX_ALLIES    = 3

ZONE_SIZES    = [MAX_HAND, MAX_ARSENAL, MAX_EQUIPMENT, MAX_AURAS, MAX_ITEMS, MAX_ALLIES]
ZONE_DIM      = sum(ZONE_SIZES) * CARD_DIM  # 378
N_CARD_SLOTS  = sum(ZONE_SIZES)             # 27 — one slot per card position across all zones

GLOBAL_DIM  = 7   # my health, their health, resources, ap, deck, their_deck, their_hand
PHASE_DIM   = 8   # one-hot phase
CC_DIM      = 5   # combat chain scalars
STACK_DIM   = 1
OPP_DIM     = 4   # opponent scalars (health, deck, hand, soul)
HAND_AGG_DIM = 6  # aggregated hand totals + lethal flags (see docstring)

OBS_DIM = ZONE_DIM + GLOBAL_DIM + PHASE_DIM + CC_DIM + STACK_DIM + OPP_DIM + HAND_AGG_DIM  # 409

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

def card_to_vec(card: dict[str, Any]) -> np.ndarray:
    """
    Encode a single card dict (from GetAIState) into a 14-float vector.

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

    return vec


def zone_to_block(cards: list[dict], max_slots: int) -> np.ndarray:
    """
    Encode a zone list into a (max_slots * CARD_DIM,) vector, zero-padded.
    """
    block = np.zeros(max_slots * CARD_DIM, dtype=np.float32)
    for i, card in enumerate(cards[:max_slots]):
        block[i * CARD_DIM : (i + 1) * CARD_DIM] = card_to_vec(card)
    return block


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

    def __init__(self, vocab: "CardVocab | None" = None) -> None:
        self._vocab = vocab

    def encode(self, state: dict[str, Any]) -> np.ndarray:
        """Return a float32 array of shape (OBS_DIM,)."""
        my  = state.get("myState",    {})
        opp = state.get("theirState", {})

        parts: list[np.ndarray] = []

        # -- Zones -----------------------------------------------------------
        for zone_key, max_slots in zip(_ZONE_KEYS, ZONE_SIZES):
            parts.append(zone_to_block(my.get(zone_key, []), max_slots))

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
        parts.append(np.array([
            min(_i(cc.get("totalPower",   0)), 15) / 15.0,
            min(_i(cc.get("totalDefense", 0)), 15) / 15.0,
            1.0 if cc.get("goAgain")   else 0.0,
            1.0 if cc.get("dominate")  else 0.0,
            1.0 if cc.get("piercing")  else 0.0,
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
        # These six numbers give the model pre-computed summaries of the hand
        # so it doesn't have to learn to sum across 7 individual card vectors.
        #
        # Think of it like giving a player a quick "hand report":
        #   • How much total damage can I threaten?
        #   • How much can I block with everything?
        #   • How much pitch do I have available?
        #   • Am I in a position to kill the opponent this attack?
        #   • Will I die if I don't block at all?
        #   • Can I survive by blocking with my full hand?
        #
        hand_cards = my.get("hand", [])

        def _hand_stat(card: dict, field: str) -> int:
            return int((card.get("stats") or {}).get(field, 0) or 0)

        total_hand_power   = sum(_hand_stat(c, "power")   for c in hand_cards)
        total_hand_defense = sum(_hand_stat(c, "defense") for c in hand_cards)
        total_hand_pitch   = sum(_hand_stat(c, "pitch")   for c in hand_cards)

        my_health  = _i(my.get("health",  0))
        opp_health = _i(opp.get("health", 0))
        cc_power   = _i((state.get("combatChain") or {}).get("totalPower", 0))

        # 1.0 if my hand's combined attack power is enough to kill the opponent
        # right now — signals "go for lethal" mode.
        can_threaten_lethal = 1.0 if (opp_health > 0 and total_hand_power >= opp_health) else 0.0

        # 1.0 if the current attack on the chain will kill me if I block nothing
        # — signals "must block or I die".
        incoming_lethal_if_no_block = 1.0 if (my_health > 0 and cc_power >= my_health) else 0.0

        # 1.0 if blocking with everything in hand still leaves me alive.
        # Formula: damage I actually take = max(0, cc_power - total_hand_defense).
        # If that's less than my health, I survive.  0.0 means even a full block
        # doesn't save me (should consider floating cards for counter-attack value).
        damage_through_full_block = max(0, cc_power - total_hand_defense)
        survive_after_full_block  = 1.0 if my_health - damage_through_full_block > 0 else 0.0

        parts.append(np.array([
            min(total_hand_power,   50) / 50.0,   # normalised: 7 cards × ~7 power
            min(total_hand_defense, 35) / 35.0,   # normalised: 7 cards × 5 defense
            min(total_hand_pitch,   21) / 21.0,   # normalised: 7 cards × 3 pitch
            can_threaten_lethal,
            incoming_lethal_if_no_block,
            survive_after_full_block,
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
