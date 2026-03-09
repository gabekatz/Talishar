"""
features.py — Convert a GetAIState JSON response into a fixed-size numpy vector.

Observation layout (OBS_DIM = 413 floats, all normalised to roughly [0, 1]):

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
  -------
  Slots 399–412 opponent scalars: health, deckCount, handCount, soulCount (4)
  ... total = 413

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

from typing import Any

import numpy as np

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

ZONE_SIZES = [MAX_HAND, MAX_ARSENAL, MAX_EQUIPMENT, MAX_AURAS, MAX_ITEMS, MAX_ALLIES]
ZONE_DIM   = sum(ZONE_SIZES) * CARD_DIM  # 378

GLOBAL_DIM  = 7   # my health, their health, resources, ap, deck, their_deck, their_hand
PHASE_DIM   = 8   # one-hot phase
CC_DIM      = 5   # combat chain scalars
STACK_DIM   = 1
OPP_DIM     = 4   # opponent scalars (health, deck, hand, soul)

OBS_DIM = ZONE_DIM + GLOBAL_DIM + PHASE_DIM + CC_DIM + STACK_DIM + OPP_DIM  # 399+14=413

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

    vec[0] = min(stats.get("cost",    0), 10) / 10.0
    vec[1] = min(stats.get("power",   0), 10) / 10.0
    vec[2] = min(stats.get("defense", 0), 10) / 10.0
    vec[3] = min(stats.get("pitch",   0),  3) /  3.0

    # Type one-hot — look at comma-separated primary type
    raw_type = (stats.get("type") or "").split(",")[0].strip()
    type_idx = _TYPE_INDEX.get(raw_type, 7)
    vec[4 + type_idx] = 1.0

    vec[12] = min(card.get("counters", 0), 5) / 5.0
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

class StateEncoder:
    """Stateless encoder — converts a GetAIState dict to a numpy observation."""

    def encode(self, state: dict[str, Any]) -> np.ndarray:
        """Return a float32 array of shape (OBS_DIM,)."""
        my  = state.get("myState",    {})
        opp = state.get("theirState", {})

        parts: list[np.ndarray] = []

        # -- Zones -----------------------------------------------------------
        for zone_key, max_slots in zip(
            ["hand", "arsenal", "equipment", "auras", "items", "allies"],
            ZONE_SIZES,
        ):
            parts.append(zone_to_block(my.get(zone_key, []), max_slots))

        # -- Global scalars --------------------------------------------------
        parts.append(np.array([
            min(my.get("health",    0),  40) / 40.0,
            min(opp.get("health",   0),  40) / 40.0,
            min(my.get("resources", 0),  10) / 10.0,
            min(my.get("ap",        0),   3) /  3.0,
            min(my.get("deckCount", 0),  80) / 80.0,
            min(opp.get("deckCount",0),  80) / 80.0,
            min(opp.get("handCount",0),   7) /  7.0,
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
            min(cc.get("totalPower",   0), 15) / 15.0,
            min(cc.get("totalDefense", 0), 15) / 15.0,
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
            min(opp.get("health",    0), 40) / 40.0,
            min(opp.get("deckCount", 0), 80) / 80.0,
            min(opp.get("handCount", 0),  7) /  7.0,
            min(opp.get("soulCount", 0),  5) /  5.0,
        ], dtype=np.float32))

        obs = np.concatenate(parts)
        assert obs.shape == (OBS_DIM,), f"Expected ({OBS_DIM},), got {obs.shape}"
        return obs

    def action_mask(self, state: dict[str, Any]) -> np.ndarray:
        """
        Return a bool array of shape (MAX_ACTIONS,).

        Index i is True if legalMoves[i] exists (i.e. that action slot is valid).
        """
        mask = np.zeros(MAX_ACTIONS, dtype=bool)
        n = min(len(state.get("legalMoves", [])), MAX_ACTIONS)
        mask[:n] = True
        return mask
