"""
game_stats.py — Per-game statistics collected during play.

GameStatsCollector is attached to TalisharEnv and populates
info["game_stats"] at the end of each episode.

Tracked quantities
------------------
result           "win" | "loss" | "draw"
total_steps      number of decision steps taken by P1
damage_dealt     total health lost by the opponent
damage_taken     total health lost by P1
deck_remaining_p1  cards left in P1's deck at game end
deck_remaining_p2  cards left in P2's deck at game end
deck_out         True if either player ran out of cards
truncated        True if the episode hit max_steps (not a real terminal)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class GameStats:
    result:            str    # "win" | "loss" | "draw"
    total_steps:       int
    damage_dealt:      int    # total damage P1 dealt (opp health lost)
    damage_taken:      int    # total damage P1 took (own health lost)
    deck_remaining_p1: int    # P1 deck size at game end
    deck_remaining_p2: int    # P2 deck size at game end
    deck_out:          bool   # True if either deck hit 0
    truncated:         bool   # True if game was cut short by max_steps

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GameStatsCollector:
    """
    Lightweight accumulator attached to TalisharEnv.

    Call ``reset(state)`` at episode start and ``step(state)`` after each
    transition.  Call ``finalize(result, truncated)`` at episode end.
    """

    def __init__(self) -> None:
        self._start_my_hp:  int  = 20
        self._start_opp_hp: int  = 20
        self._steps:        int  = 0
        self._last_state:   dict = {}

    def reset(self, start_state: dict) -> None:
        """Record starting health values and zero step counter."""
        my  = start_state.get("myState",    {})
        opp = start_state.get("theirState", {})
        self._start_my_hp  = int(my.get("health",  20) or 20)
        self._start_opp_hp = int(opp.get("health", 20) or 20)
        self._steps        = 0
        self._last_state   = start_state

    def step(self, state: dict) -> None:
        """Call after every env.step() to update running state."""
        self._steps     += 1
        self._last_state = state

    def finalize(self, result: str, truncated: bool) -> GameStats:
        """
        Compute final stats from start and end states.

        Parameters
        ----------
        result:
            "win", "loss", or "draw" — from the env's perspective (P1).
        truncated:
            True if the episode ended by hitting max_steps, not a real terminal.
        """
        my  = self._last_state.get("myState",    {})
        opp = self._last_state.get("theirState", {})

        final_my_hp  = int(my.get("health",    0) or 0)
        final_opp_hp = int(opp.get("health",   0) or 0)
        p1_deck      = int(my.get("deckCount",  0) or 0)
        p2_deck      = int(opp.get("deckCount", 0) or 0)

        return GameStats(
            result            = result,
            total_steps       = self._steps,
            damage_dealt      = max(0, self._start_opp_hp - final_opp_hp),
            damage_taken      = max(0, self._start_my_hp  - final_my_hp),
            deck_remaining_p1 = p1_deck,
            deck_remaining_p2 = p2_deck,
            deck_out          = (p1_deck == 0 or p2_deck == 0),
            truncated         = truncated,
        )
