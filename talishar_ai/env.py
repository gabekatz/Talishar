"""
env.py — Gymnasium-compatible environment wrapping the Talishar PHP engine.

One episode = one full game of Flesh and Blood.

Observation: float32 vector of shape (OBS_DIM,)  [see features.py]
Action:      integer in [0, MAX_ACTIONS)
             indexes into state["legalMoves"]; illegal slots must be masked.

Reward:
  terminal  +1  P1 wins
            -1  P2 wins / P1 loses
  per-step   shaped by health delta (see _shaped_reward)

Info dict keys:
  legal_mask   np.ndarray[MAX_ACTIONS] bool — which action slots are legal
  legal_moves  list[dict]              — full legalMoves from GetAIState
  raw_state    dict                    — last raw GetAIState response
  result       "win" | "loss" | None
"""

from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .features import StateEncoder, OBS_DIM, MAX_ACTIONS
from .game_manager import GameManager


class TalisharEnv(gym.Env):
    """
    Parameters
    ----------
    game_manager:
        Configured :class:`GameManager` instance pointing at the game server.
    p1_deck:
        Deck file name for player 1 (without .txt, must exist in Assets/).
    p2_deck:
        Deck file name for player 2.
    p2_is_ai:
        When True, the server-side EncounterAI drives P2 automatically after
        each P1 action.  When False, P2 must be driven externally (self-play).
    player_id:
        Which player this env controls (1 or 2).  Normally 1.
    shaped_reward_scale:
        Weight of the per-step health-delta reward component.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        game_manager: GameManager,
        p1_deck: str = "Ira",
        p2_deck: str = "Ira",
        p2_is_ai: bool = True,
        player_id: int = 1,
        shaped_reward_scale: float = 0.01,
    ) -> None:
        super().__init__()
        self.gm                  = game_manager
        self.p1_deck             = p1_deck
        self.p2_deck             = p2_deck
        self.p2_is_ai            = p2_is_ai
        self.player_id           = player_id
        self.shaped_reward_scale = shaped_reward_scale

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(MAX_ACTIONS)

        self._encoder    = StateEncoder()

        # Episode state (set in reset)
        self._game_name: str  = ""
        self._auth_key:  str  = ""
        self._p2_key:    str  = ""
        self._prev_my_health:   int = 20
        self._prev_opp_health:  int = 20
        self._last_state: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Core Gym API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        name, p1k, p2k = self.gm.create_game(
            p1_deck=self.p1_deck,
            p2_deck=self.p2_deck,
            p2_is_ai=self.p2_is_ai,
        )
        self._game_name = name
        self._auth_key  = p1k if self.player_id == 1 else p2k
        self._p2_key    = p2k

        state = self.gm.get_state_blocking(
            self._game_name, self.player_id, self._auth_key
        )
        self._last_state = state
        self._prev_my_health  = state.get("myState",    {}).get("health", 20)
        self._prev_opp_health = state.get("theirState", {}).get("health", 20)

        obs  = self._encoder.encode(state)
        info = self._make_info(state, result=None)
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        state = self._last_state
        legal = state.get("legalMoves", [])

        if action >= len(legal):
            raise ValueError(
                f"Action {action} out of range — only {len(legal)} legal moves."
            )

        move   = legal[action]
        params = move["params"]

        next_state = self.gm.submit_action(
            self._game_name, self.player_id, self._auth_key, params
        )

        # SubmitAIAction returns the updated state directly (P2 AI has already
        # moved server-side), but poll if we still don't have priority.
        if not next_state.get("havePriority") and not self._is_terminal(next_state):
            next_state = self.gm.get_state_blocking(
                self._game_name, self.player_id, self._auth_key
            )

        self._last_state = next_state

        terminated = self._is_terminal(next_state)
        reward     = self._compute_reward(next_state, terminated)

        # Update prev health for next step
        self._prev_my_health  = next_state.get("myState",    {}).get("health", 0)
        self._prev_opp_health = next_state.get("theirState", {}).get("health", 0)

        obs    = self._encoder.encode(next_state)
        result = self._result(next_state) if terminated else None
        info   = self._make_info(next_state, result=result)

        return obs, reward, terminated, False, info

    # ------------------------------------------------------------------
    # Action masking (for masked PPO)
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """Return bool mask over the MAX_ACTIONS action space."""
        return self._encoder.action_mask(self._last_state)

    # ------------------------------------------------------------------
    # Rendering (no-op)
    # ------------------------------------------------------------------

    def render(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_reward(self, state: dict, terminated: bool) -> float:
        my_hp  = state.get("myState",    {}).get("health", 0)
        opp_hp = state.get("theirState", {}).get("health", 0)

        if terminated:
            return self._terminal_reward(state)

        # Dense shaping: reward for dealing damage, penalty for taking it
        delta_opp = self._prev_opp_health - opp_hp   # positive = we dealt damage
        delta_my  = self._prev_my_health  - my_hp    # positive = we took damage
        shaped    = self.shaped_reward_scale * (delta_opp - delta_my)
        return float(np.clip(shaped, -1.0, 1.0))

    def _terminal_reward(self, state: dict) -> float:
        r = self._result(state)
        if r == "win":
            return 1.0
        if r == "loss":
            return -1.0
        return 0.0  # draw / unknown

    def _result(self, state: dict) -> str:
        """Return "win", "loss", or "draw"."""
        my_hp  = state.get("myState",    {}).get("health", 0)
        opp_hp = state.get("theirState", {}).get("health", 0)
        if opp_hp <= 0 and my_hp > 0:
            return "win"
        if my_hp <= 0 and opp_hp > 0:
            return "loss"
        # Fall back to turnPlayer heuristic if health deltas are ambiguous
        return "draw"

    @staticmethod
    def _is_terminal(state: dict) -> bool:
        phase = (state.get("phase") or {}).get("turnPhase", "")
        my_hp  = state.get("myState",    {}).get("health", 1)
        opp_hp = state.get("theirState", {}).get("health", 1)
        return phase == "OVER" or my_hp <= 0 or opp_hp <= 0

    def _make_info(self, state: dict, result: str | None) -> dict:
        return {
            "legal_mask":  self._encoder.action_mask(state),
            "legal_moves": state.get("legalMoves", []),
            "raw_state":   state,
            "result":      result,
        }