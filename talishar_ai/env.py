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

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import random

from .features import StateEncoder, OBS_DIM, MAX_ACTIONS
from .game_manager import GameManager
from .evaluation.game_stats import GameStatsCollector

_METADATA_PATH = Path(__file__).parent / "card_metadata.json"
_DEFAULT_EQUIP_UTILITY = 5  # mid-range default when metadata is unavailable


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
        equip_penalty_scale: float = 0.02,
        max_steps: int = 2000,
        deck_pool: list[str] | None = None,
        fill_from_inventory: bool = False,
        use_strategy_mask: bool = True,
    ) -> None:
        super().__init__()
        self.gm                  = game_manager
        self.p1_deck             = p1_deck
        self.p2_deck             = p2_deck
        self.p2_is_ai            = p2_is_ai
        self.player_id           = player_id
        self.shaped_reward_scale = shaped_reward_scale
        self.equip_penalty_scale = equip_penalty_scale
        self.max_steps           = max_steps
        self.deck_pool           = deck_pool
        self.fill_from_inventory = fill_from_inventory
        self.use_strategy_mask   = use_strategy_mask

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(MAX_ACTIONS)

        self._encoder    = StateEncoder()
        self._stats      = GameStatsCollector()

        # Card metadata for utility-weighted equipment penalties
        self._card_metadata: dict[str, dict] = {}
        if _METADATA_PATH.exists():
            try:
                self._card_metadata = json.loads(_METADATA_PATH.read_text())
            except Exception:
                pass

        # Episode state (set in reset)
        self._game_name: str  = ""
        self._auth_key:  str  = ""
        self._p2_key:    str  = ""
        self._hero_id:   str  = ""
        self._prev_phase: str = ""
        self._prev_turn_no: int = 0
        self._turn0_damage_dealt: int = 0  # cumulative damage dealt on turn 0
        self._turn0_was_offensive: bool = False  # were we the attacking player on turn 0?
        self._prev_my_health:   int = 20
        self._prev_opp_health:  int = 20
        self._prev_my_equip_ids:  list[str] = []
        self._prev_opp_equip_ids: list[str] = []
        self._last_state: dict[str, Any] = {}
        self._last_chosen_move: dict[str, Any] | None = None
        self._prev_hand_size: int = 0  # for stranded-hand detection
        self._prev_ap: int = 0  # previous action points
        self._attacks_this_turn: int = 0  # chain length tracker
        self._prev_turn_for_chain: int = -1  # reset chain counter on new turn
        self._prev_opp_hand_size: int = 4  # for on-hit opportunity detection
        self._prev_arsenal_size: int = 0  # for arsenal utilization tracking
        self._arsenaled_last_turn: bool = False  # did we arsenal last turn?
        self._step_count: int = 0

        # Pitch stack tracking: records the pitch values (1=red, 2=yellow, 3=blue)
        # of cards as they are placed on the deck bottom during PDECK.  This lets
        # the model learn to interleave colors for balanced second-cycle hands.
        self._pitch_history: list[int] = []
        self._starting_deck_size: int = 60

    # ------------------------------------------------------------------
    # Core Gym API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Pick random decks from pool if configured
        p1 = self.p1_deck
        p2 = self.p2_deck
        if self.deck_pool:
            p1 = random.choice(self.deck_pool)
            p2 = random.choice(self.deck_pool)

        name, p1k, p2k = self.gm.create_game(
            p1_deck=p1,
            p2_deck=p2,
            p2_is_ai=self.p2_is_ai,
            fill_from_inventory=self.fill_from_inventory,
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
        self._prev_my_equip_ids  = self._extract_equip_ids(state, "myState")
        self._prev_opp_equip_ids = self._extract_equip_ids(state, "theirState")
        # Detect hero ID for hero-aware equipment utility
        char_zone = state.get("myState", {}).get("character", [])
        self._hero_id = char_zone[0].get("cardID", "") if char_zone else ""
        self._prev_phase = (state.get("phase") or {}).get("turnPhase", "")
        self._prev_turn_no = 0
        self._turn0_damage_dealt = 0
        # On turn 0, if our first phase is NOT defense, we're the attacking player
        self._turn0_was_offensive = self._prev_phase != "D"
        self._step_count = 0
        self._last_chosen_move = None
        self._prev_hand_size = len(state.get("myState", {}).get("hand", []))
        self._prev_ap = int(state.get("myState", {}).get("ap", 0) or 0)
        self._attacks_this_turn = 0
        self._prev_turn_for_chain = 0
        self._prev_opp_hand_size = len(state.get("theirState", {}).get("hand", []))
        self._prev_arsenal_size = len(state.get("myState", {}).get("arsenal", []))
        self._arsenaled_last_turn = False
        self._pitch_history = []
        self._starting_deck_size = int(
            state.get("myState", {}).get("deckCount", 60) or 60
        )
        self._stats.reset(state)

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
        self._last_chosen_move = move

        # Track pitch stacking: when the AI places a card on deck bottom
        # during PDECK phase, record its pitch value for second-cycle planning.
        if move.get("type") == "PITCH_TO_DECK" or move.get("mode") == 6:
            card_id = move.get("cardID", "")
            pitch_val = int((move.get("stats") or {}).get("pitch", 0) or 0)
            if pitch_val == 0 and card_id:
                # Fallback: look up pitch from metadata
                meta = self._card_metadata.get(card_id, {})
                pitch_val = int(meta.get("pitch", 0) or 0)
            self._pitch_history.append(pitch_val)

        next_state = self.gm.submit_action(
            self._game_name, self.player_id, self._auth_key, params
        )

        # SubmitAIAction returns the updated state directly (P2 AI has already
        # moved server-side), but poll if we still don't have priority.
        if not next_state.get("havePriority") and not self._is_terminal(next_state):
            next_state = self.gm.get_state_blocking(
                self._game_name, self.player_id, self._auth_key
            )

        return self._finalize_step(next_state)

    def _finalize_step(
        self, next_state: dict[str, Any]
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Compute obs / reward / info from a fully-resolved next state.

        Separated from step() so SelfPlayEnv can inject a state that has
        already had P2's responses driven externally, bypassing the normal
        get_state_blocking() poll.
        """
        self._last_state = next_state
        self._step_count += 1
        self._stats.step(next_state)

        terminated = self._is_terminal(next_state)
        truncated  = (not terminated) and (self._step_count >= self.max_steps)
        reward     = self._compute_reward(next_state, terminated)

        # Update prev health and equipment for next step
        self._prev_my_health  = next_state.get("myState",    {}).get("health", 0)
        self._prev_opp_health = next_state.get("theirState", {}).get("health", 0)
        self._prev_my_equip_ids  = self._extract_equip_ids(next_state, "myState")
        self._prev_opp_equip_ids = self._extract_equip_ids(next_state, "theirState")
        self._prev_phase = (next_state.get("phase") or {}).get("turnPhase", "")

        # Inject pitch stack state so the encoder can build features from it
        next_state["_pitch_stack"] = {
            "history": self._pitch_history,
            "starting_deck_size": self._starting_deck_size,
        }

        obs    = self._encoder.encode(next_state)
        result = self._result(next_state) if (terminated or truncated) else None
        info   = self._make_info(next_state, result=result, truncated=truncated)

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Action masking (for masked PPO)
    # ------------------------------------------------------------------

    _PASS_MODE = 99

    def _strategy_mask(self, state: dict, base_mask: np.ndarray) -> np.ndarray:
        """Zero out legal-but-dominated actions.

        These are actions that are technically legal but never correct
        given the game state.  Hard-masking them removes the need for
        the model to learn these rules via reward shaping.

        Rules are conservative — only mask when the dominated-ness is
        unambiguous.  If unsure, leave the action legal and let the
        model decide.
        """
        moves = state.get("legalMoves", [])
        if not moves:
            return base_mask

        mask = base_mask.copy()
        phase = (state.get("phase") or {}).get("turnPhase", "")
        turn_no = int(state.get("turnNumber", 0) or 0)
        my = state.get("myState", {})
        equip_ids = {c.get("cardID", "") for c in my.get("equipment", [])}

        # -------------------------------------------------------
        # Rule 1: Turn-0 defense — block with ALL hand cards
        #
        # On turn 0, both players redraw to intellect after combat,
        # so blocking with hand cards is free.  Equipment lasts the
        # entire game.  Two sub-rules:
        #
        # 1a: If hand defense covers incoming damage, disallow
        #     equipment blocking and equipment activation.
        # 1b: If there's still unblocked damage AND hand cards
        #     are available to block with, disallow passing.
        #     Every hand card should be thrown in front — they
        #     cost nothing on turn 0.
        # -------------------------------------------------------
        if phase == "D" and turn_no == 0:
            hand_cards = my.get("hand", [])
            hand_defense = sum(
                int((c.get("stats") or {}).get("defense", 0) or 0)
                for c in hand_cards
            )
            cc = state.get("combatChain") or {}
            chain_power = int(cc.get("totalPower", 0) or 0)
            chain_defense = int(cc.get("totalDefense", 0) or 0)
            damage_remaining = chain_power - chain_defense

            # 1a: Mask equipment when hand cards suffice
            if hand_defense >= damage_remaining:
                for i, move in enumerate(moves[:len(mask)]):
                    if not mask[i]:
                        continue
                    mtype = move.get("type", "")
                    cid = move.get("cardID", "") or ""
                    if mtype == "ACTIVATE_EQUIPMENT":
                        mask[i] = False
                    elif mtype in ("CHOOSE_CARD", "CHOOSE_CARD_OPT") and cid in equip_ids:
                        mask[i] = False

            # 1b: Force full blocking — don't let the model pass
            # while damage is still coming and hand cards can block.
            # A hand card with defense > 0 that we haven't committed
            # yet means passing is dominated.
            has_hand_blocker = any(
                mask[i] and moves[i].get("type") in ("CHOOSE_CARD", "CHOOSE_CARD_OPT")
                and (moves[i].get("cardID", "") or "") not in equip_ids
                and int((moves[i].get("stats") or {}).get("defense", 0) or 0) > 0
                for i, _ in enumerate(moves[:len(mask)])
            )
            if damage_remaining > 0 and has_hand_blocker:
                for i, move in enumerate(moves[:len(mask)]):
                    if not mask[i]:
                        continue
                    if move.get("type") == "OK" or move.get("params", {}).get("mode") == self._PASS_MODE:
                        mask[i] = False

            if not mask.any():
                return base_mask

        # -------------------------------------------------------
        # Rule 2: Defense phase — no weapon/equipment activation
        #
        # Activating weapons or offensive equipment during the
        # opponent's attack does nothing useful (no attack to buff)
        # and destroys the equipment.  Always dominated by passing.
        #
        # Exception: some equipment has defensive activated abilities
        # (e.g. Fyendal's Spring Tunic gaining a resource).  We use
        # a simple heuristic: if the activation has power > 0, it's
        # offensive and should be masked.
        # -------------------------------------------------------
        if phase == "D":
            for i, move in enumerate(moves[:len(mask)]):
                if not mask[i]:
                    continue
                if move.get("type") != "ACTIVATE_EQUIPMENT":
                    continue
                stats = move.get("stats") or {}
                power = int(stats.get("power", 0) or 0)
                if power > 0:
                    mask[i] = False

            if not mask.any():
                return base_mask

        # -------------------------------------------------------
        # Rule 3: Arsenal before ending turn
        #
        # If the model is about to end its turn (pass/OK in main
        # or action phase), but ADD_TO_ARSENAL is available, mask
        # the pass.  Arsenaling a card for next turn is almost
        # always better than wasting it.
        #
        # Only applies when arsenal is empty (the model already
        # chose not to arsenal earlier if it's full).
        # -------------------------------------------------------
        if phase in ("M", "A"):
            arsenal = my.get("arsenal", [])
            has_arsenal_action = any(
                mask[i] and moves[i].get("type") == "ADD_TO_ARSENAL"
                for i in range(min(len(moves), len(mask)))
            )
            if has_arsenal_action and len(arsenal) == 0:
                for i, move in enumerate(moves[:len(mask)]):
                    if not mask[i]:
                        continue
                    if move.get("type") == "OK" or move.get("params", {}).get("mode") == self._PASS_MODE:
                        mask[i] = False

                if not mask.any():
                    return base_mask

        return mask

    def action_masks(self) -> np.ndarray:
        """Return bool mask over the MAX_ACTIONS action space."""
        base = self._encoder.action_mask(self._last_state)
        if self.use_strategy_mask:
            return self._strategy_mask(self._last_state, base)
        return base

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
        delta_opp = int(self._prev_opp_health) - int(opp_hp)   # positive = we dealt damage
        delta_my  = int(self._prev_my_health)  - int(my_hp)    # positive = we took damage

        # Turn-0 damage amplifier: on turn 0 defense, blocking with hand
        # cards is FREE (both players redraw to intellect).  Failing to
        # block is a strict mistake — amplify the damage-taken penalty so
        # the model learns to always block with hand cards on turn 0.
        damage_mult = 1.0
        turn_no = int(state.get("turnNumber", 0) or 0)
        if delta_my > 0 and self._prev_phase == "D" and self._prev_turn_no == 0:
            damage_mult = 5.0

        shaped = self.shaped_reward_scale * (delta_opp - delta_my * damage_mult)

        # -----------------------------------------------------------
        # Action-level equipment penalty: penalise the DECISION to use
        # equipment (block/activate), not the state change.  This
        # guarantees the penalty is on the exact step where the AI
        # chose the action, giving PPO a clean gradient signal.
        #
        # The old state-diff approach could misattribute penalties when
        # equipment disappeared on a different step than the decision
        # (e.g. during post-combat hit-effect processing) or when the
        # opponent's effects destroyed our equipment (penalising us for
        # something we didn't choose).
        # -----------------------------------------------------------
        if self.equip_penalty_scale > 0:
            shaped += self._action_equip_penalty(state)

            # State-diff: ONLY reward destroying opponent's equipment
            # (positive signal for good attacks — attribution doesn't
            # matter as much for rewards since any recent attacking
            # action contributed).
            curr_opp_ids = self._extract_equip_ids(state, "theirState")
            opp_lost_ids = list(
                (Counter(self._prev_opp_equip_ids) - Counter(curr_opp_ids)).elements()
            )
            if opp_lost_ids:
                opp_lost_utility = sum(self._equip_utility(cid) for cid in opp_lost_ids)
                shaped += self.equip_penalty_scale * opp_lost_utility

        # Turn-0 wasted aggression: if we were the offensive player on turn 0
        # and dealt zero damage the entire turn, we wasted our attack.  On
        # turn 0, hand cards are redrawn either way, so attacks that get fully
        # blocked have zero value — the model should have arsenaled a strong
        # card instead.  Signal fires once at the turn 0→1 transition.
        if delta_opp > 0 and self._prev_turn_no == 0:
            self._turn0_damage_dealt += delta_opp
        if turn_no > 0 and self._prev_turn_no == 0:
            if self._turn0_was_offensive and self._turn0_damage_dealt == 0:
                shaped -= self.shaped_reward_scale * 2.0

        # Stranded hand penalty: if the model played a card on its own turn
        # (offensive phase) and action points dropped to 0, but hand cards
        # remain that could have been played, it likely played a non-go-again
        # card before exhausting its attack chain.  This wastes potential
        # damage — the remaining hand cards are stranded.
        #
        # Detect: we were in action/main phase, had AP > 0, played a card,
        # and now AP = 0 with cards still in hand.
        curr_ap = int(state.get("myState", {}).get("ap", 0) or 0)
        curr_hand_size = len(state.get("myState", {}).get("hand", []))
        if (self._prev_phase in ("M", "A", "B")
                and self._prev_ap > 0
                and curr_ap == 0
                and curr_hand_size > 0
                and self._last_chosen_move
                and self._last_chosen_move.get("type", "") in (
                    "PLAY_CARD", "PLAY_ATTACK", "CHOOSE_CARD",
                )):
            # Penalty scales with stranded cards — more stranded = bigger mistake
            stranded = min(curr_hand_size, 3)
            shaped -= self.shaped_reward_scale * 1.5 * stranded

        # -----------------------------------------------------------
        # Attack chain length bonus: Ninja's power comes from chaining
        # many attacks via go-again.  Reward each consecutive attack in
        # a turn with a small diminishing bonus.  This teaches the model
        # to sequence go-again cards before closers and to activate
        # Kodachis as part of chains rather than in isolation.
        #
        # Bonus: 0.005 per chain link (3rd attack = 0.015 cumulative)
        # Capped at chain length 6 to avoid degenerate incentives.
        # -----------------------------------------------------------
        if turn_no != self._prev_turn_for_chain:
            self._attacks_this_turn = 0
            self._prev_turn_for_chain = turn_no

        played_attack = (
            self._last_chosen_move
            and self._last_chosen_move.get("type", "") in (
                "PLAY_CARD", "PLAY_ATTACK", "ACTIVATE_EQUIPMENT",
            )
            and delta_opp >= 0  # didn't somehow hurt us
            and self._prev_phase in ("M", "A", "B")
        )
        if played_attack:
            self._attacks_this_turn += 1
            if self._attacks_this_turn >= 2:
                # Diminishing bonus: 2nd attack = 0.5x, 3rd = 0.5x, ...
                chain_bonus = min(self._attacks_this_turn, 6) * 0.5
                shaped += self.shaped_reward_scale * chain_bonus

        # -----------------------------------------------------------
        # On-hit landing bonus: when we deal damage and the attacking
        # card has an on-hit effect, give extra reward.  On-hit effects
        # (like Command and Conquer destroying arsenal) are often worth
        # more than the raw damage.  This teaches the model to push
        # attacks with on-hit effects through and to go wide to exhaust
        # the opponent's blocks before the on-hit attack.
        # -----------------------------------------------------------
        if delta_opp > 0:
            cc = state.get("combatChain", {})
            atk_card = cc.get("attackingCard", "") or ""
            if atk_card:
                atk_meta = self._card_metadata.get(atk_card, {})
                on_hit_val = int(atk_meta.get("on_hit_value", 0))
                if on_hit_val > 0 and cc.get("activeOnHits"):
                    # On-hit landed — bonus proportional to on-hit value
                    shaped += self.shaped_reward_scale * min(on_hit_val, 5)

        # -----------------------------------------------------------
        # Arsenal utilization: reward the cycle of arsenal → play.
        # Arsenaling a good card and playing it next turn is a core
        # FaB pattern.  Penalize empty arsenal when the model had the
        # option to fill it (captures wasted tempo).
        # -----------------------------------------------------------
        curr_arsenal = state.get("myState", {}).get("arsenal", [])
        curr_arsenal_size = len(curr_arsenal)

        # Reward playing from arsenal (arsenal shrunk this step during our turn)
        if (self._prev_arsenal_size > 0
                and curr_arsenal_size < self._prev_arsenal_size
                and self._prev_phase in ("M", "A", "B")):
            shaped += self.shaped_reward_scale * 1.0  # played our arsenaled card

        # Track if we arsenaled this turn (for next turn's reward)
        if curr_arsenal_size > self._prev_arsenal_size:
            self._arsenaled_last_turn = True

        # -----------------------------------------------------------
        # Lethal awareness: when opponent is within kill range, amplify
        # the damage-dealt reward.  The model should recognise lethal
        # opportunities and go all-in rather than playing conservatively.
        # Also reduce damage-taken penalty when pushing lethal — trading
        # HP for damage is correct when you can close the game.
        # -----------------------------------------------------------
        opp_hp_int = int(opp_hp)
        if opp_hp_int > 0 and opp_hp_int <= 10 and delta_opp > 0:
            # Amplify damage reward when opponent is in lethal range
            lethal_bonus = (11 - opp_hp_int) / 10.0  # 1.0 at 1hp, 0.1 at 10hp
            shaped += self.shaped_reward_scale * delta_opp * lethal_bonus * 2.0

        # Update tracking
        self._prev_turn_no = turn_no
        self._prev_hand_size = curr_hand_size
        self._prev_ap = curr_ap
        self._prev_opp_hand_size = len(state.get("theirState", {}).get("hand", []))
        self._prev_arsenal_size = curr_arsenal_size

        return float(np.clip(shaped, -1.0, 1.0))

    def _action_equip_penalty(self, state: dict) -> float:
        """Penalty for the AI's CHOSEN action if it uses/destroys equipment.

        Fires on the exact step where the decision was made, ensuring
        correct credit assignment for PPO.  Covers:
        - Blocking with equipment (CHOOSE_CARD for an equipment card)
        - Activating equipment (ACTIVATE_EQUIPMENT)

        The penalty is scaled by card utility, turn number, and phase.

        Extra penalty when blocking with equipment while hand cards are
        still available — the model should exhaust hand cards first,
        especially on turn 0 where hand cards are free (redrawn).
        """
        move = self._last_chosen_move
        if not move:
            return 0.0

        action_type = move.get("type", "")
        card_id = move.get("cardID", "") or ""
        if not card_id:
            return 0.0

        # Detect equipment-consuming actions
        is_equip_activate = action_type == "ACTIVATE_EQUIPMENT"
        is_equip_block = (
            action_type in ("CHOOSE_CARD", "CHOOSE_CARD_OPT")
            and card_id in self._prev_my_equip_ids
        )

        if not (is_equip_activate or is_equip_block):
            return 0.0

        utility = self._equip_utility(card_id)
        turn_no = int(state.get("turnNumber", 0) or 0)

        # Turn-based multiplier
        if turn_no == 0:
            early_mult = 5.0
        else:
            early_mult = max(1.0, 3.0 - turn_no / 7.5)

        # Defense-phase multiplier: using equipment while defending is
        # almost always wasteful.  Use _prev_phase (the phase when the
        # AI had priority and chose the action) for reliable detection
        # even during post-combat hit-effect processing.
        if self._prev_phase == "D":
            early_mult *= 2.0

        penalty = -self.equip_penalty_scale * utility * early_mult

        # ---------------------------------------------------------------
        # Health-context scaling: equipment should only be sacrificed when
        # the damage is life-threatening.  At 30 HP, 6 damage is trivial
        # and never worth losing Fyendal's Spring Tunic or Mask of
        # Momentum.  At 5 HP, 6 damage is lethal — blocking is correct.
        #
        # Also factor in opponent threat: empty opponent hand means no
        # follow-up damage, so the isolated hit is even less threatening.
        # ---------------------------------------------------------------
        if is_equip_block and self._prev_phase == "D":
            my_hp = int(state.get("myState", {}).get("health", 20) or 20)
            opp_hand = len(state.get("theirState", {}).get("hand", []))

            # Estimate incoming damage from combat chain
            cc = state.get("combatChain", {})
            incoming = int(cc.get("totalAttack", 0) or 0) - int(cc.get("totalBlock", 0) or 0)
            incoming = max(incoming, 0)

            is_lethal = incoming >= my_hp
            # "Near-lethal" = damage would put us at <=5 HP
            is_near_lethal = (my_hp - incoming) <= 5

            if is_lethal:
                # Blocking to survive is correct — reduce penalty significantly
                # (small residual penalty so model still prefers hand cards first)
                penalty *= 0.1
            elif is_near_lethal:
                # Borderline — mild penalty, model can learn the nuance
                penalty *= 0.5
            else:
                # Not threatening — amplify penalty based on health cushion.
                # More health = more wasteful to sacrifice equipment.
                health_ratio = min(my_hp / 20.0, 2.0)  # 1.0 at 20hp, 1.5 at 30hp
                penalty *= health_ratio

                # Opponent empty hand = no follow-up threat, even less reason
                # to panic-block with equipment.
                if opp_hand == 0:
                    penalty *= 2.0

        # Hand-cards-available penalty: if blocking with equipment while
        # hand cards are still available, apply a steep extra penalty.
        # Hand cards should ALWAYS be exhausted before equipment,
        # especially on turn 0 where hand cards are free (redrawn).
        if is_equip_block and self._prev_phase == "D":
            hand_size = len(state.get("myState", {}).get("hand", []))
            if hand_size > 0:
                # Scale by hand size — more cards available = worse mistake
                hand_mult = min(hand_size, 4)  # cap at 4
                if turn_no == 0:
                    # On turn 0, hand cards are FREE. This is never correct.
                    penalty += -self.equip_penalty_scale * utility * 10.0 * hand_mult
                else:
                    # After turn 0, still wrong but less egregious
                    penalty += -self.equip_penalty_scale * utility * 3.0 * hand_mult

        # Wasted activation penalty: activating equipment/weapons when
        # hand is empty means no follow-up is possible (e.g. Tearing Shuko
        # buffs next Crouching Tiger, but with no cards there's nothing to
        # buff). Penalize proportional to the action's futility.
        if is_equip_activate:
            hand_size = len(state.get("myState", {}).get("hand", []))
            if hand_size == 0:
                penalty += -self.equip_penalty_scale * 3.0

        return penalty

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
        if int(opp_hp) <= 0 and int(my_hp) > 0:
            return "win"
        if int(my_hp) <= 0 and int(opp_hp) > 0:
            return "loss"
        # Fall back to turnPlayer heuristic if health deltas are ambiguous
        return "draw"

    @staticmethod
    def _is_terminal(state: dict) -> bool:
        phase = (state.get("phase") or {}).get("turnPhase", "")
        my_hp  = state.get("myState",    {}).get("health", 1)
        opp_hp = state.get("theirState", {}).get("health", 1)
        return phase == "OVER" or int(my_hp) <= 0 or int(opp_hp) <= 0

    @staticmethod
    def _extract_equip_ids(state: dict, player_key: str) -> list[str]:
        """Extract list of card IDs from a player's equipment zone."""
        return [
            c.get("cardID", "") for c in
            state.get(player_key, {}).get("equipment", [])
        ]

    def _equip_utility(self, card_id: str) -> float:
        """Return equipment utility weight incorporating hero synergy.

        Base utility comes from card_metadata (LLM-scored 0-10).  If the
        current hero has a synergy score for this equipment, we take the
        max of base utility and synergy — equipment that's core to the
        hero's strategy should never have a trivial penalty.

        Scale: 5→1.0 (average), 10→2.0, 0→0.0.
        """
        meta = self._card_metadata.get(card_id, {})
        util = int(meta.get("equipment_utility", _DEFAULT_EQUIP_UTILITY))
        if self._hero_id:
            hero_syn = meta.get("hero_scores", {}).get(self._hero_id, {})
            syn = int(hero_syn.get("hero_synergy", 0))
            util = max(util, syn)
        return util / 5.0

    def _make_info(
        self, state: dict, result: str | None, truncated: bool = False
    ) -> dict:
        base_mask = self._encoder.action_mask(state)
        mask = self._strategy_mask(state, base_mask) if self.use_strategy_mask else base_mask
        info: dict = {
            "legal_mask":  mask,
            "legal_moves": state.get("legalMoves", []),
            "raw_state":   state,
            "result":      result,
        }
        if result is not None:
            info["game_stats"] = self._stats.finalize(result, truncated)
        return info