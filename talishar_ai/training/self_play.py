"""
self_play.py — Self-play infrastructure for training against a frozen copy
of the active policy instead of the fixed EncounterAI.

Why self-play?
--------------
EncounterAI is a static, weak opponent.  Once the policy beats it reliably,
training signal degenerates.  Self-play creates an auto-curriculum: the
frozen opponent keeps pace with the active policy, so the game is always
competitive and the policy is forced to discover stronger strategies.

Components
----------
SelfPlayEnv
    Wraps a TalisharEnv (p2_is_ai=False) and drives P2 with a frozen
    ActorCritic after each P1 action.  Presents the same Gymnasium API as
    TalisharEnv, so it is a drop-in replacement in Trainer/ParallelEnvManager.

SelfPlayManager
    Holds N SelfPlayEnv instances and rotates the frozen opponent every
    *update_freq* PPO updates by copying the active model's state_dict into
    each env's frozen model.

Usage
-----
# In train.py:
sp_envs = [SelfPlayEnv(make_env(), active_model, device) for _ in range(n_envs)]
parallel = ParallelEnvManager([lambda e=e: e for e in sp_envs])
manager  = SelfPlayManager(sp_envs, update_freq=20)

trainer = Trainer(
    env=parallel, model=active_model, ppo=ppo, ...,
    post_update_fn=manager.maybe_rotate,
)
"""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import torch
import gymnasium as gym

from ..env import TalisharEnv
from ..features import StateEncoder
from ..models.network import ActorCritic


class SelfPlayEnv(gym.Env):
    """
    TalisharEnv wrapper that drives P2 with a frozen policy after every P1 action.

    The active (learning) policy always plays P1.  The frozen opponent plays P2.
    Call ``update_opponent(state_dict)`` to rotate the frozen opponent to a new
    checkpoint.

    Parameters
    ----------
    env:
        A TalisharEnv constructed with ``p2_is_ai=False``.
    active_model:
        The live ActorCritic being trained.  Used only to clone the initial
        frozen opponent — it is NOT called during rollout collection.
    device:
        Torch device for frozen-model inference.
    encoder:
        StateEncoder (optionally with CardVocab) for encoding P2's observations.
        Should match the active model's embedding configuration.
    p2_temperature:
        Sampling temperature for the frozen opponent.  1.0 = same distribution
        as the policy; <1 = sharper (more exploitative); >1 = more random.
        Using sampling instead of argmax prevents the frozen opponent from
        collapsing to a single degenerate action (e.g. always passing).
    poll_sleep:
        Seconds to sleep when neither player has priority (engine processing).
    max_drive_iters:
        Safety limit on the P2-driving loop per P1 step.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env:            TalisharEnv,
        active_model:   ActorCritic,
        device:         torch.device,
        encoder:        StateEncoder | None = None,
        p2_temperature: float = 1.0,
        poll_sleep:     float = 0.1,
        max_drive_iters: int  = 200,
    ) -> None:
        super().__init__()
        assert not env.p2_is_ai, (
            "SelfPlayEnv requires TalisharEnv(p2_is_ai=False) so P2 is not "
            "auto-driven server-side."
        )
        self._env            = env
        self._device         = device
        self._encoder        = encoder or StateEncoder()
        self._p2_temperature = p2_temperature
        self._poll_sleep     = poll_sleep
        self._max_iters      = max_drive_iters

        # Frozen opponent: deep copy of the active model at construction time.
        # Weights are updated via update_opponent(); gradients are never computed.
        # Always kept on CPU so that 32+ threads can run inference without
        # contending on MPS's single Metal command queue.
        self._frozen = copy.deepcopy(active_model).cpu()
        self._frozen.eval()
        for p in self._frozen.parameters():
            p.requires_grad_(False)
        self._frozen_device = torch.device("cpu")

        # LSTM-specific: maintain P2's hidden state across actions within a game.
        self._frozen_is_lstm = getattr(self._frozen, "use_lstm", False)
        if self._frozen_is_lstm:
            self._p2_hidden_h, self._p2_hidden_c = self._frozen.init_hidden(1, self._frozen_device)
        else:
            self._p2_hidden_h = self._p2_hidden_c = None

        # Mirror the inner env's Gymnasium spaces
        self.observation_space = env.observation_space
        self.action_space      = env.action_space

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        # New game → wipe P2's LSTM memory so each episode starts clean.
        if self._frozen_is_lstm:
            self._p2_hidden_h, self._p2_hidden_c = self._frozen.init_hidden(1, self._frozen_device)
        return self._env.reset(seed=seed, options=options)

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        1. Submit P1's chosen action.
        2. Drive P2's decisions until P1 regains priority (or game ends).
        3. Delegate reward / obs / info computation to the inner env.
        """
        state = self._env._last_state
        legal = state.get("legalMoves", [])

        if action >= len(legal):
            raise ValueError(
                f"Action {action} out of range — only {len(legal)} legal moves."
            )

        params = legal[action]["params"]

        # Both the P1 submission and the P2 driving loop can fail with a
        # RuntimeError when the PHP engine returns an empty/invalid response
        # (happens when the game file is gone or the server is overloaded).
        # Treat any such failure as a truncated episode so training continues.
        try:
            next_state = self._env.gm.submit_action(
                self._env._game_name, self._env.player_id, self._env._auth_key, params
            )
            resolved = self._drive_p2(next_state)
        except RuntimeError as exc:
            print(
                f"[SelfPlay] WARNING: game {self._env._game_name} failed "
                f"(step {self._env._step_count}): {exc}"
            )
            obs  = self._env._encoder.encode(state)
            info = {
                "legal_mask":  self._env._encoder.action_mask(state),
                "legal_moves": state.get("legalMoves", []),
                "raw_state":   state,
                "result":      "truncated",
            }
            # Neutral reward: P2 failures (pass-loop, undo-loop, server error)
            # are not P1's fault — penalizing P1 poisons the learning signal.
            return obs, 0.0, False, True, info

        # Delegate all reward / obs / info logic to the inner env
        return self._env._finalize_step(resolved)

    def action_masks(self) -> np.ndarray:
        return self._env.action_masks()

    def render(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Opponent rotation
    # ------------------------------------------------------------------

    def update_opponent(self, state_dict: dict) -> None:
        """
        Replace the frozen opponent's weights with *state_dict*.

        Uses ``copy.deepcopy`` on each tensor so the frozen model has no
        shared storage with the active model.
        """
        self._frozen.load_state_dict(
            {k: v.clone().cpu() for k, v in state_dict.items()}
        )
        # Stale hidden state from the old policy is meaningless for the new
        # one — reset so the next game starts clean.
        if self._frozen_is_lstm:
            self._p2_hidden_h, self._p2_hidden_c = self._frozen.init_hidden(1, self._frozen_device)

    # ------------------------------------------------------------------
    # P2 driving loop
    # ------------------------------------------------------------------

    def _drive_p2(self, state_after_p1: dict[str, Any]) -> dict[str, Any]:
        """
        Poll P2's priority and sample frozen-policy actions until P1
        regains priority or the game terminates.

        Returns the game state from P1's perspective, ready for _finalize_step.
        """
        gm         = self._env.gm
        game_name  = self._env._game_name
        p1_id      = self._env.player_id
        p1_key     = self._env._auth_key
        p2_key     = self._env._p2_key
        p2_id      = 2 if p1_id == 1 else 1

        _UNDO_MODES = {10000, 10001, 10003}
        _PASS_MODE  = 99
        # After this many consecutive *chosen* passes (when other actions
        # existed), filter pass from the move list so P2 takes a real action.
        _PASS_PATIENCE    = 3
        _MAX_IDLE_POLLS   = 60   # max polls where neither player has priority
        _MAX_TOTAL_ITERS  = 500  # safety cap on ALL iterations (incl. auto-pass)
        _LOG_INTERVAL     = 50   # log diagnostics every N iterations

        total_undo     = 0   # total forced-cancel submissions this drive call
        consec_pass    = 0   # consecutive policy-chosen pass actions
        idle_polls     = 0   # polls where neither player had priority
        total_iters    = 0   # all iterations (for safety cap + logging)
        auto_passes    = 0   # auto-pass count (for diagnostics)

        while idle_polls < _MAX_IDLE_POLLS and total_iters < _MAX_TOTAL_ITERS:
            total_iters += 1

            # Periodic diagnostics — helps identify what's stalling.
            if total_iters % _LOG_INTERVAL == 0:
                print(
                    f"[_drive_p2] game {game_name} iter={total_iters} "
                    f"auto_pass={auto_passes} idle={idle_polls} "
                    f"undo={total_undo}"
                )

            # Fast path: if terminal already (e.g. P1 died from an on-hit)
            if TalisharEnv._is_terminal(state_after_p1):
                return gm.get_state_blocking(game_name, p1_id, p1_key)

            # Check P2's state
            p2_state = gm.get_state(game_name, p2_id, p2_key)

            if TalisharEnv._is_terminal(p2_state):
                return gm.get_state_blocking(game_name, p1_id, p1_key)

            if p2_state.get("havePriority") and p2_state.get("legalMoves"):
                idle_polls = 0  # game is progressing — reset idle counter

                # Filter undo/cancel loop modes.
                all_moves   = p2_state["legalMoves"]
                non_undo    = [m for m in all_moves if m.get("mode") not in _UNDO_MODES]
                valid_moves = non_undo if non_undo else all_moves

                # Count total forced-cancel submissions this drive call.
                if not non_undo:
                    total_undo += 1
                    if total_undo >= 5:
                        raise RuntimeError(
                            f"P2 stuck in undo/cancel loop ({total_undo} cancels) "
                            f"in game {game_name}"
                        )

                # Separate pass moves from real (non-pass) moves.
                non_pass = [m for m in valid_moves
                            if m.get("mode") != _PASS_MODE
                            and m["params"].get("mode") != _PASS_MODE]

                # Fast path: pass is the only legal action — auto-submit it
                # without model inference.  This is normal gameplay (e.g. P2
                # has no blocks/reactions during combat) and not a stuck state.
                if not non_pass:
                    auto_passes += 1
                    state_after_p1 = gm.submit_action(
                        game_name, p2_id, p2_key, valid_moves[0]["params"]
                    )
                    continue

                # After several consecutive policy-chosen passes, remove pass
                # from the options so P2 is forced to take a real action.
                if consec_pass >= _PASS_PATIENCE:
                    valid_moves = non_pass
                    consec_pass = 0

                # P2 has a decision — sample from frozen policy.
                p2_state_filtered = {**p2_state, "legalMoves": valid_moves}

                obs2  = self._encoder.encode(p2_state_filtered)
                mask2 = self._encoder.action_mask(p2_state_filtered)
                ids2 = self._encoder.card_ids(p2_state_filtered)

                obs_t  = torch.from_numpy(obs2).unsqueeze(0).to(self._frozen_device)
                mask_t = torch.from_numpy(mask2).unsqueeze(0).to(self._frozen_device)
                ids_t  = torch.from_numpy(ids2).unsqueeze(0).to(torch.int32).to(self._frozen_device)

                with torch.no_grad():
                    use_emb = self._frozen.use_embeddings
                    if self._frozen_is_lstm:
                        logits, _, self._p2_hidden_h, self._p2_hidden_c = self._frozen(
                            obs_t, mask_t,
                            self._p2_hidden_h, self._p2_hidden_c,
                            ids_t if use_emb else None,
                        )
                    else:
                        logits, _ = self._frozen(
                            obs_t, mask_t,
                            ids_t if use_emb else None,
                        )
                    dist = torch.distributions.Categorical(
                        logits=logits / self._p2_temperature
                    )
                    p2_action = int(dist.sample().item())

                move2 = valid_moves[p2_action]

                # Track consecutive passes for the filter above.
                if move2["params"].get("mode") == _PASS_MODE:
                    consec_pass += 1
                else:
                    consec_pass = 0

                state_after_p1 = gm.submit_action(
                    game_name, p2_id, p2_key, move2["params"]
                )
                continue  # re-check after P2 acts

            # P2 doesn't have priority — check if P1 has legal moves ready.
            p1_state = gm.get_state(game_name, p1_id, p1_key)
            if TalisharEnv._is_terminal(p1_state):
                return p1_state
            if p1_state.get("havePriority") and p1_state.get("legalMoves"):
                return p1_state

            # Neither player has priority: engine is processing.
            idle_polls += 1
            time.sleep(self._poll_sleep)

        # ---- Diagnostics on stall ----
        # Fetch both players' states for the error message.
        def _state_summary(state: dict) -> str:
            phase = (state.get("phase") or {}).get("turnPhase", "?")
            has_pri = state.get("havePriority", False)
            moves = state.get("legalMoves", [])
            move_types = [
                f"{m.get('type','?')}(m{m.get('mode','?')})"
                for m in moves[:8]
            ]
            pending = state.get("pendingDecision")
            pending_str = ""
            if pending:
                pending_str = f" pending={pending.get('type','?')}"
            return (
                f"phase={phase} pri={has_pri} "
                f"moves=[{', '.join(move_types)}]{pending_str}"
            )

        try:
            p2_final = gm.get_state(game_name, p2_id, p2_key)
            p1_final = gm.get_state(game_name, p1_id, p1_key)
            p2_summary = _state_summary(p2_final)
            p1_summary = _state_summary(p1_final)
        except Exception:
            p2_summary = p1_summary = "<fetch failed>"

        reason = "idle timeout" if idle_polls >= _MAX_IDLE_POLLS else "iteration cap"
        raise RuntimeError(
            f"_drive_p2 stalled in game {game_name} ({reason}): "
            f"iters={total_iters} auto_pass={auto_passes} idle={idle_polls} | "
            f"P1=[{p1_summary}] P2=[{p2_summary}]"
        )


class SelfPlayManager:
    """
    Coordinates opponent rotation across N SelfPlayEnv instances.

    Call ``maybe_rotate(update_count, state_dict)`` after every PPO update.
    When ``update_count`` is a multiple of ``update_freq``, the frozen
    opponent in every env is updated to the latest active model weights.

    Parameters
    ----------
    envs:
        All SelfPlayEnv instances (one per parallel worker).
    update_freq:
        Number of PPO updates between opponent rotations.  Lower values
        mean a harder, faster-moving target; higher values give more
        stable training signal.  20–50 is a reasonable starting range.
    """

    def __init__(
        self,
        envs:        list[SelfPlayEnv],
        update_freq: int = 20,
    ) -> None:
        self._envs        = envs
        self.update_freq  = update_freq
        self._rotations   = 0

    def maybe_rotate(self, update_count: int, state_dict: dict) -> bool:
        """
        Rotate the frozen opponent if *update_count* is a multiple of
        *update_freq*.

        Parameters
        ----------
        update_count:
            Cumulative PPO update count (from the Trainer).
        state_dict:
            ``model.state_dict()`` of the active (learning) policy.

        Returns
        -------
        True if the opponent was rotated this call, False otherwise.
        """
        if update_count > 0 and update_count % self.update_freq == 0:
            for env in self._envs:
                env.update_opponent(state_dict)
            self._rotations += 1
            print(
                f"[SelfPlay] Opponent rotated (rotation #{self._rotations}, "
                f"update {update_count})"
            )
            return True
        return False

    @property
    def rotations(self) -> int:
        """Total number of opponent rotations performed so far."""
        return self._rotations
