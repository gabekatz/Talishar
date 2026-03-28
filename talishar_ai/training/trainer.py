"""
trainer.py — PPO training loop supporting N parallel environments.

With n_envs=1 this is equivalent to the original single-env trainer.
With n_envs>1, all envs are stepped concurrently via ParallelEnvManager
(a thread pool), and each env's rollout data is kept in its own
RolloutBuffer so GAE is computed correctly per trajectory.  Before each
PPO update the N buffers are merged into one via RolloutBuffer.merge().

Sample throughput scales approximately linearly with n_envs since the
bottleneck is HTTP I/O to the PHP engine, not CPU computation.

High-level flow
---------------
1. reset() all envs in parallel → obs[N], mask[N]
2. For each step in rollout_steps:
     a. Batch model forward: (N, OBS_DIM) → actions[N], log_probs[N], values[N]
     b. ParallelEnvManager.step(actions) → (next_obs, reward, done, …)[N]  (parallel)
     c. Store each transition in its env's RolloutBuffer
     d. Auto-reset any envs that finished (mid-rollout)
3. When all N buffers are full:
     a. Batch bootstrap: (N, OBS_DIM) → last_values[N]
     b. compute_returns_and_advantages() per buffer
     c. RolloutBuffer.merge(buffers) → one big buffer
     d. ppo.update(merged buffer)
     e. Reset all buffers
4. Log metrics; checkpoint periodically
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Union

import numpy as np
import torch
from torch.distributions import Categorical

from ..env import TalisharEnv
from ..parallel_env import ParallelEnvManager
from ..features import StateEncoder, N_CARD_SLOTS, MAX_ACTIONS, ACTION_DIM
from ..models.network import ActorCritic
from .rollout import RolloutBuffer
from .ppo import PPOTrainer
from .tb_logger import TBLogger


class Trainer:
    def __init__(
        self,
        env:             Union[TalisharEnv, ParallelEnvManager],
        model:           ActorCritic,
        ppo:             PPOTrainer,
        rollout_steps:    int                          = 512,
        checkpoint_dir:   str                          = "checkpoints",
        checkpoint_freq:  int                          = 50_000,
        log_freq:         int                          = 10,
        device:           torch.device | None          = None,
        encoder:          StateEncoder | None          = None,
        post_update_fn:   Callable[[int, dict], None] | None = None,
        tb_logger:        TBLogger | None              = None,
    ) -> None:
        self.model           = model
        self.ppo             = ppo
        self.rollout_steps   = rollout_steps
        self.checkpoint_dir  = Path(checkpoint_dir)
        self.checkpoint_freq = checkpoint_freq
        self.log_freq        = log_freq
        self.device          = device or torch.device("cpu")
        # Encoder is used to extract card IDs when embeddings are enabled.
        # Falls back to a plain StateEncoder (returns all-zero card_ids).
        self._encoder        = encoder or StateEncoder()
        self.use_embeddings  = model.use_embeddings
        self.use_action_embed = getattr(model, "use_action_embed", False)
        self._post_update_fn = post_update_fn
        self._tb             = tb_logger

        # LSTM support: detect recurrent model and pre-allocate hidden state.
        self.use_lstm = getattr(model, "use_lstm", False)
        if self.use_lstm:
            # Shape: (n_lstm_layers, N_envs, lstm_hidden)
            # Will be resized in train() once n_envs is known.
            self._hidden_h: torch.Tensor | None = None
            self._hidden_c: torch.Tensor | None = None

        # Wrap a single TalisharEnv in a 1-env ParallelEnvManager so the
        # rest of the code is always vectorised.
        if isinstance(env, TalisharEnv):
            self.envs = ParallelEnvManager([lambda e=env: e])
        else:
            self.envs = env

        self.n_envs = self.envs.n_envs

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # One buffer per env — keeps trajectories separate so GAE is correct.
        self._buffers = [
            RolloutBuffer(capacity=rollout_steps, device=self.device)
            for _ in range(self.n_envs)
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, total_steps: int, start_step: int = 0) -> None:
        """Run the training loop for *total_steps* environment steps.

        Parameters
        ----------
        total_steps : int
            Total number of environment steps to run.
        start_step : int
            Global step offset (from a resumed checkpoint).  Checkpoints
            and log lines will show ``start_step + local_step``.
        """
        model = self.model.to(self.device)
        # MPS has bugs in nn.LSTM — keep it on CPU (I/O-bound, no perf impact).
        if self.use_lstm and self.device.type == "mps":
            model.lstm = model.lstm.cpu()

        # Reset all envs in parallel
        obs_list, info_list = self.envs.reset()
        obs_arr      = np.stack(obs_list)                                    # (N, OBS_DIM)
        mask_arr     = np.stack([i["legal_mask"] for i in info_list])        # (N, MAX_ACTIONS)
        card_ids_arr = self._extract_card_ids(info_list)                     # (N, N_CARD_SLOTS)
        afeats_arr   = self._extract_action_feats(info_list)                 # (N, MAX_ACTIONS, ACTION_DIM)

        ep_rewards: list[float] = [0.0] * self.n_envs
        all_ep_rewards: list[float] = []
        all_ep_results: list[str]   = []
        global_step     = start_step
        update_count    = 0
        last_checkpoint = global_step
        t0              = time.time()

        # LSTM state — initialise to zeros; each column i corresponds to env i.
        # episode_starts[i] tracks whether env i's NEXT step begins a new episode.
        if self.use_lstm:
            self._hidden_h, self._hidden_c = model.init_hidden(self.n_envs, self.device)
            # All envs just reset → first step of each is an episode start
            episode_starts_np = np.ones(self.n_envs, dtype=bool)
            # Seed initial hidden states into the (empty) buffers so the very
            # first rollout's update_lstm() call has h0/c0 available.
            for i, buf in enumerate(self._buffers):
                buf.initial_hidden_h = self._hidden_h[:, i, :].detach().cpu().numpy()
                buf.initial_hidden_c = self._hidden_c[:, i, :].detach().cpu().numpy()
        else:
            episode_starts_np = np.zeros(self.n_envs, dtype=bool)

        transitions_per_update = self.rollout_steps * self.n_envs
        target_step = start_step + total_steps

        # Periodic heartbeat (once per minute)
        _hb_time       = time.time()
        _hb_steps      = 0
        _hb_games      = 0
        _HB_INTERVAL   = 60.0
        print(
            f"[Trainer] Starting training for {total_steps:,} steps "
            f"(start={start_step:,}, target={target_step:,}, n_envs={self.n_envs})"
        )
        print(
            f"[Trainer] Device: {self.device} | "
            f"Rollout: {self.rollout_steps} steps × {self.n_envs} envs "
            f"= {transitions_per_update:,} transitions/update"
        )

        while global_step < target_step:
            # ---- Batch model inference (one forward pass for all N envs) ----
            obs_t      = torch.from_numpy(obs_arr).to(self.device)       # (N, OBS_DIM)
            mask_t     = torch.from_numpy(mask_arr).to(self.device)      # (N, MAX_ACTIONS)
            # Use int32 to avoid MPS int64 corruption issues
            card_ids_t = torch.from_numpy(card_ids_arr).to(torch.int32).to(self.device)
            afeats_t   = torch.from_numpy(afeats_arr).to(self.device)    # (N, MAX_ACTIONS, ACTION_DIM)

            with torch.no_grad():
                if self.use_action_embed and self.use_lstm:
                    logits, values, new_h, new_c = model(
                        obs_t, mask_t,
                        self._hidden_h, self._hidden_c,
                        afeats_t,
                        card_ids_t if self.use_embeddings else None,
                    )
                    self._hidden_h, self._hidden_c = new_h, new_c
                elif self.use_action_embed:
                    logits, values = model(obs_t, mask_t, afeats_t, card_ids_t if self.use_embeddings else None)
                elif self.use_lstm:
                    logits, values, new_h, new_c = model(
                        obs_t, mask_t,
                        self._hidden_h, self._hidden_c,
                        card_ids_t if self.use_embeddings else None,
                    )
                    self._hidden_h, self._hidden_c = new_h, new_c
                else:
                    logits, values = model(obs_t, mask_t, card_ids_t if self.use_embeddings else None)
                dist        = Categorical(logits=logits)
                actions_t   = dist.sample()             # (N,)
                log_probs_t = dist.log_prob(actions_t)  # (N,)

            actions_np   = actions_t.cpu().numpy()    # (N,) int64
            log_probs_np = log_probs_t.cpu().numpy()  # (N,) float32
            values_np    = values.cpu().numpy()       # (N,) float32

            # ---- Step all envs in parallel ---------------------------------
            next_obs_list, rew_list, term_list, trunc_list, next_info_list = (
                self.envs.step(actions_np.tolist())
            )

            # ---- Store transitions and handle episode boundaries -----------
            next_episode_starts = np.zeros(self.n_envs, dtype=bool)
            for i in range(self.n_envs):
                done = term_list[i] or trunc_list[i]

                self._buffers[i].add(
                    obs           = obs_arr[i],
                    action        = int(actions_np[i]),
                    log_prob      = float(log_probs_np[i]),
                    value         = float(values_np[i]),
                    reward        = float(rew_list[i]),
                    done          = done,
                    action_mask   = mask_arr[i],
                    card_ids      = card_ids_arr[i],
                    action_feats  = afeats_arr[i],
                    episode_start = bool(episode_starts_np[i]),
                )

                ep_rewards[i] += float(rew_list[i])
                global_step   += 1
                _hb_steps     += 1

                if done:
                    all_ep_rewards.append(ep_rewards[i])
                    _hb_games += 1
                    if next_info_list[i].get("result"):
                        all_ep_results.append(next_info_list[i]["result"])
                    if self._tb and next_info_list[i].get("game_stats"):
                        gs = next_info_list[i]["game_stats"]
                        gs_dict = gs.to_dict() if hasattr(gs, "to_dict") else gs
                        self._tb.log_game(global_step, gs_dict)
                    ep_rewards[i] = 0.0

                    # Auto-reset this env without blocking the others —
                    # the new obs overwrites this slot for the next step.
                    new_obs, new_info      = self.envs.reset_single(i)
                    next_obs_list[i]       = new_obs
                    next_info_list[i]      = new_info

                    # LSTM: wipe this env's hidden state so the new episode
                    # starts with a blank memory.
                    if self.use_lstm:
                        self._hidden_h[:, i, :] = 0.0
                        self._hidden_c[:, i, :] = 0.0

                    next_episode_starts[i] = True

            episode_starts_np = next_episode_starts

            # Advance obs/mask/card_ids/action_feats for next step
            obs_arr      = np.stack(next_obs_list)
            mask_arr     = np.stack([info["legal_mask"] for info in next_info_list])
            card_ids_arr = self._extract_card_ids(next_info_list)
            afeats_arr   = self._extract_action_feats(next_info_list)

            # ---- Periodic heartbeat (once per minute) --------------------
            now = time.time()
            if now - _hb_time >= _HB_INTERVAL:
                elapsed_min = (now - _hb_time) / 60.0
                ts = datetime.now().strftime("%H:%M:%S")
                print(
                    f"[{ts}] [heartbeat] "
                    f"{_hb_steps:,} steps | "
                    f"{_hb_games} games | "
                    f"{_hb_steps / elapsed_min:,.0f} steps/min | "
                    f"global={global_step:,}/{target_step:,}"
                )
                _hb_time  = now
                _hb_steps = 0
                _hb_games = 0

            # ---- Update when all N buffers are full ------------------------
            if all(b.is_full() for b in self._buffers):
                # Bootstrap last values with a single batched forward pass
                obs_t      = torch.from_numpy(obs_arr).to(self.device)
                mask_t     = torch.from_numpy(mask_arr).to(self.device)
                card_ids_t = torch.from_numpy(card_ids_arr).to(torch.int32).to(self.device)
                afeats_t   = torch.from_numpy(afeats_arr).to(self.device)
                with torch.no_grad():
                    if self.use_action_embed and self.use_lstm:
                        _, last_values, _, _ = model(
                            obs_t, mask_t,
                            self._hidden_h, self._hidden_c,
                            afeats_t,
                            card_ids_t if self.use_embeddings else None,
                        )
                    elif self.use_action_embed:
                        _, last_values = model(
                            obs_t, mask_t, afeats_t,
                            card_ids_t if self.use_embeddings else None,
                        )
                    elif self.use_lstm:
                        _, last_values, _, _ = model(
                            obs_t, mask_t,
                            self._hidden_h, self._hidden_c,
                            card_ids_t if self.use_embeddings else None,
                        )
                    else:
                        _, last_values = model(
                            obs_t, mask_t,
                            card_ids_t if self.use_embeddings else None,
                        )
                last_values_np = last_values.cpu().numpy()  # (N,)

                # For LSTM: never normalise per-env — update_lstm normalises
                # globally.  For MLP: normalise per-env only when n_envs == 1;
                # merge() handles multi-env normalisation otherwise.
                normalize_per_env = (not self.use_lstm) and (self.n_envs == 1)
                for i, buf in enumerate(self._buffers):
                    buf.compute_returns_and_advantages(
                        float(last_values_np[i]), normalize=normalize_per_env
                    )

                if self.use_lstm:
                    # Capture initial hidden states for the NEXT rollout
                    # before buffers are reset (they'll be overwritten at
                    # the top of the loop).
                    stats = self.ppo.update_lstm(self._buffers)
                else:
                    # Merge all per-env buffers into one for the PPO update
                    merged = RolloutBuffer.merge(self._buffers, self.device)
                    stats  = self.ppo.update(merged)

                for i, buf in enumerate(self._buffers):
                    buf.reset()
                    # Capture current hidden state as the starting point for
                    # the next rollout (done here so it's always up to date).
                    if self.use_lstm:
                        buf.initial_hidden_h = self._hidden_h[:, i, :].detach().cpu().numpy()
                        buf.initial_hidden_c = self._hidden_c[:, i, :].detach().cpu().numpy()

                update_count += 1

                if self._post_update_fn is not None:
                    self._post_update_fn(update_count, self.model.state_dict())

                if update_count % self.log_freq == 0:
                    self._log(
                        global_step, target_step, stats,
                        all_ep_rewards, all_ep_results, t0,
                    )
                    all_ep_rewards.clear()
                    all_ep_results.clear()

            # ---- Checkpoint ------------------------------------------------
            if global_step - last_checkpoint >= self.checkpoint_freq:
                self._save(global_step)
                last_checkpoint = global_step

        self._save(global_step, final=True)
        if self._tb:
            self._tb.close()
        print(f"[Trainer] Training complete ({global_step:,} steps).")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(
        self,
        step:       int,
        total:      int,
        stats:      dict[str, float],
        ep_rewards: list[float],
        ep_results: list[str],
        t0:         float,
    ) -> None:
        elapsed  = time.time() - t0
        sps      = step / elapsed
        mean_rew = np.mean(ep_rewards) if ep_rewards else float("nan")
        n_ep     = len(ep_results)
        win_rate = ep_results.count("win") / n_ep if n_ep else float("nan")

        now = datetime.now().strftime("%H:%M:%S")
        print(
            f"[{now}] [{step:>8,}/{total:,}] "
            f"sps={sps:,.0f} | "
            f"ep_rew={mean_rew:+.3f} | "
            f"win%={win_rate:.1%} ({n_ep} ep) | "
            f"π={stats['policy_loss']:+.4f} "
            f"V={stats['value_loss']:.4f} "
            f"H={stats['entropy']:.4f}"
        )

        if self._tb:
            self._tb.log_training(step, stats, ep_rewards, ep_results, sps)

    def _extract_action_feats(self, info_list: list[dict]) -> np.ndarray:
        """Extract action feature arrays from a list of env info dicts."""
        if self.use_action_embed:
            return np.stack([
                self._encoder.encode_actions(info["raw_state"]) for info in info_list
            ])  # (N, MAX_ACTIONS, ACTION_DIM)
        # When action embeddings are disabled, return zeros — unused by the model
        # but keeps the buffer shapes consistent.
        return np.zeros(
            (len(info_list), MAX_ACTIONS, ACTION_DIM), dtype=np.float32
        )

    def _extract_card_ids(self, info_list: list[dict]) -> np.ndarray:
        """
        Extract card ID arrays from a list of env info dicts.

        Uses the encoder's card_ids() method on each info["raw_state"].
        When embeddings are disabled the encoder returns all-zeros, so the
        buffer always has a card_ids array of the right shape but it's
        simply unused by the model.
        """
        return np.stack([
            self._encoder.card_ids(info["raw_state"]) for info in info_list
        ])  # (N, N_CARD_SLOTS)

    def _save(self, step: int, final: bool = False) -> None:
        tag  = "final" if final else f"{step}"
        path = self.checkpoint_dir / f"model_{tag}.pt"
        torch.save(
            {
                "step":        step,
                "model_state": self.model.state_dict(),
                "optim_state": self.ppo.optimizer.state_dict(),
            },
            path,
        )
        print(f"[Trainer] Saved checkpoint → {path}")
