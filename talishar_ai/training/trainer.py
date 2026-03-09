"""
trainer.py — Main PPO training loop.

High-level flow
---------------
1. reset() the environment → get initial obs + info
2. For each step:
     a. model.act(obs, mask)   → action, log_prob, value
     b. env.step(action)       → next_obs, reward, done, truncated, info
     c. buffer.add(...)
3. When buffer is full:
     a. Bootstrap last value with model.critic
     b. buffer.compute_returns_and_advantages()
     c. ppo.update(buffer)
     d. buffer.reset()
4. Log metrics every update; save checkpoint every checkpoint_freq steps.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..env import TalisharEnv
from ..features import OBS_DIM, MAX_ACTIONS
from ..models.network import ActorCritic
from .rollout import RolloutBuffer
from .ppo import PPOTrainer


class Trainer:
    def __init__(
        self,
        env:              TalisharEnv,
        model:            ActorCritic,
        ppo:              PPOTrainer,
        rollout_steps:    int  = 512,
        checkpoint_dir:   str  = "checkpoints",
        checkpoint_freq:  int  = 50_000,
        log_freq:         int  = 10,
        device:           torch.device | None = None,
    ) -> None:
        self.env             = env
        self.model           = model
        self.ppo             = ppo
        self.rollout_steps   = rollout_steps
        self.checkpoint_dir  = Path(checkpoint_dir)
        self.checkpoint_freq = checkpoint_freq
        self.log_freq        = log_freq
        self.device          = device or torch.device("cpu")

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._buffer = RolloutBuffer(capacity=rollout_steps, device=self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, total_steps: int) -> None:
        """Run the training loop for *total_steps* environment steps."""
        model = self.model.to(self.device)

        obs_np, info = self.env.reset()
        obs  = torch.from_numpy(obs_np).to(self.device)
        mask = torch.from_numpy(info["legal_mask"]).to(self.device)

        global_step     = 0
        episode_rewards: list[float] = []
        episode_results: list[str]   = []
        ep_reward       = 0.0
        update_count    = 0
        last_checkpoint = 0
        t0              = time.time()

        print(f"[Trainer] Starting training for {total_steps:,} steps")
        print(f"[Trainer] Device: {self.device} | Rollout: {self.rollout_steps}")

        while global_step < total_steps:
            # ---- Collect rollout -------------------------------------------
            with torch.no_grad():
                action, log_prob, value = model.act(obs, mask)

            action_np = int(action.item())
            next_obs_np, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated

            self._buffer.add(
                obs=obs_np,
                action=action_np,
                log_prob=float(log_prob.item()),
                value=float(value.item()),
                reward=float(reward),
                done=done,
                action_mask=info["legal_mask"],
            )

            ep_reward  += reward
            global_step += 1

            if done:
                episode_rewards.append(ep_reward)
                if info.get("result"):
                    episode_results.append(info["result"])
                ep_reward = 0.0
                obs_np, info = self.env.reset()
                obs  = torch.from_numpy(obs_np).to(self.device)
                mask = torch.from_numpy(info["legal_mask"]).to(self.device)
            else:
                obs_np = next_obs_np
                obs    = torch.from_numpy(obs_np).to(self.device)
                mask   = torch.from_numpy(info["legal_mask"]).to(self.device)

            # ---- Update when buffer full -----------------------------------
            if self._buffer.is_full():
                with torch.no_grad():
                    _, last_value = model(obs.unsqueeze(0), mask.unsqueeze(0))
                    last_value = float(last_value.item())

                self._buffer.compute_returns_and_advantages(last_value)
                stats = self.ppo.update(self._buffer)
                self._buffer.reset()
                update_count += 1

                if update_count % self.log_freq == 0:
                    self._log(global_step, total_steps, stats, episode_rewards,
                              episode_results, t0)
                    episode_rewards.clear()
                    episode_results.clear()

            # ---- Checkpoint ------------------------------------------------
            if global_step - last_checkpoint >= self.checkpoint_freq:
                self._save(global_step)
                last_checkpoint = global_step

        self._save(global_step, final=True)
        print(f"[Trainer] Training complete ({global_step:,} steps).")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(
        self,
        step: int,
        total: int,
        stats: dict[str, float],
        ep_rewards: list[float],
        ep_results: list[str],
        t0: float,
    ) -> None:
        elapsed   = time.time() - t0
        sps       = step / elapsed
        mean_rew  = np.mean(ep_rewards)  if ep_rewards  else float("nan")
        wins      = ep_results.count("win")
        n_ep      = len(ep_results)
        win_rate  = wins / n_ep if n_ep else float("nan")

        print(
            f"[{step:>8,}/{total:,}] "
            f"sps={sps:,.0f} | "
            f"ep_rew={mean_rew:+.3f} | "
            f"win%={win_rate:.1%} ({n_ep} ep) | "
            f"π={stats['policy_loss']:+.4f} "
            f"V={stats['value_loss']:.4f} "
            f"H={stats['entropy']:.4f}"
        )

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
