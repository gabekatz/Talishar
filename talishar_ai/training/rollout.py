"""
rollout.py — Fixed-capacity rollout buffer for on-policy PPO.

The buffer stores one contiguous trajectory segment of *capacity* steps
collected from a single environment.  After collection, we call
``compute_returns_and_advantages()`` once, then iterate over mini-batches
with ``get_batches()``.

Generalized Advantage Estimation (GAE)
---------------------------------------
GAE(γ, λ) balances bias and variance in the advantage estimate:

    δₜ   = rₜ + γ · V(sₜ₊₁) · (1 - doneₜ) - V(sₜ)   (TD error)
    Aₜ   = δₜ + (γλ) · Aₜ₊₁ · (1 - doneₜ)            (recursive)
    Rₜ   = Aₜ + V(sₜ)                                  (return)

γ = 0.99 discounts future rewards; λ = 0.95 trades bias for variance.
"""

from __future__ import annotations

from typing import Generator, Optional

import numpy as np
import torch

from ..features import OBS_DIM, MAX_ACTIONS, N_CARD_SLOTS, ACTION_DIM


class RolloutBuffer:
    def __init__(self, capacity: int, device: torch.device) -> None:
        self.capacity = capacity
        self.device   = device
        self.reset()

    def reset(self) -> None:
        C = self.capacity
        self.obs            = np.zeros((C, OBS_DIM),       dtype=np.float32)
        self.actions        = np.zeros(C,                   dtype=np.int64)
        self.log_probs      = np.zeros(C,                   dtype=np.float32)
        self.values         = np.zeros(C,                   dtype=np.float32)
        self.rewards        = np.zeros(C,                   dtype=np.float32)
        self.dones          = np.zeros(C,                   dtype=np.float32)
        self.action_masks   = np.zeros((C, MAX_ACTIONS),    dtype=bool)
        self.card_ids       = np.zeros((C, N_CARD_SLOTS),   dtype=np.int64)
        self.action_feats   = np.zeros((C, MAX_ACTIONS, ACTION_DIM), dtype=np.float32)
        self.episode_starts = np.zeros(C,                   dtype=bool)
        # LSTM-only: set by Trainer before filling the buffer each rollout.
        # Shape (n_lstm_layers, lstm_hidden); None when not using LSTM.
        self.initial_hidden_h: Optional[np.ndarray] = None
        self.initial_hidden_c: Optional[np.ndarray] = None
        self._ptr           = 0

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def add(
        self,
        obs:            np.ndarray,
        action:         int,
        log_prob:       float,
        value:          float,
        reward:         float,
        done:           bool,
        action_mask:    np.ndarray,
        card_ids:       Optional[np.ndarray] = None,
        action_feats:   Optional[np.ndarray] = None,
        episode_start:  bool = False,
    ) -> None:
        i = self._ptr
        self.obs[i]            = obs
        self.actions[i]        = action
        self.log_probs[i]      = log_prob
        self.values[i]         = value
        self.rewards[i]        = reward
        self.dones[i]          = float(done)
        self.action_masks[i]   = action_mask
        self.episode_starts[i] = episode_start
        if card_ids is not None:
            self.card_ids[i] = card_ids
        if action_feats is not None:
            self.action_feats[i] = action_feats
        self._ptr             += 1

    def is_full(self) -> bool:
        return self._ptr >= self.capacity

    def pad_remaining(self) -> None:
        """Fill unfilled slots with zero-reward done transitions.

        Used when a worker hits an unrecoverable error mid-rollout so the
        buffer can still be presented to the barrier / PPO update without
        deadlocking.  The padded transitions are effectively no-ops (zero
        reward, done=True) and won't meaningfully affect the update.
        """
        while self._ptr < self.capacity:
            self.dones[self._ptr] = 1.0
            self.episode_starts[self._ptr] = True
            self._ptr += 1

    # ------------------------------------------------------------------
    # Post-collection
    # ------------------------------------------------------------------

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma:      float = 0.99,
        gae_lambda: float = 0.95,
        normalize:  bool  = True,
    ) -> None:
        """
        Compute GAE advantages and discounted returns in-place.
        Call this once after the rollout is full, before get_batches().

        *last_value* is the critic's estimate for the state AFTER the last
        step in the buffer (used to bootstrap the return).
        """
        advantages = np.zeros(self.capacity, dtype=np.float32)
        last_gae   = 0.0
        for t in reversed(range(self.capacity)):
            next_val = last_value if t == self.capacity - 1 else self.values[t + 1]
            delta      = self.rewards[t] + gamma * next_val * (1.0 - self.dones[t]) - self.values[t]
            last_gae   = delta + gamma * gae_lambda * (1.0 - self.dones[t]) * last_gae
            advantages[t] = last_gae

        self.advantages = advantages
        self.returns    = advantages + self.values[: self.capacity]

        # Normalise advantages for training stability.
        # Skip when using multiple envs — merge() re-normalises over all envs
        # together, which is more accurate than normalising per-env first.
        if normalize:
            self.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # ------------------------------------------------------------------
    # Mini-batch iteration
    # ------------------------------------------------------------------

    def get_batches(
        self, batch_size: int
    ) -> Generator[dict[str, torch.Tensor], None, None]:
        """Yield random mini-batches as dicts of tensors on self.device."""
        indices = np.random.permutation(self.capacity)
        for start in range(0, self.capacity, batch_size):
            idx = indices[start : start + batch_size]
            yield {
                "obs":          torch.from_numpy(self.obs[idx]).to(self.device),
                "actions":      torch.from_numpy(self.actions[idx]).to(self.device),
                "old_log_probs":torch.from_numpy(self.log_probs[idx]).to(self.device),
                "advantages":   torch.from_numpy(self.advantages[idx]).to(self.device),
                "returns":      torch.from_numpy(self.returns[idx]).to(self.device),
                "action_masks": torch.from_numpy(self.action_masks[idx]).to(self.device),
                "card_ids":     torch.from_numpy(self.card_ids[idx]).to(torch.int32).to(self.device),
                "action_feats": torch.from_numpy(self.action_feats[idx]).to(self.device),
            }

    # ------------------------------------------------------------------
    # Multi-env support
    # ------------------------------------------------------------------

    @classmethod
    def merge(
        cls, buffers: list["RolloutBuffer"], device: torch.device
    ) -> "RolloutBuffer":
        """
        Concatenate N full, already-computed buffers into one for a joint
        PPO update.

        Each buffer must have had ``compute_returns_and_advantages()`` called
        before merging.  The merged buffer's capacity equals the sum of all
        individual capacities.

        Parameters
        ----------
        buffers:
            List of full RolloutBuffers (one per parallel env).
        device:
            Device for the returned buffer's tensor operations.
        """
        total   = sum(b.capacity for b in buffers)
        merged  = cls(capacity=total, device=device)

        merged.obs          = np.concatenate([b.obs          for b in buffers], axis=0)
        merged.actions      = np.concatenate([b.actions      for b in buffers], axis=0)
        merged.log_probs    = np.concatenate([b.log_probs    for b in buffers], axis=0)
        merged.values       = np.concatenate([b.values       for b in buffers], axis=0)
        merged.rewards      = np.concatenate([b.rewards      for b in buffers], axis=0)
        merged.dones        = np.concatenate([b.dones        for b in buffers], axis=0)
        merged.action_masks = np.concatenate([b.action_masks for b in buffers], axis=0)
        merged.card_ids     = np.concatenate([b.card_ids     for b in buffers], axis=0)
        merged.action_feats = np.concatenate([b.action_feats for b in buffers], axis=0)
        merged.advantages   = np.concatenate([b.advantages   for b in buffers], axis=0)
        merged.returns      = np.concatenate([b.returns      for b in buffers], axis=0)
        merged._ptr         = total

        # Re-normalise advantages over the full merged set so the PPO update
        # sees a consistent scale regardless of how many envs were merged.
        merged.advantages = (
            (merged.advantages - merged.advantages.mean())
            / (merged.advantages.std() + 1e-8)
        )

        return merged
