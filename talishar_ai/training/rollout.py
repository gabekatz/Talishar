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

from typing import Generator

import numpy as np
import torch

from ..features import OBS_DIM, MAX_ACTIONS


class RolloutBuffer:
    def __init__(self, capacity: int, device: torch.device) -> None:
        self.capacity = capacity
        self.device   = device
        self.reset()

    def reset(self) -> None:
        C = self.capacity
        self.obs          = np.zeros((C, OBS_DIM),      dtype=np.float32)
        self.actions      = np.zeros(C,                  dtype=np.int64)
        self.log_probs    = np.zeros(C,                  dtype=np.float32)
        self.values       = np.zeros(C,                  dtype=np.float32)
        self.rewards      = np.zeros(C,                  dtype=np.float32)
        self.dones        = np.zeros(C,                  dtype=np.float32)
        self.action_masks = np.zeros((C, MAX_ACTIONS),   dtype=bool)
        self._ptr         = 0

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def add(
        self,
        obs:          np.ndarray,
        action:       int,
        log_prob:     float,
        value:        float,
        reward:       float,
        done:         bool,
        action_mask:  np.ndarray,
    ) -> None:
        i = self._ptr
        self.obs[i]          = obs
        self.actions[i]      = action
        self.log_probs[i]    = log_prob
        self.values[i]       = value
        self.rewards[i]      = reward
        self.dones[i]        = float(done)
        self.action_masks[i] = action_mask
        self._ptr           += 1

    def is_full(self) -> bool:
        return self._ptr >= self.capacity

    # ------------------------------------------------------------------
    # Post-collection
    # ------------------------------------------------------------------

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma:      float = 0.99,
        gae_lambda: float = 0.95,
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

        # Normalise advantages for training stability
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
            }
