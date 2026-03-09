"""
ppo.py — Proximal Policy Optimisation (PPO-Clip).

PPO overview
------------
After collecting a fixed rollout of N steps we run K epochs of mini-batch
gradient descent on three loss terms:

  L_clip   = E[ min(r·A, clip(r, 1-ε, 1+ε)·A) ]   (policy loss)
  L_vf     = E[ (V_θ(s) - R)² ]                     (value loss)
  L_ent    = E[ H(π_θ(·|s)) ]                       (entropy bonus)

  L_total  = -L_clip + c_vf·L_vf - c_ent·L_ent

Where r = π_θ(a|s) / π_θ_old(a|s) is the probability ratio.

Clipping r to [1-ε, 1+ε] prevents catastrophically large policy updates.
The entropy bonus encourages exploration early in training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from ..models.network import ActorCritic
from .rollout import RolloutBuffer


class PPOTrainer:
    def __init__(
        self,
        model:          ActorCritic,
        lr:             float = 3e-4,
        clip_eps:       float = 0.2,
        ent_coef:       float = 0.01,
        vf_coef:        float = 0.5,
        max_grad_norm:  float = 0.5,
        n_epochs:       int   = 4,
        batch_size:     int   = 64,
    ) -> None:
        self.model         = model
        self.clip_eps      = clip_eps
        self.ent_coef      = ent_coef
        self.vf_coef       = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs      = n_epochs
        self.batch_size    = batch_size
        self.optimizer     = optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """
        Run *n_epochs* passes over the buffer and return mean losses.

        Parameters
        ----------
        buffer:
            A full buffer with ``compute_returns_and_advantages()`` already called.

        Returns
        -------
        dict with keys: policy_loss, value_loss, entropy, total_loss
        """
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}
        n_updates = 0

        for _ in range(self.n_epochs):
            for batch in buffer.get_batches(self.batch_size):
                log_probs, values, entropy = self.model.evaluate_actions(
                    batch["obs"],
                    batch["actions"],
                    batch["action_masks"],
                )

                # Probability ratio π_new / π_old  (log space → exp)
                ratio = torch.exp(log_probs - batch["old_log_probs"])

                # Clipped surrogate objective
                adv = batch["advantages"]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (mean-squared error)
                value_loss = nn.functional.mse_loss(values, batch["returns"])

                # Entropy bonus (negative because we maximise it)
                entropy_loss = entropy.mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"]  += value_loss.item()
                stats["entropy"]     += entropy_loss.item()
                stats["total_loss"]  += loss.item()
                n_updates            += 1

        for k in stats:
            stats[k] /= max(n_updates, 1)
        return stats
