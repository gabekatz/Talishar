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

import numpy as np
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
                    card_ids=batch.get("card_ids"),
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

    # ------------------------------------------------------------------
    # LSTM variant: process per-env sequences (no shuffling)
    # ------------------------------------------------------------------

    def update_lstm(self, buffers: list[RolloutBuffer]) -> dict[str, float]:
        """
        PPO update for the recurrent policy.

        Unlike ``update()``, we do **not** shuffle transitions across time
        because the LSTM hidden state must be replayed in sequence order.
        Instead, for each epoch we process every per-env buffer as one
        contiguous sequence and accumulate gradients across all envs.

        Advantages are normalised globally across all buffers before
        training begins (equivalent to what ``merge()`` does for the MLP
        path, but without destroying the temporal structure).

        Parameters
        ----------
        buffers:
            Per-env RolloutBuffers.  Each must have had
            ``compute_returns_and_advantages()`` called and
            ``initial_hidden_h / initial_hidden_c`` set.

        Returns
        -------
        dict with keys: policy_loss, value_loss, entropy, total_loss
        """
        device = next(self.model.parameters()).device

        # Global advantage normalisation across all envs
        all_adv = np.concatenate([b.advantages for b in buffers])
        adv_mean = float(all_adv.mean())
        adv_std  = float(all_adv.std()) + 1e-8

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}
        n_updates = 0

        for _ in range(self.n_epochs):
            for buf in buffers:
                # Tensors for this env's sequence
                obs_t    = torch.from_numpy(buf.obs).to(device)
                acts_t   = torch.from_numpy(buf.actions).to(device)
                masks_t  = torch.from_numpy(buf.action_masks).to(device)
                adv_t    = torch.from_numpy(
                    (buf.advantages - adv_mean) / adv_std
                ).to(device)
                ret_t    = torch.from_numpy(buf.returns).to(device)
                olp_t    = torch.from_numpy(buf.log_probs).to(device)
                ep_st    = torch.from_numpy(buf.episode_starts).to(device)
                ids_t    = (
                    torch.from_numpy(buf.card_ids).to(torch.int32).to(device)
                    if self.model.use_embeddings else None
                )

                # Restore initial hidden state for this env's rollout
                assert buf.initial_hidden_h is not None, (
                    "LSTM buffers must have initial_hidden_h/c set before update_lstm()"
                )
                h0 = torch.from_numpy(buf.initial_hidden_h).unsqueeze(1).to(device)
                c0 = torch.from_numpy(buf.initial_hidden_c).unsqueeze(1).to(device)

                # Re-evaluate the full sequence under the current policy
                log_probs, values, entropy = self.model.evaluate_sequence(
                    obs_t, masks_t, h0, c0, ep_st, acts_t, ids_t
                )

                ratio  = torch.exp(log_probs - olp_t)
                surr1  = ratio * adv_t
                surr2  = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = nn.functional.mse_loss(values, ret_t)
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
