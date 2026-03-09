"""
network.py — Actor-Critic neural network with action masking.

Architecture
------------
Input:  float32 observation vector of shape (OBS_DIM,)

Shared trunk
  Linear(OBS_DIM, 256) → LayerNorm(256) → ReLU
  Linear(256, 256)      → LayerNorm(256) → ReLU

Actor head (policy)
  Linear(256, MAX_ACTIONS)
  → illegal actions masked to -1e9  (applied at call time, not stored)
  → log-softmax → Categorical distribution

Critic head (value)
  Linear(256, 1)

Why this architecture?
- LayerNorm instead of BatchNorm: batch size during rollout is 1 (sequential
  steps from one game), so BN statistics would be unstable.
- Shared trunk: lets policy and value share low-level game-state features,
  reducing sample complexity.
- Action masking: never assigns probability to illegal moves, which is
  critical for a variable-action-space game.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from ..features import OBS_DIM, MAX_ACTIONS


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        action_dim: int = MAX_ACTIONS,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.actor  = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)

        # Orthogonal init (standard for PPO)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        obs: torch.Tensor,          # (B, OBS_DIM)
        action_mask: torch.Tensor,  # (B, MAX_ACTIONS) bool — True = legal
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logits : (B, MAX_ACTIONS)  — masked, ready for log-softmax
        value  : (B,)
        """
        h      = self.trunk(obs)
        logits = self.actor(h)
        # Set illegal actions to large negative so they get ~0 probability
        logits = logits.masked_fill(~action_mask, -1e9)
        value  = self.critic(h).squeeze(-1)
        return logits, value

    # ------------------------------------------------------------------
    # Convenience: sample one action
    # ------------------------------------------------------------------

    def act(
        self,
        obs: torch.Tensor,          # (OBS_DIM,) or (1, OBS_DIM)
        action_mask: torch.Tensor,  # (MAX_ACTIONS,) or (1, MAX_ACTIONS)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the masked policy.

        Returns
        -------
        action   : scalar int64 tensor
        log_prob : scalar float tensor
        value    : scalar float tensor
        """
        if obs.dim() == 1:
            obs         = obs.unsqueeze(0)
            action_mask = action_mask.unsqueeze(0)

        logits, value = self.forward(obs, action_mask)
        dist     = Categorical(logits=logits)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        return action.squeeze(0), log_prob.squeeze(0), value.squeeze(0)

    def evaluate_actions(
        self,
        obs: torch.Tensor,          # (B, OBS_DIM)
        actions: torch.Tensor,      # (B,) int64
        action_mask: torch.Tensor,  # (B, MAX_ACTIONS) bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-evaluate stored actions under the current policy (used in PPO update).

        Returns
        -------
        log_probs : (B,)
        values    : (B,)
        entropy   : (B,)
        """
        logits, values = self.forward(obs, action_mask)
        dist      = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()
        return log_probs, values, entropy
