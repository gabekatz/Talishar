"""
network.py — Actor-Critic neural network with action masking.

Architecture (base, use_embeddings=False)
------------------------------------------
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

Architecture (with card embeddings, use_embeddings=True)
---------------------------------------------------------
An nn.Embedding table maps each of the N_CARD_SLOTS card indices to a
dense emb_dim-float vector.  The flat embedding block is concatenated
with the obs before entering the trunk:

  Input: [obs (OBS_DIM) | card_embs (N_CARD_SLOTS × emb_dim)]
         → Linear(OBS_DIM + N_CARD_SLOTS*emb_dim, 256) → …

The embedding table uses padding_idx=0 so the PAD vector is always
zeroed-out and its gradient is suppressed.  Weights are initialised
with a small normal distribution to avoid dominating the stat features
early in training.

Backward compatibility
-----------------------
``use_embeddings=False`` (default) behaves identically to the original
network.  Existing checkpoints load without modification.  When
``use_embeddings=True``, ``card_ids`` must be passed to forward/act/
evaluate_actions; if omitted they default to None and the embedding
branch is skipped gracefully.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from ..features import OBS_DIM, MAX_ACTIONS, N_CARD_SLOTS


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim:        int  = OBS_DIM,
        action_dim:     int  = MAX_ACTIONS,
        hidden:         int  = 256,
        use_embeddings: bool = False,
        vocab_size:     int  = 5000,    # must be ≥ CardVocab.size
        emb_dim:        int  = 32,
        n_card_slots:   int  = N_CARD_SLOTS,
    ) -> None:
        super().__init__()
        self.use_embeddings = use_embeddings

        if use_embeddings:
            # padding_idx=0: PAD always maps to zeros, gradient suppressed
            self.embedding: nn.Embedding | None = nn.Embedding(
                vocab_size, emb_dim, padding_idx=0
            )
            # Small init — avoids embedding gradients overwhelming stat features
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
            trunk_in = obs_dim + n_card_slots * emb_dim
        else:
            self.embedding = None
            trunk_in = obs_dim

        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.actor  = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)

        # Orthogonal init for Linear layers (standard for PPO).
        # Skip the embedding weight (already initialised above).
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        obs:         torch.Tensor,               # (B, OBS_DIM)
        action_mask: torch.Tensor,               # (B, MAX_ACTIONS) bool
        card_ids:    torch.Tensor | None = None, # (B, N_CARD_SLOTS) int64
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logits : (B, MAX_ACTIONS)  — masked, ready for log-softmax
        value  : (B,)
        """
        if self.embedding is not None and card_ids is not None:
            # Clamp to valid vocab range — MPS can corrupt int64 values during
            # transfer, producing garbage indices that crash the embedding lookup.
            card_ids = card_ids.clamp(0, self.embedding.num_embeddings - 1)
            embs      = self.embedding(card_ids)            # (B, N_CARD_SLOTS, emb_dim)
            embs_flat = embs.reshape(embs.shape[0], -1)     # (B, N_CARD_SLOTS * emb_dim)
            trunk_in  = torch.cat([obs, embs_flat], dim=-1)
        else:
            trunk_in = obs

        h      = self.trunk(trunk_in)
        logits = self.actor(h)
        logits = logits.masked_fill(~action_mask, -1e9)
        value  = self.critic(h).squeeze(-1)
        return logits, value

    # ------------------------------------------------------------------
    # Convenience: sample one action
    # ------------------------------------------------------------------

    def act(
        self,
        obs:         torch.Tensor,               # (OBS_DIM,) or (1, OBS_DIM)
        action_mask: torch.Tensor,               # (MAX_ACTIONS,) or (1, MAX_ACTIONS)
        card_ids:    torch.Tensor | None = None, # (N_CARD_SLOTS,) or (1, N_CARD_SLOTS) int64
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
            if card_ids is not None:
                card_ids = card_ids.unsqueeze(0)

        logits, value = self.forward(obs, action_mask, card_ids)
        dist     = Categorical(logits=logits)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        return action.squeeze(0), log_prob.squeeze(0), value.squeeze(0)

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        obs:         torch.Tensor,               # (B, OBS_DIM)
        actions:     torch.Tensor,               # (B,) int64
        action_mask: torch.Tensor,               # (B, MAX_ACTIONS) bool
        action_feats: torch.Tensor | None = None,  # ignored (compat with action-embed models)
        card_ids:    torch.Tensor | None = None, # (B, N_CARD_SLOTS) int64
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-evaluate stored actions under the current policy (used in PPO update).

        Returns
        -------
        log_probs : (B,)
        values    : (B,)
        entropy   : (B,)
        """
        logits, values = self.forward(obs, action_mask, card_ids)
        dist      = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()
        return log_probs, values, entropy
