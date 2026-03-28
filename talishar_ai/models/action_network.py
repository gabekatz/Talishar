"""
action_network.py — Actor-Critic with action embeddings and dot-product scoring.

Instead of a flat Linear(hidden, MAX_ACTIONS) actor head, each legal action is
represented by a feature vector (action type, card stats, metadata).  The model
produces a state-query vector and scores each action via scaled dot-product:

  logits[i] = query · action_embed[i] / sqrt(query_dim)

This lets the model generalise across actions: "any ACTIVATE_EQUIPMENT action
during defense is bad" instead of memorising "action index 7 was bad."

Architecture
------------
State encoder (shared trunk):
  [obs | optional card_embs] → Linear → LayerNorm → ReLU
                              → Linear → LayerNorm → ReLU → h_state

Action encoder:
  action_feats → Linear(ACTION_DIM, action_hidden) → ReLU
               → Linear(action_hidden, query_dim) → a_emb

Actor (dot-product scoring):
  query  = Linear(hidden, query_dim)
  logits = (query · a_emb) / sqrt(query_dim)
  logits = masked_fill(~action_mask, -1e9)

Critic (state-only, unchanged):
  value = Linear(hidden, 1)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Categorical

from ..features import OBS_DIM, MAX_ACTIONS, N_CARD_SLOTS, ACTION_DIM


class ActionEmbedActorCritic(nn.Module):

    use_action_embed: bool = True  # flag for Trainer detection

    def __init__(
        self,
        obs_dim:        int  = OBS_DIM,
        action_dim:     int  = MAX_ACTIONS,
        action_feat_dim: int = ACTION_DIM,
        hidden:         int  = 256,
        query_dim:      int  = 64,
        action_hidden:  int  = 64,
        use_embeddings: bool = False,
        vocab_size:     int  = 5000,
        emb_dim:        int  = 32,
        n_card_slots:   int  = N_CARD_SLOTS,
    ) -> None:
        super().__init__()
        self.use_embeddings = use_embeddings
        self.query_dim = query_dim

        # Optional card-identity embeddings
        if use_embeddings:
            self.embedding: nn.Embedding | None = nn.Embedding(
                vocab_size, emb_dim, padding_idx=0
            )
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
            trunk_in = obs_dim + n_card_slots * emb_dim
        else:
            self.embedding = None
            trunk_in = obs_dim

        # State encoder (shared trunk)
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )

        # Action encoder: project per-action features into query space
        self.action_encoder = nn.Sequential(
            nn.Linear(action_feat_dim, action_hidden),
            nn.ReLU(),
            nn.Linear(action_hidden, query_dim),
        )

        # State → query projection for dot-product scoring
        self.query_proj = nn.Linear(hidden, query_dim)

        # Critic (state-only — V(s), not Q(s,a))
        self.critic = nn.Linear(hidden, 1)

        # Weight init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def _encode_state(
        self,
        obs: torch.Tensor,
        card_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode observation into hidden state. Returns (B, hidden)."""
        if self.embedding is not None and card_ids is not None:
            card_ids = card_ids.clamp(0, self.embedding.num_embeddings - 1)
            embs = self.embedding(card_ids)
            embs_flat = embs.reshape(embs.shape[0], -1)
            trunk_in = torch.cat([obs, embs_flat], dim=-1)
        else:
            trunk_in = obs
        return self.trunk(trunk_in)

    def _score_actions(
        self,
        h: torch.Tensor,              # (B, hidden)
        action_feats: torch.Tensor,    # (B, MAX_ACTIONS, ACTION_DIM)
        action_mask: torch.Tensor,     # (B, MAX_ACTIONS) bool
    ) -> torch.Tensor:
        """Compute masked logits via scaled dot-product. Returns (B, MAX_ACTIONS)."""
        query = self.query_proj(h)                        # (B, query_dim)
        a_emb = self.action_encoder(action_feats)         # (B, MAX_ACTIONS, query_dim)
        # Scaled dot product: (B, MAX_ACTIONS)
        logits = torch.einsum("bq,baq->ba", query, a_emb) / math.sqrt(self.query_dim)
        logits = logits.masked_fill(~action_mask, -1e9)
        return logits

    def forward(
        self,
        obs:          torch.Tensor,                # (B, OBS_DIM)
        action_mask:  torch.Tensor,                # (B, MAX_ACTIONS) bool
        action_feats: torch.Tensor,                # (B, MAX_ACTIONS, ACTION_DIM)
        card_ids:     torch.Tensor | None = None,  # (B, N_CARD_SLOTS) int64
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, value)."""
        h = self._encode_state(obs, card_ids)
        logits = self._score_actions(h, action_feats, action_mask)
        value = self.critic(h).squeeze(-1)
        return logits, value

    def act(
        self,
        obs:          torch.Tensor,
        action_mask:  torch.Tensor,
        action_feats: torch.Tensor,
        card_ids:     torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample one action. Returns (action, log_prob, value)."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            action_mask = action_mask.unsqueeze(0)
            action_feats = action_feats.unsqueeze(0)
            if card_ids is not None:
                card_ids = card_ids.unsqueeze(0)

        logits, value = self.forward(obs, action_mask, action_feats, card_ids)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.squeeze(0), log_prob.squeeze(0), value.squeeze(0)

    def evaluate_actions(
        self,
        obs:          torch.Tensor,                # (B, OBS_DIM)
        actions:      torch.Tensor,                # (B,) int64
        action_mask:  torch.Tensor,                # (B, MAX_ACTIONS) bool
        action_feats: torch.Tensor | None = None,  # (B, MAX_ACTIONS, ACTION_DIM)
        card_ids:     torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate stored actions (PPO update). Returns (log_probs, values, entropy)."""
        logits, values = self.forward(obs, action_mask, action_feats, card_ids)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy
