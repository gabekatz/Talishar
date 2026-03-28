"""
action_lstm_network.py — LSTM Actor-Critic with action embedding scoring.

Combines the LSTM recurrent memory (for hidden information reasoning) with
action embeddings (for generalising across action types).  The LSTM hidden
state serves as the "query" for attention over action embeddings.

Architecture
------------
  [obs | card_embs] → Linear → LayerNorm → ReLU   (pre-LSTM encoder)
                                    ↓
  LSTM(hidden, lstm_hidden, n_layers)              (recurrent memory)
                                    ↓
  query = Linear(lstm_hidden, query_dim)           (state query)
  a_emb = ActionEncoder(action_feats)              (per-action keys)
  logits = query · a_emb / sqrt(query_dim)         (scaled dot-product)
  value  = Linear(lstm_hidden, 1)                  (state value)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Categorical

from ..features import OBS_DIM, MAX_ACTIONS, N_CARD_SLOTS, ACTION_DIM


class LSTMActionEmbedActorCritic(nn.Module):

    use_lstm: bool = True
    use_action_embed: bool = True

    def __init__(
        self,
        obs_dim:        int  = OBS_DIM,
        action_dim:     int  = MAX_ACTIONS,
        action_feat_dim: int = ACTION_DIM,
        hidden:         int  = 256,
        lstm_hidden:    int  = 256,
        n_lstm_layers:  int  = 1,
        query_dim:      int  = 64,
        action_hidden:  int  = 64,
        use_embeddings: bool = False,
        vocab_size:     int  = 5000,
        emb_dim:        int  = 32,
        n_card_slots:   int  = N_CARD_SLOTS,
    ) -> None:
        super().__init__()

        self.use_embeddings = use_embeddings
        self.lstm_hidden = lstm_hidden
        self.n_lstm_layers = n_lstm_layers
        self.query_dim = query_dim

        # Optional card-identity embeddings
        if use_embeddings:
            self.embedding: nn.Embedding | None = nn.Embedding(
                vocab_size, emb_dim, padding_idx=0
            )
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
            encoder_in = obs_dim + n_card_slots * emb_dim
        else:
            self.embedding = None
            encoder_in = obs_dim

        # Pre-LSTM encoder
        self.pre_lstm = nn.Sequential(
            nn.Linear(encoder_in, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )

        # Recurrent core
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=lstm_hidden,
            num_layers=n_lstm_layers,
            batch_first=True,
        )

        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_feat_dim, action_hidden),
            nn.ReLU(),
            nn.Linear(action_hidden, query_dim),
        )

        # State → query for dot-product scoring
        self.query_proj = nn.Linear(lstm_hidden, query_dim)

        # Critic (state-only)
        self.critic = nn.Linear(lstm_hidden, 1)

        # Weight init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Hidden-state helpers
    # ------------------------------------------------------------------

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros(self.n_lstm_layers, batch_size, self.lstm_hidden, device=device)
        return zeros, zeros.clone()

    def _lstm_forward(
        self, inp: torch.Tensor, h: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run LSTM, routing through CPU on MPS to avoid Metal bugs."""
        orig_device = inp.device
        if orig_device.type == "mps":
            out, (h_new, c_new) = self.lstm(inp.cpu(), (h.cpu(), c.cpu()))
            return out.to(orig_device), h_new.to(orig_device), c_new.to(orig_device)
        out, (h_new, c_new) = self.lstm(inp, (h, c))
        return out, h_new, c_new

    def _encode_features(
        self, obs: torch.Tensor, card_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.embedding is not None and card_ids is not None:
            card_ids = card_ids.clamp(0, self.embedding.num_embeddings - 1)
            embs = self.embedding(card_ids)
            embs_flat = embs.reshape(embs.shape[0], -1)
            x = torch.cat([obs, embs_flat], dim=-1)
        else:
            x = obs
        return self.pre_lstm(x)

    def _score_actions(
        self,
        h: torch.Tensor,              # (B, lstm_hidden)
        action_feats: torch.Tensor,    # (B, MAX_ACTIONS, ACTION_DIM)
        action_mask: torch.Tensor,     # (B, MAX_ACTIONS) bool
    ) -> torch.Tensor:
        query = self.query_proj(h)                        # (B, query_dim)
        a_emb = self.action_encoder(action_feats)         # (B, MAX_ACTIONS, query_dim)
        logits = torch.einsum("bq,baq->ba", query, a_emb) / math.sqrt(self.query_dim)
        logits = logits.masked_fill(~action_mask, -1e9)
        return logits

    # ------------------------------------------------------------------
    # Forward (one step, batched over B envs)
    # ------------------------------------------------------------------

    def forward(
        self,
        obs:          torch.Tensor,                # (B, OBS_DIM)
        action_mask:  torch.Tensor,                # (B, MAX_ACTIONS) bool
        h:            torch.Tensor,                # (n_layers, B, lstm_hidden)
        c:            torch.Tensor,                # (n_layers, B, lstm_hidden)
        action_feats: torch.Tensor,                # (B, MAX_ACTIONS, ACTION_DIM)
        card_ids:     torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (logits, value, new_h, new_c)."""
        feats = self._encode_features(obs, card_ids)
        lstm_in = feats.unsqueeze(1)  # (B, 1, hidden)
        out, new_h, new_c = self._lstm_forward(lstm_in, h, c)
        out = out.squeeze(1)  # (B, lstm_hidden)

        logits = self._score_actions(out, action_feats, action_mask)
        value = self.critic(out).squeeze(-1)
        return logits, value, new_h, new_c

    # ------------------------------------------------------------------
    # Sequence evaluation (PPO update)
    # ------------------------------------------------------------------

    def evaluate_sequence(
        self,
        obs:            torch.Tensor,               # (T, OBS_DIM)
        action_masks:   torch.Tensor,               # (T, MAX_ACTIONS) bool
        h0:             torch.Tensor,               # (n_layers, 1, lstm_hidden)
        c0:             torch.Tensor,               # (n_layers, 1, lstm_hidden)
        episode_starts: torch.Tensor,               # (T,) bool
        actions:        torch.Tensor,               # (T,) int64
        action_feats:   torch.Tensor,                # (T, MAX_ACTIONS, ACTION_DIM)
        card_ids:       torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate stored actions for a full rollout sequence."""
        T = obs.shape[0]
        feats = self._encode_features(obs, card_ids)

        ep_starts_cpu = episode_starts.cpu().numpy().tolist()
        boundaries = [0]
        for t in range(1, T):
            if ep_starts_cpu[t]:
                boundaries.append(t)
        boundaries.append(T)

        h, c = h0, c0
        if ep_starts_cpu[0]:
            h = torch.zeros_like(h)
            c = torch.zeros_like(c)

        all_lstm_out: list[torch.Tensor] = []
        for seg_idx in range(len(boundaries) - 1):
            s, e = boundaries[seg_idx], boundaries[seg_idx + 1]
            if seg_idx > 0:
                h = torch.zeros_like(h)
                c = torch.zeros_like(c)
            seg_in = feats[s:e].unsqueeze(0)
            seg_out, h, c = self._lstm_forward(seg_in, h, c)
            all_lstm_out.append(seg_out.squeeze(0))

        lstm_out = torch.cat(all_lstm_out, dim=0)  # (T, lstm_hidden)

        # Score actions at each timestep
        a_emb = self.action_encoder(action_feats)  # (T, MAX_ACTIONS, query_dim)
        query = self.query_proj(lstm_out)           # (T, query_dim)
        logits = torch.einsum("tq,taq->ta", query, a_emb) / math.sqrt(self.query_dim)
        logits = logits.masked_fill(~action_masks, -1e9)

        values = self.critic(lstm_out).squeeze(-1)

        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy
