"""
lstm_network.py — LSTM-based Actor-Critic for partial-observability RL.

Why LSTM?
---------
Flesh and Blood is a game of hidden information: the opponent's hand, the
deck order, and whether they're holding a reaction are all unobservable.
An MLP policy must infer this from a single frame; an LSTM can accumulate
evidence across many turns and maintain a hidden memory of what has been
played and what has not.

Architecture
------------
                                    ┌── [optional] card embeddings ──┐
Input: obs (B, OBS_DIM)             │  nn.Embedding → flatten         │
       card_ids (B, N_CARD_SLOTS)   └────────────────────────────────┘
                                                   ↓
  [obs | card_embs]  ──→  Linear(in, hidden) → LayerNorm → ReLU   (pre-LSTM encoder)
                                                   ↓
  LSTM(hidden, lstm_hidden, n_layers, batch_first=True)            (recurrent memory)
                                                   ↓
  ┌─ actor  ──→ Linear(lstm_hidden, MAX_ACTIONS) → mask → Categorical ─┐
  └─ critic ──→ Linear(lstm_hidden, 1)                                  ┘

Rollout collection (one env-step at a time, batched over N envs)
-----------------------------------------------------------------
  logits, value, new_h, new_c = model(obs, mask, h, c, card_ids)

PPO update (full T-step sequence, one env at a time)
----------------------------------------------------
  log_probs, values, entropy = model.evaluate_sequence(
      obs, masks, h0, c0, episode_starts, actions, card_ids
  )
  *episode_starts* is a bool tensor (T,): True at positions where the
  hidden state should be reset to zeros (new episode started).

Backward compatibility
----------------------
``ActorCritic`` (non-LSTM) remains unchanged.  ``LSTMActorCritic`` has
``use_lstm = True`` so Trainer can detect it at runtime.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from ..features import OBS_DIM, MAX_ACTIONS, N_CARD_SLOTS


class LSTMActorCritic(nn.Module):
    """
    Actor-Critic with LSTM memory core.

    Parameters
    ----------
    obs_dim:
        Observation feature dimension (default: OBS_DIM from features.py).
    action_dim:
        Number of discrete actions (default: MAX_ACTIONS).
    hidden:
        Width of the pre-LSTM linear encoder.
    lstm_hidden:
        Hidden size of the LSTM cell.
    n_lstm_layers:
        Number of stacked LSTM layers.
    use_embeddings:
        Whether to append learned card-identity embeddings to the obs.
    vocab_size:
        Embedding table rows (must be ≥ CardVocab.size when use_embeddings=True).
    emb_dim:
        Embedding dimension per card slot.
    n_card_slots:
        Number of card slots encoded by StateEncoder.
    """

    use_lstm: bool = True  # flag for Trainer / PPOTrainer detection

    def __init__(
        self,
        obs_dim:       int  = OBS_DIM,
        action_dim:    int  = MAX_ACTIONS,
        hidden:        int  = 256,
        lstm_hidden:   int  = 256,
        n_lstm_layers: int  = 1,
        use_embeddings: bool = False,
        vocab_size:    int  = 5000,
        emb_dim:       int  = 32,
        n_card_slots:  int  = N_CARD_SLOTS,
    ) -> None:
        super().__init__()

        self.use_embeddings = use_embeddings
        self.lstm_hidden    = lstm_hidden
        self.n_lstm_layers  = n_lstm_layers

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

        # Pre-LSTM feature extractor (single layer — keeps gradient paths short)
        self.pre_lstm = nn.Sequential(
            nn.Linear(encoder_in, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )

        # Recurrent core (batch_first=True: input shape (B, T, hidden))
        self.lstm = nn.LSTM(
            input_size   = hidden,
            hidden_size  = lstm_hidden,
            num_layers   = n_lstm_layers,
            batch_first  = True,
        )

        # Output heads
        self.actor  = nn.Linear(lstm_hidden, action_dim)
        self.critic = nn.Linear(lstm_hidden, 1)

        # Weight init: orthogonal for Linear layers, leave LSTM at default
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
        """Return zero (h, c) for *batch_size* independent sequences."""
        zeros = torch.zeros(self.n_lstm_layers, batch_size, self.lstm_hidden, device=device)
        return zeros, zeros.clone()

    def _lstm_forward(
        self,
        inp: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run self.lstm, routing through CPU when on MPS to avoid Metal bugs.

        MPS has known issues with nn.LSTM (dimension mismatches in internal
        gate operations).  The LSTM is a tiny fraction of total compute — the
        bottleneck is HTTP I/O — so the CPU detour has negligible impact.
        """
        orig_device = inp.device
        if orig_device.type == "mps":
            out, (h_new, c_new) = self.lstm(
                inp.cpu(), (h.cpu(), c.cpu())
            )
            return (
                out.to(orig_device),
                h_new.to(orig_device),
                c_new.to(orig_device),
            )
        out, (h_new, c_new) = self.lstm(inp, (h, c))
        return out, h_new, c_new

    # ------------------------------------------------------------------
    # Internal feature encoder (shared by forward + evaluate_sequence)
    # ------------------------------------------------------------------

    def _encode_features(
        self,
        obs:      torch.Tensor,               # (B, obs_dim)
        card_ids: torch.Tensor | None = None, # (B, n_card_slots) int64
    ) -> torch.Tensor:                        # (B, hidden)
        if self.embedding is not None and card_ids is not None:
            # Clamp to valid vocab range — MPS can corrupt int64 values during
            # transfer, producing garbage indices that crash the embedding lookup.
            card_ids = card_ids.clamp(0, self.embedding.num_embeddings - 1)
            embs     = self.embedding(card_ids)           # (B, n_card_slots, emb_dim)
            embs_flat = embs.reshape(embs.shape[0], -1)   # (B, n_card_slots * emb_dim)
            x = torch.cat([obs, embs_flat], dim=-1)
        else:
            x = obs
        return self.pre_lstm(x)  # (B, hidden)

    # ------------------------------------------------------------------
    # Forward (one step, batched over B envs)
    # ------------------------------------------------------------------

    def forward(
        self,
        obs:         torch.Tensor,               # (B, obs_dim)
        action_mask: torch.Tensor,               # (B, MAX_ACTIONS) bool
        h:           torch.Tensor,               # (n_layers, B, lstm_hidden)
        c:           torch.Tensor,               # (n_layers, B, lstm_hidden)
        card_ids:    torch.Tensor | None = None, # (B, n_card_slots) int64
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single-step inference for B parallel environments.

        Returns
        -------
        logits  : (B, MAX_ACTIONS)  — masked, ready for Categorical
        value   : (B,)
        new_h   : (n_layers, B, lstm_hidden)
        new_c   : (n_layers, B, lstm_hidden)
        """
        feats    = self._encode_features(obs, card_ids)     # (B, hidden)
        lstm_in  = feats.unsqueeze(1)                       # (B, 1, hidden) — T=1
        out, new_h, new_c = self._lstm_forward(lstm_in, h, c)
        out      = out.squeeze(1)                           # (B, lstm_hidden)

        logits = self.actor(out).masked_fill(~action_mask, -1e9)
        value  = self.critic(out).squeeze(-1)               # (B,)
        return logits, value, new_h, new_c

    # ------------------------------------------------------------------
    # Sequence evaluation (PPO update — one env's full rollout at a time)
    # ------------------------------------------------------------------

    def evaluate_sequence(
        self,
        obs:            torch.Tensor,               # (T, obs_dim)
        action_masks:   torch.Tensor,               # (T, MAX_ACTIONS) bool
        h0:             torch.Tensor,               # (n_layers, 1, lstm_hidden)
        c0:             torch.Tensor,               # (n_layers, 1, lstm_hidden)
        episode_starts: torch.Tensor,               # (T,) bool
        actions:        torch.Tensor,               # (T,) int64
        card_ids:       torch.Tensor | None = None, # (T, n_card_slots) int64
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-evaluate stored actions for a full rollout sequence.

        The LSTM hidden state is reset to zeros wherever *episode_starts* is
        True, correctly handling mid-rollout episode boundaries.

        Returns
        -------
        log_probs : (T,)
        values    : (T,)
        entropy   : (T,)
        """
        T      = obs.shape[0]
        feats  = self._encode_features(obs, card_ids)  # (T, hidden)

        # Find segment boundaries: indices where new episodes start
        # Each segment is processed as a single contiguous LSTM call.
        ep_starts_cpu = episode_starts.cpu().numpy().tolist()
        boundaries = [0]
        for t in range(1, T):
            if ep_starts_cpu[t]:
                boundaries.append(t)
        boundaries.append(T)

        h, c = h0, c0
        # If the very first step begins a new episode, wipe the initial state
        if ep_starts_cpu[0]:
            h = torch.zeros_like(h)
            c = torch.zeros_like(c)

        all_lstm_out: list[torch.Tensor] = []
        for seg_idx in range(len(boundaries) - 1):
            s, e = boundaries[seg_idx], boundaries[seg_idx + 1]

            # All segments after the first start new episodes → reset
            if seg_idx > 0:
                h = torch.zeros_like(h)
                c = torch.zeros_like(c)

            # feats[s:e]: (seg_len, hidden) → (1, seg_len, hidden) for batch=1
            seg_in = feats[s:e].unsqueeze(0)
            seg_out, h, c = self._lstm_forward(seg_in, h, c)
            all_lstm_out.append(seg_out.squeeze(0))       # (seg_len, lstm_hidden)

        lstm_out = torch.cat(all_lstm_out, dim=0)          # (T, lstm_hidden)

        logits = self.actor(lstm_out).masked_fill(~action_masks, -1e9)
        values = self.critic(lstm_out).squeeze(-1)

        dist      = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()
        return log_probs, values, entropy
