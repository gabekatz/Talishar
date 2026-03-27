"""
bc.py — Behavioral Cloning (supervised imitation learning).

What is behavioral cloning?
----------------------------
We have a dataset of (game_state, action_chosen_by_human) pairs.
The goal is to train the policy to imitate the human: given the same game
state, output the same action.

This is just multi-class classification:
  loss = cross_entropy(model_logits, human_action_index)

Unlike PPO (which needs thousands of games to explore and evaluate),
behavioral cloning works with a small number of human demonstrations and
converges quickly.  It gives the model a strong starting policy before
PPO fine-tuning.

Typical workflow
----------------
1. Collect 50–200 human games via demo_server.py or collect_demo_cli.py
2. Run train_bc.py to pre-train the policy
3. Resume PPO from the BC checkpoint:
     python -m scripts.train --resume checkpoints/bc_final.pt

MLP vs LSTM
-----------
For MLP (ActorCritic): shuffle all steps and train with random mini-batches.
For LSTM (LSTMActorCritic): preserve sequence order per game; process each
game as a contiguous sequence (same logic as PPOTrainer.update_lstm).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..data.demo_dataset import DemoDataset


class BCTrainer:
    """
    Behavioral cloning trainer.

    Parameters
    ----------
    model:
        ActorCritic or LSTMActorCritic — the policy to train.
    lr:
        Learning rate for Adam.
    device:
        Torch device.
    """

    def __init__(
        self,
        model:   nn.Module,
        lr:      float        = 1e-4,
        device:  torch.device | None = None,
        confidence_weight: bool = False,
    ) -> None:
        self.model     = model
        self.device    = device or torch.device("cpu")
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.use_lstm  = getattr(model, "use_lstm", False)
        self.confidence_weight = confidence_weight

    # ------------------------------------------------------------------
    # MLP training (shuffled mini-batches)
    # ------------------------------------------------------------------

    def _train_epoch_mlp(
        self, dataset: DemoDataset, batch_size: int
    ) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_steps = 0

        for batch in dataset.batches(batch_size, self.device, shuffle=True):
            ids = batch["card_ids"] if self.model.use_embeddings else None
            logits, _ = self.model(batch["obs"], batch["legal_mask"], ids)

            if self.confidence_weight:
                per_sample = nn.functional.cross_entropy(
                    logits, batch["actions"], reduction="none"
                )
                loss = (per_sample * batch["confidence"]).mean()
            else:
                loss = nn.functional.cross_entropy(logits, batch["actions"])

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            preds = logits.argmax(dim=-1)
            total_correct += (preds == batch["actions"]).sum().item()
            total_loss    += loss.item() * len(batch["actions"])
            total_steps   += len(batch["actions"])

        n = max(total_steps, 1)
        return {"loss": total_loss / n, "accuracy": total_correct / n}

    # ------------------------------------------------------------------
    # LSTM training (sequential, per-game)
    # ------------------------------------------------------------------

    def _train_epoch_lstm(
        self, dataset: DemoDataset, batch_size: int  # batch_size unused for LSTM
    ) -> dict[str, float]:
        self.model.train()
        total_loss    = 0.0
        total_correct = 0
        total_steps   = 0

        for game_recs in dataset.games():
            t = dataset.game_tensors(game_recs, self.device)
            ids = t["card_ids"] if self.model.use_embeddings else None

            h0, c0 = self.model.init_hidden(1, self.device)
            # No mid-game resets needed — each game is its own sequence
            ep_starts = torch.zeros(len(game_recs), dtype=torch.bool, device=self.device)

            log_probs, _, entropy = self.model.evaluate_sequence(
                t["obs"], t["legal_mask"], h0, c0, ep_starts, t["actions"], ids
            )
            # evaluate_sequence returns log_prob of the CHOSEN action.
            # BC loss = negative log-likelihood = -mean(log_prob(human_action))
            if self.confidence_weight:
                loss = -(log_probs * t["confidence"]).mean()
            else:
                loss = -log_probs.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            # Accuracy: re-run forward to get greedy predictions
            with torch.no_grad():
                h0b, c0b = self.model.init_hidden(1, self.device)
                ep0 = torch.zeros(len(game_recs), dtype=torch.bool, device=self.device)
                lp_all, _, _ = self.model.evaluate_sequence(
                    t["obs"], t["legal_mask"], h0b, c0b, ep0, t["actions"], ids
                )
            # (can't easily get greedy pred from evaluate_sequence without another pass;
            #  use loss as proxy — skip accuracy for LSTM for now)
            total_loss  += loss.item() * len(game_recs)
            total_steps += len(game_recs)

        n = max(total_steps, 1)
        return {"loss": total_loss / n, "accuracy": float("nan")}

    # ------------------------------------------------------------------
    # Public training loop
    # ------------------------------------------------------------------

    def train(
        self,
        dataset:        DemoDataset,
        n_epochs:       int  = 20,
        batch_size:     int  = 64,
        checkpoint_dir: str  = "checkpoints",
        checkpoint_freq: int = 5,
    ) -> None:
        """
        Train for *n_epochs* passes over the dataset.

        Saves checkpoints every *checkpoint_freq* epochs in the same format
        as PPO checkpoints (``model_state`` + ``optim_state`` + ``step``) so
        they can be loaded by ``train.py --resume``.
        """
        ckpt_dir = Path(checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.model.to(self.device)
        print(
            f"[BC] Training {'LSTM' if self.use_lstm else 'MLP'} policy "
            f"on {dataset.summary()} for {n_epochs} epochs"
        )

        for epoch in range(1, n_epochs + 1):
            if self.use_lstm:
                stats = self._train_epoch_lstm(dataset, batch_size)
            else:
                stats = self._train_epoch_mlp(dataset, batch_size)

            acc_str = f"{stats['accuracy']:.3f}" if not np.isnan(stats["accuracy"]) else "n/a"
            print(
                f"[BC] Epoch {epoch:>3}/{n_epochs}  "
                f"loss={stats['loss']:.4f}  acc={acc_str}"
            )

            if epoch % checkpoint_freq == 0 or epoch == n_epochs:
                path = ckpt_dir / f"bc_epoch{epoch:03d}.pt"
                torch.save(
                    {
                        "step":        0,   # no env steps yet; PPO can resume from 0
                        "model_state": self.model.state_dict(),
                        "optim_state": self.optimizer.state_dict(),
                        "bc_epoch":    epoch,
                    },
                    path,
                )
                print(f"[BC] Saved → {path}")

        final = ckpt_dir / "bc_final.pt"
        torch.save(
            {
                "step":        0,
                "model_state": self.model.state_dict(),
                "optim_state": self.optimizer.state_dict(),
                "bc_epoch":    n_epochs,
            },
            final,
        )
        print(f"[BC] Training complete. Final checkpoint → {final}")
