"""
demo_dataset.py — Storage for human demonstration data used in behavioral cloning.

Each record captures one human decision:
  obs          float32 (OBS_DIM,)      — encoded game state
  action       int                     — index the human chose from legalMoves
  legal_mask   bool (MAX_ACTIONS,)     — which slots were legal at this step
  card_ids     int64 (N_CARD_SLOTS,)   — card identity indices (0 = no embedding)
  description  str                     — human-readable description of the chosen move
  game         str                     — game name (groups steps into sequences for LSTM)
  step         int                     — step index within the game

Saved as JSON-lines (.jsonl): one JSON object per line, numpy arrays as lists.
This format is human-inspectable and easy to append to without loading the whole file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from ..features import OBS_DIM, MAX_ACTIONS, N_CARD_SLOTS


class DemoDataset:
    """
    In-memory collection of human demonstration steps.

    Usage
    -----
    # Collect
    ds = DemoDataset()
    ds.add(obs, action, legal_mask, card_ids, description="Play Surging Strike", game="127", step=3)
    ds.save("demos/session1.jsonl")

    # Train
    ds = DemoDataset.load("demos/session1.jsonl")
    for batch in ds.batches(batch_size=64, device=device):
        ...
    """

    def __init__(self) -> None:
        self._records: list[dict] = []

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def add(
        self,
        obs:         np.ndarray,       # (OBS_DIM,)
        action:      int,
        legal_mask:  np.ndarray,       # (MAX_ACTIONS,) bool
        card_ids:    np.ndarray,       # (N_CARD_SLOTS,) int64
        description: str = "",
        game:        str = "",
        step:        int = 0,
    ) -> None:
        self._records.append({
            "obs":         obs.tolist(),
            "action":      int(action),
            "legal_mask":  legal_mask.tolist(),
            "card_ids":    card_ids.tolist(),
            "description": description,
            "game":        game,
            "step":        step,
        })

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path, append: bool = False) -> None:
        """Write records to a .jsonl file.  Pass append=True to extend an existing file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with open(path, mode) as f:
            for rec in self._records:
                f.write(json.dumps(rec) + "\n")
        print(f"[DemoDataset] Saved {len(self._records)} records → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "DemoDataset":
        """Load all records from a .jsonl file."""
        ds = cls()
        path = Path(path)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    ds._records.append(json.loads(line))
        print(f"[DemoDataset] Loaded {len(ds._records)} records from {path}")
        return ds

    @classmethod
    def load_dir(cls, directory: str | Path) -> "DemoDataset":
        """Merge all .jsonl files in a directory into one dataset."""
        ds = cls()
        for p in sorted(Path(directory).glob("*.jsonl")):
            sub = cls.load(p)
            ds._records.extend(sub._records)
        print(f"[DemoDataset] Merged {len(ds._records)} total records from {directory}/")
        return ds

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def n_steps(self) -> int:
        return len(self._records)

    @property
    def n_games(self) -> int:
        return len({r["game"] for r in self._records if r["game"]})

    def summary(self) -> str:
        return f"DemoDataset: {self.n_steps} steps across {self.n_games} games"

    # ------------------------------------------------------------------
    # Batched iteration (for MLP behavioral cloning — shuffled)
    # ------------------------------------------------------------------

    def batches(
        self,
        batch_size: int,
        device:     torch.device,
        shuffle:    bool = True,
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield shuffled mini-batches of tensors for MLP training."""
        n = len(self._records)
        indices = np.random.permutation(n) if shuffle else np.arange(n)
        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            batch_recs = [self._records[i] for i in idx]
            yield {
                "obs":        torch.tensor(
                    [r["obs"]        for r in batch_recs], dtype=torch.float32
                ).to(device),
                "actions":    torch.tensor(
                    [r["action"]     for r in batch_recs], dtype=torch.long
                ).to(device),
                "legal_mask": torch.tensor(
                    [r["legal_mask"] for r in batch_recs], dtype=torch.bool
                ).to(device),
                "card_ids":   torch.tensor(
                    [r["card_ids"]   for r in batch_recs], dtype=torch.long
                ).to(device),
            }

    # ------------------------------------------------------------------
    # Per-game sequences (for LSTM behavioral cloning — ordered)
    # ------------------------------------------------------------------

    def games(self) -> list[list[dict]]:
        """Return records grouped by game, in step order."""
        game_map: dict[str, list[dict]] = {}
        for r in self._records:
            game_map.setdefault(r["game"], []).append(r)
        # Sort each game's steps by step index
        return [sorted(steps, key=lambda x: x["step"]) for steps in game_map.values()]

    def game_tensors(
        self, game_records: list[dict], device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Convert one game's records to a dict of tensors (T = len(game_records))."""
        return {
            "obs":        torch.tensor(
                [r["obs"]        for r in game_records], dtype=torch.float32
            ).to(device),
            "actions":    torch.tensor(
                [r["action"]     for r in game_records], dtype=torch.long
            ).to(device),
            "legal_mask": torch.tensor(
                [r["legal_mask"] for r in game_records], dtype=torch.bool
            ).to(device),
            "card_ids":   torch.tensor(
                [r["card_ids"]   for r in game_records], dtype=torch.long
            ).to(device),
        }
