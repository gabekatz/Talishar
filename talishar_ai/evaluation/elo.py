"""
elo.py — Standard Elo rating system for tracking checkpoint strength.

Elo update rule
---------------
    expected_W = 1 / (1 + 10^((R_L - R_W) / 400))
    R_W_new    = R_W + K * (1 - expected_W)
    R_L_new    = R_L + K * (0 - (1 - expected_W))

K=32 is the standard value for normal play (FIDE uses K=20/40; K=32
gives faster convergence which suits training checkpoints that change
frequently).

Ratings are stored as a dict keyed by an arbitrary string name (typically
the checkpoint label, e.g. "model_50000" or "EncounterAI").  Unknown
names start at DEFAULT_RATING=1000.
"""

from __future__ import annotations

import json
from pathlib import Path


class EloTracker:
    DEFAULT_RATING = 1000.0
    K              = 32.0

    def __init__(self) -> None:
        self._ratings: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def rating(self, name: str) -> float:
        """Return the current Elo for *name*, initialising to DEFAULT if unseen."""
        return self._ratings.setdefault(name, self.DEFAULT_RATING)

    def all_ratings(self) -> dict[str, float]:
        """Return all ratings sorted descending by rating."""
        return dict(sorted(self._ratings.items(), key=lambda kv: -kv[1]))

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, winner: str, loser: str) -> tuple[float, float]:
        """
        Update ratings after one game.

        Parameters
        ----------
        winner, loser:
            Names of the winning and losing players (created automatically
            if not yet seen).

        Returns
        -------
        (new_winner_rating, new_loser_rating)
        """
        rw = self.rating(winner)
        rl = self.rating(loser)

        expected_w = 1.0 / (1.0 + 10.0 ** ((rl - rw) / 400.0))

        self._ratings[winner] = rw + self.K * (1.0 - expected_w)
        self._ratings[loser]  = rl + self.K * (0.0 - (1.0 - expected_w))

        return self._ratings[winner], self._ratings[loser]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write ratings to a JSON file."""
        with open(path, "w") as f:
            json.dump(self._ratings, f, indent=2)

    def load(self, path: str | Path) -> None:
        """Load ratings from a JSON file (merges into existing ratings)."""
        with open(path) as f:
            self._ratings.update(json.load(f))

    def __repr__(self) -> str:
        top = list(self.all_ratings().items())[:5]
        return f"EloTracker({top!r}{'…' if len(self._ratings) > 5 else ''})"
