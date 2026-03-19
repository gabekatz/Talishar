"""
logger.py — Persistent logging for evaluation results.

EvalLogger writes two files:
  eval_log.csv      — one row per evaluated game (append-only)
  elo_ratings.json  — current Elo ratings for all named agents (overwritten)

CSV columns
-----------
global_step, checkpoint, result, total_steps,
damage_dealt, damage_taken,
deck_remaining_p1, deck_remaining_p2,
deck_out, truncated
"""

from __future__ import annotations

import csv
from pathlib import Path

from .elo import EloTracker
from .game_stats import GameStats


_CSV_HEADER = [
    "global_step",
    "checkpoint",
    "result",
    "total_steps",
    "damage_dealt",
    "damage_taken",
    "deck_remaining_p1",
    "deck_remaining_p2",
    "deck_out",
    "truncated",
]


class EvalLogger:
    """
    Parameters
    ----------
    log_dir:
        Directory where eval_log.csv and elo_ratings.json are written.
        Created if it does not exist.
    """

    def __init__(self, log_dir: str) -> None:
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self._dir / "eval_log.csv"
        self.elo_path = self._dir / "elo_ratings.json"

        # Write CSV header on first use
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(_CSV_HEADER)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def log_game(
        self,
        global_step: int,
        checkpoint:  str,
        stats:       GameStats,
    ) -> None:
        """Append one game result row to the CSV."""
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                global_step,
                checkpoint,
                stats.result,
                stats.total_steps,
                stats.damage_dealt,
                stats.damage_taken,
                stats.deck_remaining_p1,
                stats.deck_remaining_p2,
                int(stats.deck_out),
                int(stats.truncated),
            ])

    def log_elo(self, tracker: EloTracker) -> None:
        """Overwrite the Elo JSON with current ratings."""
        tracker.save(self.elo_path)

    # ------------------------------------------------------------------
    # Reading (for resume)
    # ------------------------------------------------------------------

    def load_elo(self, tracker: EloTracker) -> bool:
        """
        Load saved Elo ratings into *tracker* if the file exists.

        Returns True if ratings were loaded, False if no file found.
        """
        if self.elo_path.exists():
            tracker.load(self.elo_path)
            return True
        return False

    def __repr__(self) -> str:
        return f"EvalLogger(log_dir={str(self._dir)!r})"
