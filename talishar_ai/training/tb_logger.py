"""
tb_logger.py — TensorBoard logging for training metrics.

Wraps torch.utils.tensorboard.SummaryWriter with convenience methods
for the metrics both Trainer and AsyncTrainer track.

Usage:
    tensorboard --logdir runs/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter


class TBLogger:
    """Thin wrapper around SummaryWriter with domain-specific log methods."""

    def __init__(self, log_dir: str | Path = "runs") -> None:
        self._writer = SummaryWriter(log_dir=str(log_dir))

    def close(self) -> None:
        self._writer.close()

    # ------------------------------------------------------------------
    # Training loop metrics (called every log_freq updates)
    # ------------------------------------------------------------------

    def log_training(
        self,
        step: int,
        stats: dict[str, float],
        ep_rewards: list[float],
        ep_results: list[str],
        sps: float,
    ) -> None:
        """Log PPO losses, reward, win rate, and throughput."""
        w = self._writer

        # PPO losses
        w.add_scalar("losses/policy",  stats["policy_loss"], step)
        w.add_scalar("losses/value",   stats["value_loss"],  step)
        w.add_scalar("losses/entropy", stats["entropy"],     step)
        w.add_scalar("losses/total",   stats["total_loss"],  step)

        # Throughput
        w.add_scalar("perf/steps_per_sec", sps, step)

        # Episode reward
        if ep_rewards:
            import numpy as np
            w.add_scalar("episode/reward_mean", float(np.mean(ep_rewards)), step)

        # Win rate (exclude truncated games)
        real = [r for r in ep_results if r != "truncated"]
        if real:
            wins   = real.count("win")
            losses = real.count("loss")
            draws  = real.count("draw")
            n = len(real)
            w.add_scalar("episode/win_rate",  wins / n,   step)
            w.add_scalar("episode/loss_rate", losses / n,  step)
            w.add_scalar("episode/draw_rate", draws / n,   step)
            w.add_scalar("episode/n_games",   n,           step)

        n_trunc = ep_results.count("truncated")
        if n_trunc:
            w.add_scalar("episode/truncated", n_trunc, step)

    # ------------------------------------------------------------------
    # Per-game stats (called at episode end)
    # ------------------------------------------------------------------

    def log_game(self, step: int, game_stats: dict[str, Any]) -> None:
        """Log per-game statistics (damage, length, equipment, etc.)."""
        w = self._writer

        if "total_steps" in game_stats:
            w.add_scalar("game/length", game_stats["total_steps"], step)
        if "damage_dealt" in game_stats:
            w.add_scalar("game/damage_dealt", game_stats["damage_dealt"], step)
        if "damage_taken" in game_stats:
            w.add_scalar("game/damage_taken", game_stats["damage_taken"], step)
        if "deck_remaining_p1" in game_stats:
            w.add_scalar("game/deck_remaining", game_stats["deck_remaining_p1"], step)
        if "equip_lost_p1" in game_stats:
            w.add_scalar("game/equip_lost_p1", game_stats["equip_lost_p1"], step)
        if "equip_lost_p2" in game_stats:
            w.add_scalar("game/equip_lost_p2", game_stats["equip_lost_p2"], step)

    # ------------------------------------------------------------------
    # Hparams (called once at training start)
    # ------------------------------------------------------------------

    def log_hparams(self, hparams: dict[str, Any]) -> None:
        """Log hyperparameters as text for run comparison."""
        lines = [f"| {k} | {v} |" for k, v in sorted(hparams.items())]
        table = "| Param | Value |\n|---|---|\n" + "\n".join(lines)
        self._writer.add_text("hparams", table, 0)
