"""
scripts/evaluate.py — Evaluate a trained checkpoint against EncounterAI.

Usage
-----
python -m talishar_ai.scripts.evaluate \\
  --checkpoint checkpoints/model_final.pt \\
  --base-url   http://localhost:8080 \\
  --p1-deck    Ira \\
  --p2-deck    Ira \\
  --n-games    50 \\
  --log-dir    eval_logs/

Plays *n-games* complete games with a greedy policy (argmax over masked
logits), prints per-game stats, and writes results to eval_logs/:

  eval_log.csv     — one row per game
  elo_ratings.json — Elo ratings updated after each game
                     (EncounterAI is treated as a fixed-strength reference
                     opponent with a starting rating of 1000)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from talishar_ai.game_manager import GameManager
from talishar_ai.env import TalisharEnv
from talishar_ai.models.network import ActorCritic
from talishar_ai.evaluation import EloTracker, EvalLogger
from talishar_ai.evaluation.game_stats import GameStats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a Talishar AI checkpoint.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--base-url",   default="http://localhost:8080")
    p.add_argument("--p1-deck",    default="Ira")
    p.add_argument("--p2-deck",    default="Ira")
    p.add_argument("--n-games",    type=int, default=20)
    p.add_argument("--log-dir",    default="eval_logs",
                   help="Directory for eval_log.csv and elo_ratings.json.")
    p.add_argument("--device",     default="cpu")
    p.add_argument("--hidden",     type=int, default=256)
    return p.parse_args()


def play_game(
    env:    TalisharEnv,
    model:  ActorCritic,
    device: torch.device,
) -> tuple[str, GameStats]:
    """
    Play one complete game greedily.

    Returns
    -------
    (result, game_stats)
        result is "win", "loss", or "draw".
    """
    obs_np, info = env.reset()
    done = False

    while not done:
        obs  = torch.from_numpy(obs_np).unsqueeze(0).to(device)
        mask = torch.from_numpy(info["legal_mask"]).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, _ = model(obs, mask)
            action    = int(logits.argmax(dim=-1).item())

        obs_np, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    result     = info.get("result") or "draw"
    game_stats = info.get("game_stats")
    return result, game_stats


def _fmt_stats(stats: GameStats | None) -> str:
    if stats is None:
        return ""
    return (
        f"dmg={stats.damage_dealt:>2}↑ {stats.damage_taken:>2}↓  "
        f"deck={stats.deck_remaining_p1}/{stats.deck_remaining_p2}  "
        f"steps={stats.total_steps}"
        + ("  DECKOUT" if stats.deck_out   else "")
        + ("  TRUNC"   if stats.truncated  else "")
    )


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)

    ckpt  = torch.load(args.checkpoint, map_location=device)
    model = ActorCritic(hidden=args.hidden).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ckpt_step  = ckpt.get("step", 0)
    ckpt_label = f"model_{ckpt_step}"
    print(f"Loaded {args.checkpoint}  (step {ckpt_step:,})")

    gm  = GameManager(base_url=args.base_url)
    env = TalisharEnv(
        game_manager=gm,
        p1_deck=args.p1_deck,
        p2_deck=args.p2_deck,
        p2_is_ai=True,
    )

    logger = EvalLogger(log_dir=args.log_dir)
    elo    = EloTracker()
    if logger.load_elo(elo):
        print(f"Resumed Elo from {logger.elo_path}")

    wins = losses = draws = 0

    for i in range(1, args.n_games + 1):
        result, stats = play_game(env, model, device)

        if result == "win":
            wins   += 1
            elo.update(winner=ckpt_label, loser="EncounterAI")
        elif result == "loss":
            losses += 1
            elo.update(winner="EncounterAI", loser=ckpt_label)
        else:
            draws  += 1

        total    = wins + losses + draws
        win_pct  = wins / total
        my_elo   = elo.rating(ckpt_label)

        logger.log_game(global_step=ckpt_step, checkpoint=ckpt_label, stats=stats)
        logger.log_elo(elo)

        print(
            f"  Game {i:>3}: {result:4s} | "
            f"W={wins} L={losses} D={draws}  win%={win_pct:.1%}  "
            f"Elo={my_elo:.0f} | {_fmt_stats(stats)}"
        )

    # Final summary
    n = args.n_games
    print(f"\n{'─'*60}")
    print(f"Results vs EncounterAI over {n} games:")
    print(f"  Win:  {wins}  ({wins/n:.1%})")
    print(f"  Loss: {losses}  ({losses/n:.1%})")
    print(f"  Draw: {draws}  ({draws/n:.1%})")
    print(f"  Final Elo: {elo.rating(ckpt_label):.1f}")
    print(f"  Logs written to: {args.log_dir}/")

    all_stats = []
    print(f"\nAggregate stats logged to {logger.csv_path}")

    top = list(elo.all_ratings().items())
    print(f"\nElo leaderboard:")
    for name, rating in top:
        print(f"  {rating:>7.1f}  {name}")


if __name__ == "__main__":
    main()
