"""
scripts/evaluate.py — Evaluate a trained checkpoint against EncounterAI.

Usage
-----
python -m talishar_ai.scripts.evaluate \\
  --checkpoint checkpoints/model_final.pt \\
  --base-url   http://localhost:8080 \\
  --p1-deck    Ira \\
  --p2-deck    Ira \\
  --n-games    50

Plays *n-games* complete games, always greedy (argmax over masked logits),
and prints a win/loss/draw breakdown.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from talishar_ai.game_manager import GameManager
from talishar_ai.env import TalisharEnv
from talishar_ai.models.network import ActorCritic
from talishar_ai.features import OBS_DIM, MAX_ACTIONS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a Talishar AI checkpoint.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--base-url",   default="http://localhost:8080")
    p.add_argument("--p1-deck",    default="Ira")
    p.add_argument("--p2-deck",    default="Ira")
    p.add_argument("--n-games",    type=int, default=20)
    p.add_argument("--device",     default="cpu")
    p.add_argument("--hidden",     type=int, default=256)
    return p.parse_args()


def play_game(
    env: TalisharEnv,
    model: ActorCritic,
    device: torch.device,
) -> str:
    """Play one complete game greedily.  Returns "win", "loss", or "draw"."""
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

    return info.get("result") or "draw"


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)

    ckpt  = torch.load(args.checkpoint, map_location=device)
    model = ActorCritic(hidden=args.hidden).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded {args.checkpoint} (step {ckpt.get('step', '?'):,})")

    gm  = GameManager(base_url=args.base_url)
    env = TalisharEnv(
        game_manager=gm,
        p1_deck=args.p1_deck,
        p2_deck=args.p2_deck,
        p2_is_ai=True,
    )

    results: Counter[str] = Counter()
    for i in range(1, args.n_games + 1):
        result = play_game(env, model, device)
        results[result] += 1
        w, l, d = results["win"], results["loss"], results["draw"]
        total   = w + l + d
        win_pct = w / total if total else 0
        print(f"  Game {i:>3}: {result:4s}  | W={w} L={l} D={d}  win%={win_pct:.1%}")

    n = args.n_games
    print(f"\nFinal  W={results['win']}  L={results['loss']}  D={results['draw']}")
    print(f"Win rate: {results['win']/n:.1%}  (vs EncounterAI, greedy policy)")


if __name__ == "__main__":
    main()
