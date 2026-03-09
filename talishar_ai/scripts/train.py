"""
scripts/train.py — PPO training entry-point.

Usage
-----
python -m talishar_ai.scripts.train \\
  --base-url http://localhost:8080 \\
  --p1-deck  Ira \\
  --p2-deck  Ira \\
  --p2-is-ai \\
  --total-steps 1000000 \\
  --rollout-steps 512 \\
  --lr 3e-4 \\
  --checkpoint-dir checkpoints/ \\
  --resume checkpoints/model_50000.pt

The script creates one TalisharEnv, one ActorCritic model, one PPOTrainer,
and one Trainer, then calls trainer.train(total_steps).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Allow running as `python scripts/train.py` without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from talishar_ai.game_manager import GameManager
from talishar_ai.env import TalisharEnv
from talishar_ai.models.network import ActorCritic
from talishar_ai.training.ppo import PPOTrainer
from talishar_ai.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Talishar AI via PPO.")
    p.add_argument("--base-url",       default="http://localhost:8080")
    p.add_argument("--p1-deck",        default="Ira")
    p.add_argument("--p2-deck",        default="Ira")
    p.add_argument("--p2-is-ai",       action="store_true", default=True,
                   help="Use server-side EncounterAI as opponent (default: true).")
    p.add_argument("--format",         default="cc")
    p.add_argument("--total-steps",    type=int,   default=1_000_000)
    p.add_argument("--rollout-steps",  type=int,   default=512)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--clip-eps",       type=float, default=0.2)
    p.add_argument("--ent-coef",       type=float, default=0.01)
    p.add_argument("--vf-coef",        type=float, default=0.5)
    p.add_argument("--n-epochs",       type=int,   default=4)
    p.add_argument("--batch-size",     type=int,   default=64)
    p.add_argument("--hidden",         type=int,   default=256)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--checkpoint-freq",type=int,   default=50_000)
    p.add_argument("--resume",         default=None,
                   help="Path to a checkpoint .pt file to resume from.")
    p.add_argument("--device",         default="auto",
                   help="'cpu', 'cuda', 'mps', or 'auto'.")
    return p.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(choice)


def main() -> None:
    args   = parse_args()
    device = resolve_device(args.device)

    print(f"[train] base_url={args.base_url}  p1={args.p1_deck}  p2={args.p2_deck}")
    print(f"[train] device={device}  total_steps={args.total_steps:,}")

    gm  = GameManager(base_url=args.base_url)
    env = TalisharEnv(
        game_manager=gm,
        p1_deck=args.p1_deck,
        p2_deck=args.p2_deck,
        p2_is_ai=args.p2_is_ai,
    )

    model = ActorCritic(hidden=args.hidden).to(device)
    ppo   = PPOTrainer(
        model=model,
        lr=args.lr,
        clip_eps=args.clip_eps,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
    )

    start_step = 0
    if args.resume:
        ckpt       = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        ppo.optimizer.load_state_dict(ckpt["optim_state"])
        start_step = ckpt.get("step", 0)
        print(f"[train] Resumed from {args.resume}  (step {start_step:,})")

    trainer = Trainer(
        env=env,
        model=model,
        ppo=ppo,
        rollout_steps=args.rollout_steps,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        device=device,
    )
    remaining = max(args.total_steps - start_step, 0)
    trainer.train(remaining)


if __name__ == "__main__":
    main()
