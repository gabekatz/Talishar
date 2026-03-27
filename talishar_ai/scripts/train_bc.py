"""
train_bc.py — Train a policy via behavioral cloning from human demonstrations.

Usage
-----
# Train an MLP from demos:
uv run python -m scripts.train_bc --demos demos/ --checkpoint-dir checkpoints/

# Train an LSTM with embeddings:
uv run python -m scripts.train_bc --demos demos/ --use-lstm --use-embeddings

# Then resume PPO fine-tuning from the BC checkpoint:
uv run python -m scripts.train --resume checkpoints/bc_final.pt --use-lstm --use-embeddings

The --demos argument accepts either a single .jsonl file or a directory of them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from talishar_ai.card_vocab import CardVocab
from talishar_ai.data.demo_dataset import DemoDataset
from talishar_ai.features import StateEncoder
from talishar_ai.models.network import ActorCritic
from talishar_ai.models.lstm_network import LSTMActorCritic
from talishar_ai.training.bc import BCTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Behavioral cloning from human demos.")
    p.add_argument("--demos",           required=True,
                   help="Path to a .jsonl file or directory of .jsonl files.")
    p.add_argument("--checkpoint-dir",  default="checkpoints")
    p.add_argument("--checkpoint-freq", type=int,   default=5,
                   help="Save checkpoint every N epochs.")
    p.add_argument("--n-epochs",        type=int,   default=20)
    p.add_argument("--batch-size",      type=int,   default=64)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--hidden",          type=int,   default=256)
    p.add_argument("--use-embeddings",  action="store_true", default=False)
    p.add_argument("--emb-dim",         type=int,   default=32)
    p.add_argument("--vocab-path",      default=None)
    p.add_argument("--use-lstm",        action="store_true", default=False)
    p.add_argument("--lstm-hidden",     type=int,   default=256)
    p.add_argument("--lstm-layers",     type=int,   default=1)
    p.add_argument("--device",          default="auto")

    # LLM data quality flags
    p.add_argument("--confidence-weight", action="store_true", default=False,
                   help="Weight BC loss by LLM confidence (higher confidence = stronger signal).")
    p.add_argument("--drop-trivial",     action="store_true", default=False,
                   help="Remove trivial decisions (only 1 legal move) from training data.")
    p.add_argument("--min-confidence",   type=float, default=0.0,
                   help="Drop records with LLM confidence below this threshold (e.g. 0.7).")
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

    # Load demos
    demos_path = Path(args.demos)
    if demos_path.is_dir():
        dataset = DemoDataset.load_dir(demos_path)
    else:
        dataset = DemoDataset.load(demos_path)

    if dataset.n_steps == 0:
        print("No demo records found — collect demos with demo_server.py first.")
        return

    print(f"[train_bc] {dataset.summary()}")

    # Optional LLM data quality filtering
    if args.drop_trivial or args.min_confidence > 0:
        dataset = dataset.filter_llm(
            drop_trivial=args.drop_trivial,
            min_confidence=args.min_confidence,
        )
        if dataset.n_steps == 0:
            print("All records were filtered out — adjust thresholds.")
            return

    print(f"[train_bc] device={device}  epochs={args.n_epochs}  lr={args.lr}")
    if args.confidence_weight:
        print("[train_bc] Confidence-weighted loss ENABLED")

    # Build vocab / encoder if embeddings enabled
    vocab      = CardVocab.load_or_build(args.vocab_path) if args.use_embeddings else None
    vocab_size = vocab.size if vocab else 5000

    # Build model — same architecture choices as train.py
    if args.use_lstm:
        model = LSTMActorCritic(
            hidden         = args.hidden,
            lstm_hidden    = args.lstm_hidden,
            n_lstm_layers  = args.lstm_layers,
            use_embeddings = args.use_embeddings,
            vocab_size     = vocab_size,
            emb_dim        = args.emb_dim,
        ).to(device)
        print(f"[train_bc] Model: LSTMActorCritic (hidden={args.hidden}, lstm_hidden={args.lstm_hidden})")
    else:
        model = ActorCritic(
            hidden         = args.hidden,
            use_embeddings = args.use_embeddings,
            vocab_size     = vocab_size,
            emb_dim        = args.emb_dim,
        ).to(device)
        print(f"[train_bc] Model: ActorCritic (hidden={args.hidden})")

    trainer = BCTrainer(
        model=model, lr=args.lr, device=device,
        confidence_weight=args.confidence_weight,
    )
    trainer.train(
        dataset         = dataset,
        n_epochs        = args.n_epochs,
        batch_size      = args.batch_size,
        checkpoint_dir  = args.checkpoint_dir,
        checkpoint_freq = args.checkpoint_freq,
    )


if __name__ == "__main__":
    main()
