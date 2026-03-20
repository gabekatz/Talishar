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
from talishar_ai.parallel_env import ParallelEnvManager
from talishar_ai.features import StateEncoder
from talishar_ai.card_vocab import CardVocab
from talishar_ai.deck_utils import list_decks
from talishar_ai.models.network import ActorCritic
from talishar_ai.models.lstm_network import LSTMActorCritic
from talishar_ai.training.ppo import PPOTrainer
from talishar_ai.training.trainer import Trainer
from talishar_ai.training.async_trainer import AsyncTrainer
from talishar_ai.training.self_play import SelfPlayEnv, SelfPlayManager
from talishar_ai.training.tb_logger import TBLogger


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
    p.add_argument("--ent-coef",       type=float, default=0.03)
    p.add_argument("--vf-coef",        type=float, default=0.5)
    p.add_argument("--n-epochs",       type=int,   default=4)
    p.add_argument("--batch-size",     type=int,   default=64)
    p.add_argument("--hidden",          type=int,   default=256)
    p.add_argument("--use-embeddings",  action="store_true", default=False,
                   help="Enable learned card-identity embeddings.")
    p.add_argument("--emb-dim",         type=int,   default=32,
                   help="Embedding dimension per card slot (requires --use-embeddings).")
    p.add_argument("--use-lstm",        action="store_true", default=False,
                   help="Use LSTM recurrent policy (LSTMActorCritic) instead of MLP.")
    p.add_argument("--lstm-hidden",     type=int,   default=256,
                   help="LSTM hidden size (requires --use-lstm).")
    p.add_argument("--lstm-layers",     type=int,   default=1,
                   help="Number of stacked LSTM layers (requires --use-lstm).")
    p.add_argument("--vocab-path",      default=None,
                   help="Path to card_vocab.json (built automatically if absent).")
    p.add_argument("--checkpoint-dir",  default="checkpoints")
    p.add_argument("--checkpoint-freq",type=int,   default=50_000)
    p.add_argument("--n-envs",              type=int,   default=1,
                   help="Number of parallel game environments (thread-based).")
    p.add_argument("--self-play",           action="store_true", default=False,
                   help="Train against a frozen copy of the policy (self-play).")
    p.add_argument("--opponent-update-freq",type=int,   default=10,
                   help="PPO updates between frozen-opponent rotations (self-play only).")
    p.add_argument("--async",          action="store_true", default=False,
                   dest="use_async",
                   help="Use async trainer (each env runs in its own thread).")
    p.add_argument("--resume",         default=None,
                   help="Path to a checkpoint .pt file to resume from.")
    p.add_argument("--device",         default="auto",
                   help="'cpu', 'cuda', 'mps', or 'auto'.")
    p.add_argument("--log-dir",       default="runs",
                   help="TensorBoard log directory (default: runs/). "
                        "View with: tensorboard --logdir runs/")
    p.add_argument("--random-decks",  action="store_true", default=False,
                   help="Randomly select decks from Assets/ each game. "
                        "Overrides --p1-deck/--p2-deck.")
    p.add_argument("--deck-format",   default=None,
                   help="Only use decks suitable for this format when --random-decks "
                        "is set (e.g. 'cc', 'blitz').")
    p.add_argument("--fill-from-inventory", action="store_true", default=False,
                   help="Randomly fill main deck from inventory cards each game. "
                        "Creates deck variety during training.")
    return p.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(choice)


def validate_deck(deck_name: str, assets_dir: Path) -> None:
    """Validate a deck file exists and has a playable configuration."""
    deck_path = assets_dir / f"{deck_name}.txt"
    if not deck_path.exists():
        raise SystemExit(
            f"[ERROR] Deck file not found: {deck_path}\n"
            f"  Available decks: {', '.join(p.stem for p in sorted(assets_dir.glob('*.txt')))}"
        )

    lines = deck_path.read_text().strip().splitlines()
    if len(lines) < 2:
        raise SystemExit(
            f"[ERROR] Deck '{deck_name}' is malformed — needs at least 2 lines "
            f"(hero+equipment, main deck). Found {len(lines)} line(s)."
        )

    equip = lines[0].strip().split()
    deck_cards = lines[1].strip().split()

    if not equip:
        raise SystemExit(f"[ERROR] Deck '{deck_name}' has no hero on line 1.")

    hero = equip[0]
    n_equip = len(equip) - 1  # excluding hero
    n_deck = len(deck_cards)

    print(f"[deck]   {deck_name}: hero={hero}  equipment={n_equip}  deck={n_deck}", end="")

    if n_deck < 60:
        inventory = []
        for line in lines[2:]:
            inventory.extend(line.strip().split())
        inv_count = len([c for c in inventory if c])
        print(f"  inventory={inv_count}")
        if n_deck + inv_count >= 60:
            print(
                f"[WARN]   Deck '{deck_name}' has only {n_deck} main deck cards but "
                f"{inv_count} inventory cards.\n"
                f"         The AI will play with {n_deck} cards. Consider moving "
                f"cards from inventory (lines 3+) to the main deck (line 2)."
            )
        else:
            raise SystemExit(
                f"\n[ERROR] Deck '{deck_name}' has only {n_deck} main deck cards "
                f"and {inv_count} inventory cards ({n_deck + inv_count} total).\n"
                f"  A CC deck needs at least 60 main deck cards. Edit {deck_path} "
                f"to move cards from inventory (lines 3+) to the main deck (line 2)."
            )
    else:
        print()


def main() -> None:
    args   = parse_args()
    device = resolve_device(args.device)

    # Validate decks before doing anything expensive
    assets_dir = Path(__file__).resolve().parents[2] / "Assets"
    deck_pool: list[str] | None = None

    if args.random_decks:
        fmt = args.deck_format or args.format
        deck_pool = list_decks(format_filter=fmt, assets_dir=assets_dir)
        if not deck_pool:
            raise SystemExit(
                f"[ERROR] No decks found in {assets_dir} for format '{fmt}'."
            )
        print(f"[train] Random decks ON — {len(deck_pool)} decks for format '{fmt}':")
        for name in deck_pool:
            print(f"[train]   {name}")
    else:
        print(f"[train] Validating decks...")
        validate_deck(args.p1_deck, assets_dir)
        validate_deck(args.p2_deck, assets_dir)

    if args.fill_from_inventory:
        print(f"[train] Fill from inventory ON — deck composition varies each game")

    p1_label = "random" if args.random_decks else args.p1_deck
    p2_label = "random" if args.random_decks else args.p2_deck
    print(f"[train] base_url={args.base_url}  p1={p1_label}  p2={p2_label}")
    print(f"[train] device={device}  total_steps={args.total_steps:,}  n_envs={args.n_envs}")
    print(f"[train] embeddings={'ON emb_dim=' + str(args.emb_dim) if args.use_embeddings else 'OFF'}")
    print(f"[train] policy={'LSTM hidden=' + str(args.lstm_hidden) + ' layers=' + str(args.lstm_layers) if args.use_lstm else 'MLP'}")

    # Build / load card vocab if embeddings are requested
    vocab   = CardVocab.load_or_build(args.vocab_path) if args.use_embeddings else None
    encoder = StateEncoder(vocab=vocab)

    def _make_base_env(p2_is_ai: bool) -> TalisharEnv:
        # Each env gets its own GameManager (its own requests.Session) so
        # concurrent HTTP calls don't share connection state.
        return TalisharEnv(
            game_manager=GameManager(base_url=args.base_url),
            p1_deck=args.p1_deck,
            p2_deck=args.p2_deck,
            p2_is_ai=p2_is_ai,
            deck_pool=deck_pool,
            fill_from_inventory=args.fill_from_inventory,
        )

    vocab_size = vocab.size if vocab else 5000
    if args.use_lstm:
        model = LSTMActorCritic(
            hidden         = args.hidden,
            lstm_hidden    = args.lstm_hidden,
            n_lstm_layers  = args.lstm_layers,
            use_embeddings = args.use_embeddings,
            vocab_size     = vocab_size,
            emb_dim        = args.emb_dim,
        ).to(device)
        # MPS has bugs in nn.LSTM — keep the LSTM submodule on CPU.
        # Must be done before optimizer creation so param groups are consistent.
        if device.type == "mps":
            model.lstm = model.lstm.cpu()
    else:
        model = ActorCritic(
            hidden         = args.hidden,
            use_embeddings = args.use_embeddings,
            vocab_size     = vocab_size,
            emb_dim        = args.emb_dim,
        ).to(device)

    # Build environments — self-play or EncounterAI
    sp_manager: SelfPlayManager | None = None
    if args.self_play:
        print(f"[train] Self-play ON  (opponent rotates every {args.opponent_update_freq} updates)")
        sp_envs = [
            SelfPlayEnv(
                env     = _make_base_env(p2_is_ai=False),
                active_model = model,
                device  = device,
                encoder = encoder,
            )
            for _ in range(args.n_envs)
        ]
        sp_manager = SelfPlayManager(sp_envs, update_freq=args.opponent_update_freq)
    else:
        sp_envs = None

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
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        # Re-pin LSTM to CPU after checkpoint load (MPS LSTM workaround)
        if args.use_lstm and device.type == "mps":
            model.lstm = model.lstm.cpu()
        # Rebuild optimizer so param groups reference the correct devices,
        # then load state and fixup Adam buffer devices to match params.
        ppo.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
        ppo.optimizer.load_state_dict(ckpt["optim_state"])
        for group in ppo.optimizer.param_groups:
            for p in group["params"]:
                state = ppo.optimizer.state.get(p, {})
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(p.device)
        start_step = ckpt.get("step", 0)
        print(f"[train] Resumed from {args.resume}  (step {start_step:,})")

    remaining = max(args.total_steps - start_step, 0)

    # TensorBoard logger
    tb = TBLogger(log_dir=args.log_dir)
    tb.log_hparams({
        "p1_deck": p1_label,
        "p2_deck": p2_label,
        "total_steps": args.total_steps,
        "rollout_steps": args.rollout_steps,
        "lr": args.lr,
        "clip_eps": args.clip_eps,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "hidden": args.hidden,
        "n_envs": args.n_envs,
        "self_play": args.self_play,
        "use_lstm": args.use_lstm,
        "use_embeddings": args.use_embeddings,
        "device": str(device),
    })
    print(f"[train] TensorBoard → {args.log_dir}/  (tensorboard --logdir {args.log_dir})")

    if args.use_async:
        print(f"[train] Async trainer ON")
        # AsyncTrainer takes a flat list of envs (no ParallelEnvManager)
        if sp_envs is not None:
            env_list = sp_envs
        else:
            env_list = [_make_base_env(p2_is_ai=args.p2_is_ai) for _ in range(args.n_envs)]

        trainer = AsyncTrainer(
            envs            = env_list,
            model           = model,
            ppo             = ppo,
            rollout_steps   = args.rollout_steps,
            checkpoint_dir  = args.checkpoint_dir,
            checkpoint_freq = args.checkpoint_freq,
            device          = device,
            encoder         = encoder,
            post_update_fn  = sp_manager.maybe_rotate if sp_manager else None,
            tb_logger       = tb,
        )
    else:
        # Synchronous trainer (original)
        if sp_envs is not None:
            env = ParallelEnvManager([lambda e=e: e for e in sp_envs])
        else:
            env = ParallelEnvManager([lambda: _make_base_env(p2_is_ai=args.p2_is_ai)] * args.n_envs)

        trainer = Trainer(
            env             = env,
            model           = model,
            ppo             = ppo,
            rollout_steps   = args.rollout_steps,
            checkpoint_dir  = args.checkpoint_dir,
            checkpoint_freq = args.checkpoint_freq,
            device          = device,
            encoder         = encoder,
            post_update_fn  = sp_manager.maybe_rotate if sp_manager else None,
            tb_logger       = tb,
        )

    trainer.train(remaining, start_step=start_step)


if __name__ == "__main__":
    main()
