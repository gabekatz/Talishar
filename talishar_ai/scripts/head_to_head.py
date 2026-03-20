"""
scripts/head_to_head.py — Evaluate two checkpoints against each other.

Usage
-----
python -m talishar_ai.scripts.head_to_head \\
  --ckpt-a checkpoints/model_50000.pt \\
  --ckpt-b checkpoints/model_100000.pt \\
  --base-url http://localhost:8080 \\
  --p1-deck  Ira \\
  --p2-deck  Ira \\
  --n-games  50 \\
  --log-dir  eval_logs/

Plays *n-games* complete games, alternating which checkpoint plays P1
to cancel out first-player advantage.  Both players are driven by greedy
(argmax) policies.

Since P2 is not an EncounterAI, this script drives both players directly
via GameManager (bypassing TalisharEnv's single-player loop).

Output
------
- Prints per-game result and Elo after each game.
- Appends to eval_logs/eval_log.csv (as checkpoint A's perspective).
- Updates eval_logs/elo_ratings.json with both checkpoint Elo ratings.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from talishar_ai.game_manager import GameManager
from talishar_ai.features import StateEncoder
from talishar_ai.models.network import ActorCritic
from talishar_ai.models.lstm_network import LSTMActorCritic
from talishar_ai.evaluation import EloTracker, EvalLogger
from talishar_ai.evaluation.game_stats import GameStats, GameStatsCollector


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _hp(state: dict, side: str) -> int:
    return int((state.get(side) or {}).get("health", 0) or 0)

def _deck(state: dict, side: str) -> int:
    return int((state.get(side) or {}).get("deckCount", 0) or 0)

def _is_terminal(state: dict) -> bool:
    phase = (state.get("phase") or {}).get("turnPhase", "")
    return (
        phase == "OVER"
        or _hp(state, "myState")    <= 0
        or _hp(state, "theirState") <= 0
    )

def _result_from_p1(state_for_p1: dict) -> str:
    """Return result string from P1's perspective given P1's state view."""
    my_hp  = _hp(state_for_p1, "myState")
    opp_hp = _hp(state_for_p1, "theirState")
    if opp_hp <= 0 and my_hp > 0:
        return "win"
    if my_hp <= 0 and opp_hp > 0:
        return "loss"
    return "draw"

def _greedy_action(
    model:   ActorCritic | LSTMActorCritic,
    state:   dict,
    encoder: StateEncoder,
    device:  torch.device,
    hidden:  tuple | None = None,
) -> tuple[int, tuple | None]:
    obs  = encoder.encode(state)
    mask = encoder.action_mask(state)
    obs_t  = torch.from_numpy(obs).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)
    ids_t  = None
    if getattr(model, "use_embeddings", False):
        ids = encoder.card_ids(state)
        ids_t = torch.from_numpy(ids).unsqueeze(0).to(torch.int32).to(device)
    with torch.no_grad():
        is_lstm = getattr(model, "use_lstm", False)
        if is_lstm and hidden is not None:
            logits, _, hh, hc = model(obs_t, mask_t, hidden[0], hidden[1], ids_t)
            hidden = (hh, hc)
        else:
            logits, _ = model(obs_t, mask_t, ids_t)
    return int(logits.argmax(dim=-1).item()), hidden


# -----------------------------------------------------------------------
# Core game loop
# -----------------------------------------------------------------------

def play_game(
    gm:        GameManager,
    model_p1:  ActorCritic,
    model_p2:  ActorCritic,
    p1_deck:   str,
    p2_deck:   str,
    encoder:   StateEncoder,
    device:    torch.device,
    max_steps: int = 2000,
    poll_sleep: float = 0.1,
) -> tuple[str, GameStats]:
    """
    Play one game driving both players with greedy policies.

    Returns
    -------
    (result_for_p1, game_stats)
        result_for_p1: "win" | "loss" | "draw" from P1's perspective.
        game_stats: stats computed from P1's state view.
    """
    game_name, p1_key, p2_key = gm.create_game(
        p1_deck=p1_deck, p2_deck=p2_deck, p2_is_ai=False
    )
    keys   = {1: p1_key, 2: p2_key}
    models = {1: model_p1, 2: model_p2}

    # Per-player LSTM hidden state
    hiddens: dict[int, tuple | None] = {1: None, 2: None}
    for pid, mdl in models.items():
        if getattr(mdl, "use_lstm", False):
            hiddens[pid] = mdl.init_hidden(1, device)

    stats = GameStatsCollector()

    # Fetch initial state for stats baseline (from P1's perspective)
    p1_init = gm.get_state(game_name, 1, p1_key)
    stats.reset(p1_init)

    step  = 0
    stale = 0   # consecutive rounds where neither player had priority

    while step < max_steps:
        acted = False

        for player_id in [1, 2]:
            state = gm.get_state(game_name, player_id, keys[player_id])

            if _is_terminal(state):
                # Fetch P1's view for accurate result
                p1_state = gm.get_state(game_name, 1, p1_key)
                stats.step(p1_state)
                result = _result_from_p1(p1_state)
                return result, stats.finalize(result, truncated=False)

            if not state.get("havePriority") or not state.get("legalMoves"):
                continue

            action, hiddens[player_id] = _greedy_action(
                models[player_id], state, encoder, device, hiddens[player_id],
            )
            move   = state["legalMoves"][action]
            gm.submit_action(game_name, player_id, keys[player_id], move["params"])
            step  += 1
            stale  = 0
            acted  = True

            # Update stats from P1's view each time either player acts
            if player_id == 2:
                p1_state = gm.get_state(game_name, 1, p1_key)
                stats.step(p1_state)
            else:
                stats.step(state)

            break  # restart priority check from P1 after any action

        if not acted:
            stale += 1
            if stale > 40:
                # Engine is stuck (shouldn't happen); bail out
                break
            time.sleep(poll_sleep)

    # Truncated
    p1_state = gm.get_state(game_name, 1, p1_key)
    result   = _result_from_p1(p1_state)
    return result, stats.finalize(result, truncated=True)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Head-to-head evaluation of two checkpoints."
    )
    p.add_argument("--ckpt-a",   required=True, help="Path to checkpoint A (.pt)")
    p.add_argument("--ckpt-b",   required=True, help="Path to checkpoint B (.pt)")
    p.add_argument("--base-url", default="http://localhost:8080")
    p.add_argument("--p1-deck",  default="Ira")
    p.add_argument("--p2-deck",  default="Ira")
    p.add_argument("--n-games",  type=int, default=20,
                   help="Total games played (split evenly between A-as-P1 and B-as-P1).")
    p.add_argument("--log-dir",  default="eval_logs")
    p.add_argument("--device",   default="cpu")
    p.add_argument("--hidden",   type=int, default=256)
    return p.parse_args()


def _load_model(path: str, hidden: int, device: torch.device) -> tuple[ActorCritic | LSTMActorCritic, int, str]:
    ckpt = torch.load(path, map_location=device)
    sd   = ckpt["model_state"]
    has_lstm   = any("lstm" in k for k in sd)
    has_emb    = "embedding.weight" in sd
    emb_dim    = sd["embedding.weight"].shape[1] if has_emb else 32
    vocab_size = sd["embedding.weight"].shape[0] if has_emb else 5000
    if has_lstm:
        model = LSTMActorCritic(
            use_embeddings=has_emb, emb_dim=emb_dim, vocab_size=vocab_size,
        ).to(device)
    else:
        model = ActorCritic(
            hidden=hidden, use_embeddings=has_emb, emb_dim=emb_dim,
            vocab_size=vocab_size,
        ).to(device)
    model.load_state_dict(sd)
    model.eval()
    step  = ckpt.get("step", 0)
    label = f"model_{step}"
    return model, step, label


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)

    model_a, step_a, label_a = _load_model(args.ckpt_a, args.hidden, device)
    model_b, step_b, label_b = _load_model(args.ckpt_b, args.hidden, device)
    print(f"Model A: {label_a}  ({args.ckpt_a})")
    print(f"Model B: {label_b}  ({args.ckpt_b})")

    gm      = GameManager(base_url=args.base_url)
    encoder = StateEncoder()
    logger  = EvalLogger(log_dir=args.log_dir)
    elo     = EloTracker()
    if logger.load_elo(elo):
        print(f"Resumed Elo from {logger.elo_path}")

    a_wins = b_wins = draws = 0

    for i in range(1, args.n_games + 1):
        # Alternate who plays P1 to cancel first-player advantage
        a_is_p1 = (i % 2 == 1)
        p1_model, p2_model = (model_a, model_b) if a_is_p1 else (model_b, model_a)

        result_for_p1, stats = play_game(
            gm        = gm,
            model_p1  = p1_model,
            model_p2  = p2_model,
            p1_deck   = args.p1_deck,
            p2_deck   = args.p2_deck,
            encoder   = encoder,
            device    = device,
        )

        # Translate result to A's perspective
        if result_for_p1 == "draw":
            result_a = "draw"
            draws   += 1
        elif (result_for_p1 == "win") == a_is_p1:
            # P1 won and A was P1, OR P1 lost and A was P2 → A wins
            result_a = "win"
            a_wins  += 1
            elo.update(winner=label_a, loser=label_b)
        else:
            result_a = "loss"
            b_wins  += 1
            elo.update(winner=label_b, loser=label_a)

        total   = a_wins + b_wins + draws
        a_pct   = a_wins / total
        a_elo   = elo.rating(label_a)
        b_elo   = elo.rating(label_b)
        p1_tag  = f"A=P1" if a_is_p1 else f"B=P1"

        logger.log_game(global_step=step_a, checkpoint=label_a, stats=stats)
        logger.log_elo(elo)

        print(
            f"  Game {i:>3} ({p1_tag}): A={result_a:4s} | "
            f"A={a_wins} B={b_wins} D={draws}  A_win%={a_pct:.1%} | "
            f"Elo A={a_elo:.0f} B={b_elo:.0f} | "
            f"steps={stats.total_steps}"
            + ("  TRUNC" if stats.truncated else "")
        )

    # Summary
    n = args.n_games
    print(f"\n{'─'*60}")
    print(f"Head-to-head: {label_a} vs {label_b} over {n} games")
    print(f"  A wins:  {a_wins}  ({a_wins/n:.1%})")
    print(f"  B wins:  {b_wins}  ({b_wins/n:.1%})")
    print(f"  Draws:   {draws}  ({draws/n:.1%})")
    print(f"  Final Elo — A: {elo.rating(label_a):.1f}  B: {elo.rating(label_b):.1f}")
    print(f"  Logs: {args.log_dir}/")


if __name__ == "__main__":
    main()
