"""
play_llm.py — Play games using Claude as the decision engine.

Creates training games and drives one or both players with the LLM agent
backed by the RAG retrieval layer.

Usage
-----
    # Play 5 games vs EncounterAI
    uv run python -m scripts.play_llm --n-games 5

    # Self-play: LLM controls both sides (better BC data)
    uv run python -m scripts.play_llm --self-play --record --n-games 50 -v

    # Record BC training data vs EncounterAI
    uv run python -m scripts.play_llm --n-games 50 --record --output llm_demos/data.jsonl

    # Custom decks and model
    uv run python -m scripts.play_llm --deck IraScarletRevenger --opponent-deck Dummy --model claude-sonnet-4-20250514
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running as `python scripts/play_llm.py` without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from talishar_ai.game_manager import GameManager
from talishar_ai.features import StateEncoder
from talishar_ai.card_vocab import CardVocab
from talishar_ai.rag.card_index import CardIndex
from talishar_ai.rag.retriever import Retriever
from talishar_ai.rag.llm_agent import LLMAgent
from talishar_ai.rag.consumers.llm_consumer import LLMActionConsumer
from talishar_ai.rag.consumers.bc_recorder import BCRecorder


def _do_turn(
    consumer: LLMActionConsumer,
    state: dict,
    legal_moves: list[dict],
    game_name: str,
    step: int,
    player_id: int,
    bc_recorder: BCRecorder | None,
    verbose: bool,
) -> tuple[int, str]:
    """Pick an action for one player and return (action_idx, move_description)."""
    if bc_recorder:
        action_idx = bc_recorder.act_and_record(
            state, legal_moves, game_name, step
        )
        decision = consumer.last_decision
    else:
        action_idx, decision = consumer.act(state, legal_moves)

    move_desc = ""
    if 0 <= action_idx < len(legal_moves):
        move_desc = legal_moves[action_idx].get("description", "")

    if verbose and decision is not None:
        tag = f"P{player_id}" if player_id else ""
        n_moves = len(legal_moves)
        if n_moves <= 1:
            print(f"    Step {step}: {tag} [{action_idx}] {move_desc} (trivial)")
        else:
            print(
                f"    Step {step}: {tag} [{action_idx}/{n_moves-1}] {move_desc} "
                f"(conf={decision.confidence:.2f})"
            )
            if decision.reasoning:
                print(f"      > {decision.reasoning}")

    return action_idx, move_desc


def play_one_game(
    gm: GameManager,
    consumer: LLMActionConsumer,
    p1_deck: str,
    p2_deck: str,
    max_steps: int = 500,
    verbose: bool = False,
    bc_recorder: BCRecorder | None = None,
    self_play: bool = False,
) -> dict:
    """
    Play one full game with the LLM agent.

    When self_play=True, the LLM controls both P1 and P2. Otherwise P2 is
    the PHP engine's EncounterAI.

    Returns a dict with game results and stats.
    """
    # Create game — both players are external when self-playing
    game_name, p1_key, p2_key = gm.create_game(
        p1_deck=p1_deck,
        p2_deck=p2_deck,
        p2_is_ai=not self_play,
    )

    mode = "self-play" if self_play else "vs EncounterAI"
    if verbose:
        print(f"  Game {game_name} created: {p1_deck} vs {p2_deck} ({mode})")

    players = [
        (1, p1_key),
        (2, p2_key),
    ]

    step = 0
    result = None
    consecutive_passes = 0
    max_consecutive_passes = 20  # Detect stuck games
    no_priority_polls = 0
    max_no_priority_polls = 80  # Safety valve for truly stuck states

    while step < max_steps:
        # In self-play, try both players each iteration to find who has priority.
        # Against EncounterAI, only poll P1.
        players_to_poll = players if self_play else players[:1]

        acted = False
        for player_id, auth_key in players_to_poll:
            state = gm.get_state(game_name, player_id=player_id, auth_key=auth_key)

            # Check terminal
            phase = (state.get("phase") or {}).get("turnPhase", "")
            if phase == "OVER":
                # From P1's perspective
                p1_state = state if player_id == 1 else gm.get_state(
                    game_name, player_id=1, auth_key=p1_key
                )
                my_health = int(p1_state.get("myState", {}).get("health", 0) or 0)
                opp_health = int(p1_state.get("theirState", {}).get("health", 0) or 0)
                result = "win" if my_health > opp_health else "loss"
                if verbose:
                    print(
                        f"  Game over: {result} "
                        f"(P1: {my_health} HP, P2: {opp_health} HP, {step} steps)"
                    )
                break

            if not state.get("havePriority"):
                continue
            legal_moves = state.get("legalMoves", [])
            if not legal_moves:
                continue

            # This player has priority — make a decision
            action_idx, move_desc = _do_turn(
                consumer, state, legal_moves, game_name, step,
                player_id, bc_recorder, verbose,
            )

            # Loop detection
            is_pass = "pass" in move_desc.lower()
            if is_pass:
                consecutive_passes += 1
            else:
                consecutive_passes = 0

            if consecutive_passes >= max_consecutive_passes:
                result = "stuck"
                if verbose:
                    print(
                        f"  Game stuck: {consecutive_passes} consecutive passes "
                        f"at step {step}. Aborting."
                    )
                break

            # Submit action
            params = legal_moves[action_idx].get("params", {})
            gm.submit_action(
                game_name, player_id=player_id, auth_key=auth_key, params=params
            )
            step += 1
            acted = True
            no_priority_polls = 0
            break  # Re-poll from top to find who has priority next

        if result is not None:
            break

        if not acted:
            no_priority_polls += 1
            if no_priority_polls >= max_no_priority_polls:
                result = "stuck"
                if verbose:
                    print(f"  Game stuck: no player got priority for {no_priority_polls} polls")
                break
            time.sleep(0.25)

    if result is None:
        result = "truncated"
        if verbose:
            print(f"  Game truncated after {max_steps} steps")

    return {
        "game_name": game_name,
        "result": result,
        "steps": step,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play FaB games using Claude as the decision engine"
    )
    parser.add_argument("--n-games", type=int, default=5, help="Number of games to play")
    parser.add_argument("--deck", default="IraScarletRevenger", help="P1 deck name")
    parser.add_argument("--opponent-deck", default="IraScarletRevenger", help="P2 deck name")
    parser.add_argument(
        "--model",
        default=None,
        help="Claude model (default: haiku for --record, sonnet otherwise)",
    )
    parser.add_argument("--base-url", default="http://localhost:8080", help="Game server URL")
    parser.add_argument("--index-dir", default=None, help="Card index directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed output")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per game")
    parser.add_argument(
        "--self-play", action="store_true",
        help="LLM plays both sides (better BC data, 2x API cost per decision)",
    )

    # BC recording
    parser.add_argument("--record", action="store_true", help="Record BC training data")
    parser.add_argument(
        "--output",
        default="llm_demos/data.jsonl",
        help="Output path for BC data (with --record)",
    )

    # Distillation logging (for fine-tuning a local model)
    parser.add_argument(
        "--distill-log",
        default=None,
        help="Log prompt/response pairs for local model fine-tuning (JSONL)",
    )

    args = parser.parse_args()

    # Model selection: Haiku for BC recording (cheap), Sonnet for live play (quality)
    _HAIKU = "claude-haiku-4-5-20251001"
    _SONNET = "claude-sonnet-4-20250514"
    if args.model is None:
        model = _HAIKU if args.record else _SONNET
    else:
        model = args.model

    # Build components
    print("[play_llm] Loading card index...")
    index_dir = args.index_dir or str(Path(__file__).parent.parent / "indices")
    card_index = CardIndex(persist_dir=index_dir)
    retriever = Retriever(card_index)
    # Auto-enable distillation logging when recording BC data
    distill_log = args.distill_log
    if distill_log is None and args.record:
        distill_log = str(Path(args.output).parent / "distill.jsonl")

    llm_agent = LLMAgent(retriever, model=model, distill_log=distill_log)
    consumer = LLMActionConsumer(llm_agent)

    if distill_log:
        print(f"[play_llm] Distillation log: {distill_log}")

    gm = GameManager(base_url=args.base_url)

    # Optional BC recorder
    bc_recorder = None
    if args.record:
        vocab = CardVocab.load_or_build()
        encoder = StateEncoder(vocab)
        bc_recorder = BCRecorder(consumer, encoder, args.output)
        print(f"[play_llm] Recording BC data to {args.output}")

    # Play games
    results = {"win": 0, "loss": 0, "truncated": 0, "stuck": 0}
    total_steps = 0
    start_time = time.time()

    mode_str = "self-play" if args.self_play else "vs EncounterAI"
    print(
        f"[play_llm] Playing {args.n_games} games ({mode_str}): "
        f"{args.deck} vs {args.opponent_deck} (model: {model})"
    )

    for game_num in range(1, args.n_games + 1):
        print(f"\nGame {game_num}/{args.n_games}:")
        game_result = play_one_game(
            gm=gm,
            consumer=consumer,
            p1_deck=args.deck,
            p2_deck=args.opponent_deck,
            max_steps=args.max_steps,
            verbose=args.verbose,
            bc_recorder=bc_recorder,
            self_play=args.self_play,
        )
        results[game_result["result"]] += 1
        total_steps += game_result["steps"]

    elapsed = time.time() - start_time

    # Flush BC data
    if bc_recorder:
        bc_recorder.close()
        print(f"\n[play_llm] BC data: {bc_recorder.records_written} records written")

    # Summary
    print(f"\n{'='*50}")
    print(f"Results ({args.n_games} games, {elapsed:.1f}s):")
    print(f"  Wins:      {results['win']}")
    print(f"  Losses:    {results['loss']}")
    print(f"  Truncated: {results['truncated']}")
    print(f"  Stuck:     {results['stuck']}")
    if results["win"] + results["loss"] > 0:
        win_rate = results["win"] / (results["win"] + results["loss"])
        print(f"  Win rate:  {100*win_rate:.0f}%")
    print(f"  Total steps: {total_steps}")
    print(f"  Avg steps/game: {total_steps / args.n_games:.0f}")
    print(f"  LLM decisions: {consumer.decisions_made}")

    # API usage stats
    stats = llm_agent.stats
    print(f"\nAPI Usage:")
    print(f"  API calls:       {stats['api_calls']}")
    print(f"  Trivial skipped: {stats['skipped_trivial']} ({100*stats['trivial_skip_rate']:.0f}%)")
    print(f"  Input tokens:    {stats['input_tokens']:,}")
    print(f"  Output tokens:   {stats['output_tokens']:,}")
    print(f"  Total tokens:    {stats['total_tokens']:,}")
    if stats['input_tokens'] > 0:
        # Cost estimates per model ($/MTok)
        costs = {
            "haiku": (0.80, 4.0),
            "sonnet": (3.0, 15.0),
            "opus": (15.0, 75.0),
        }
        # Detect model tier from name
        tier = "sonnet"
        for t in costs:
            if t in model.lower():
                tier = t
                break
        in_rate, out_rate = costs[tier]
        est_cost = (stats['input_tokens'] * in_rate + stats['output_tokens'] * out_rate) / 1_000_000
        print(f"  Est. cost ({tier}): ${est_cost:.2f}")


if __name__ == "__main__":
    main()
