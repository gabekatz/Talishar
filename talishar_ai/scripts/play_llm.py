"""
play_llm.py — Play games using Claude as the decision engine.

Creates training games against EncounterAI (or a second LLM player)
and drives P1 with the LLM agent backed by the RAG retrieval layer.

Usage
-----
    # Play 5 games and print results
    uv run python -m talishar_ai.scripts.play_llm --n-games 5

    # Record BC training data
    uv run python -m talishar_ai.scripts.play_llm --n-games 50 --record --output llm_demos/data.jsonl

    # Custom decks and model
    uv run python -m talishar_ai.scripts.play_llm --deck IraScarletRevenger --opponent-deck Dummy --model claude-sonnet-4-20250514
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


def play_one_game(
    gm: GameManager,
    consumer: LLMActionConsumer,
    p1_deck: str,
    p2_deck: str,
    max_steps: int = 500,
    verbose: bool = False,
    bc_recorder: BCRecorder | None = None,
) -> dict:
    """
    Play one full game with the LLM agent as P1.

    Returns a dict with game results and stats.
    """
    # Create game
    game_name, p1_key, p2_key = gm.create_game(
        p1_deck=p1_deck,
        p2_deck=p2_deck,
        p2_is_ai=True,
    )

    if verbose:
        print(f"  Game {game_name} created: {p1_deck} vs {p2_deck}")

    step = 0
    result = None

    while step < max_steps:
        # Get state (blocking until we have priority or game is over)
        state = gm.get_state_blocking(game_name, player_id=1, auth_key=p1_key)

        # Check terminal
        phase = (state.get("phase") or {}).get("turnPhase", "")
        if phase == "OVER":
            # Determine winner
            my_health = int(state.get("myState", {}).get("health", 0) or 0)
            opp_health = int(state.get("theirState", {}).get("health", 0) or 0)
            result = "win" if my_health > opp_health else "loss"
            if verbose:
                print(
                    f"  Game over: {result} "
                    f"(P1: {my_health} HP, P2: {opp_health} HP, {step} steps)"
                )
            break

        legal_moves = state.get("legalMoves", [])
        if not legal_moves:
            # No legal moves but not terminal — wait and retry
            time.sleep(0.25)
            continue

        # Choose action — both paths go through consumer.act(), so we
        # always have the LLMDecision available for verbose output.
        if bc_recorder:
            action_idx = bc_recorder.act_and_record(
                state, legal_moves, game_name, step
            )
            # Pull the last decision from the consumer for verbose logging
            decision = consumer.last_decision
        else:
            action_idx, decision = consumer.act(state, legal_moves)

        if verbose and decision is not None:
            move_desc = ""
            if 0 <= action_idx < len(legal_moves):
                move_desc = legal_moves[action_idx].get("description", "")
            n_moves = len(legal_moves)
            if n_moves <= 1:
                print(f"    Step {step}: [{action_idx}] {move_desc} (trivial)")
            else:
                print(
                    f"    Step {step}: [{action_idx}/{n_moves-1}] {move_desc} "
                    f"(conf={decision.confidence:.2f})"
                )
                if decision.reasoning:
                    print(f"      > {decision.reasoning}")

        # Submit action
        params = legal_moves[action_idx].get("params", {})
        gm.submit_action(game_name, player_id=1, auth_key=p1_key, params=params)
        step += 1

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

    # BC recording
    parser.add_argument("--record", action="store_true", help="Record BC training data")
    parser.add_argument(
        "--output",
        default="llm_demos/data.jsonl",
        help="Output path for BC data (with --record)",
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
    llm_agent = LLMAgent(retriever, model=model)
    consumer = LLMActionConsumer(llm_agent)

    gm = GameManager(base_url=args.base_url)

    # Optional BC recorder
    bc_recorder = None
    if args.record:
        vocab = CardVocab.load_or_build()
        encoder = StateEncoder(vocab)
        bc_recorder = BCRecorder(consumer, encoder, args.output)
        print(f"[play_llm] Recording BC data to {args.output}")

    # Play games
    results = {"win": 0, "loss": 0, "truncated": 0}
    total_steps = 0
    start_time = time.time()

    print(
        f"[play_llm] Playing {args.n_games} games: "
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
