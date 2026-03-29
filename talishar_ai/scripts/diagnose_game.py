"""
scripts/diagnose_game.py — Play one game with a trained model and dump
every decision for analysis.

Outputs a detailed log showing the full game state context at each step:
hand cards, equipment, combat chain, legal moves, and which action the
model chose (with probabilities).  Designed for a human or LLM to read
and identify strategic mistakes.

Usage
-----
    uv run python -m scripts.diagnose_game \
      --checkpoint checkpoints/bc_final.pt \
      --p1-deck IraScarletRevenger --p2-deck IraScarletRevenger \
      --output game_diagnosis.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from talishar_ai.game_manager import GameManager
from talishar_ai.features import StateEncoder

# Reuse the model loader from play_vs_ai
from talishar_ai.scripts.play_vs_ai import load_model


def format_card(card: dict) -> str:
    """Format a single card dict for display."""
    cid = card.get("cardID", card.get("id", "?"))
    name = card.get("name", cid)
    cost = card.get("cost", "?")
    power = card.get("power", "-")
    defense = card.get("defense", "-")
    pitch = card.get("pitch", "-")
    parts = [f"{name} ({cid})"]
    if power not in (None, "-", "", "0"):
        parts.append(f"pow={power}")
    if defense not in (None, "-", "", "0"):
        parts.append(f"def={defense}")
    if cost not in (None, "?", ""):
        parts.append(f"cost={cost}")
    if pitch not in (None, "-", "", "0"):
        parts.append(f"pitch={pitch}")
    return " ".join(parts)


def format_state(state: dict, step: int) -> str:
    """Format the full game state as readable text."""
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"STEP {step}")
    lines.append(f"{'='*70}")

    phase_info = state.get("phase") or {}
    turn = phase_info.get("turnNumber", "?")
    turn_phase = phase_info.get("turnPhase", "?")
    lines.append(f"Turn: {turn}  Phase: {turn_phase}")

    my = state.get("myState") or {}
    opp = state.get("theirState") or {}
    lines.append(f"MY HP: {my.get('health', '?')}  |  OPP HP: {opp.get('health', '?')}")
    lines.append(f"MY Resources: {my.get('resources', 0)}  AP: {my.get('ap', 0)}")
    lines.append(f"OPP Hand: {opp.get('handCount', '?')} cards  |  OPP Deck: {opp.get('deckCount', '?')}")
    lines.append("")

    # Hand
    hand = my.get("hand") or []
    if hand:
        lines.append(f"MY HAND ({len(hand)} cards):")
        for i, card in enumerate(hand):
            lines.append(f"  {i+1}. {format_card(card)}")
    else:
        lines.append("MY HAND: empty")
    lines.append("")

    # Equipment
    equip = my.get("equipment") or my.get("equip") or []
    if equip:
        lines.append(f"MY EQUIPMENT ({len(equip)}):")
        for card in equip:
            lines.append(f"  - {format_card(card)}")
    lines.append("")

    # Arsenal
    arsenal = my.get("arsenal") or []
    if arsenal:
        lines.append(f"MY ARSENAL ({len(arsenal)}):")
        for card in arsenal:
            lines.append(f"  - {format_card(card)}")
        lines.append("")

    # Combat chain
    cc = state.get("combatChain") or {}
    if cc:
        atk = cc.get("totalPower", 0) or 0
        blk = cc.get("totalDefense", 0) or 0
        atk_card = cc.get("attackingCard", "")
        go_again = cc.get("goAgain", False)
        dominate = cc.get("dominate", False)
        on_hits = cc.get("activeOnHits", False)
        if atk or blk or atk_card:
            kw = []
            if go_again: kw.append("go-again")
            if dominate: kw.append("dominate")
            if on_hits: kw.append("ON-HIT")
            kw_str = f" [{', '.join(kw)}]" if kw else ""
            lines.append(f"COMBAT CHAIN: power={atk} blocked={blk} card={atk_card}{kw_str}")
            unblocked = max(0, atk - blk)
            if unblocked > 0:
                lines.append(f"  UNBLOCKED DAMAGE: {unblocked}")
            lines.append("")

    # Opponent equipment
    opp_equip = opp.get("equipment") or opp.get("equip") or []
    if opp_equip:
        lines.append(f"OPP EQUIPMENT ({len(opp_equip)}):")
        for card in opp_equip:
            lines.append(f"  - {format_card(card)}")
        lines.append("")

    return "\n".join(lines)


def format_decision(
    moves: list[dict],
    action: int,
    probs: np.ndarray,
    is_lstm: bool,
) -> str:
    """Format the model's decision with probabilities."""
    lines = []
    lines.append("LEGAL MOVES:")
    for i, move in enumerate(moves):
        desc = move.get("description", f"mode={move.get('params', {}).get('mode', '?')}")
        prob = probs[i] if i < len(probs) else 0.0
        marker = " <<<" if i == action else ""
        lines.append(f"  [{i}] {desc}  (prob={prob:.3f}){marker}")

    chosen_desc = ""
    if 0 <= action < len(moves):
        chosen_desc = moves[action].get("description", "?")
    lines.append(f"\nCHOSEN: [{action}] {chosen_desc} (prob={probs[action]:.3f})")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Diagnose a trained model's game decisions")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--p1-deck", default="IraScarletRevenger")
    parser.add_argument("--p2-deck", default="IraScarletRevenger")
    parser.add_argument("--output", default=None, help="Output file (default: stdout)")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = greedy (see what model truly prefers)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    gm = GameManager(base_url=args.base_url)
    encoder = StateEncoder()
    model, is_lstm = load_model(args.checkpoint, device)
    use_emb = getattr(model, "use_embeddings", False)
    use_action_emb = getattr(model, "use_action_embed", False)

    # Create game vs EncounterAI
    game_name, p1_key, p2_key = gm.create_game(
        p1_deck=args.p1_deck, p2_deck=args.p2_deck, p2_is_ai=True,
    )

    out = open(args.output, "w") if args.output else sys.stdout

    def log(s: str):
        print(s, file=out)
        if out is not sys.stdout:
            print(s)  # Also print to terminal

    log(f"GAME DIAGNOSIS: {game_name}")
    log(f"Model: {args.checkpoint}")
    log(f"Decks: {args.p1_deck} vs {args.p2_deck}")
    log(f"Temperature: {args.temperature}")
    log(f"Device: {device}")
    log("")

    hh = hc = None
    if is_lstm:
        hh, hc = model.init_hidden(1, device)

    step = 0
    while step < args.max_steps:
        state = gm.get_state_blocking(game_name, player_id=1, auth_key=p1_key)

        phase = (state.get("phase") or {}).get("turnPhase", "")
        if phase == "OVER":
            my_hp = int((state.get("myState") or {}).get("health", 0) or 0)
            opp_hp = int((state.get("theirState") or {}).get("health", 0) or 0)
            result = "WIN" if my_hp > opp_hp else "LOSS"
            log(f"\n{'='*70}")
            log(f"GAME OVER: {result}  (P1: {my_hp} HP, P2: {opp_hp} HP, {step} steps)")
            break

        moves = state.get("legalMoves", [])
        if not moves:
            time.sleep(0.25)
            continue

        # Log game state
        log(format_state(state, step))

        # Encode
        obs = encoder.encode(state)
        mask = encoder.action_mask(state)
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)

        ids_t = None
        if use_emb:
            ids = encoder.card_ids(state)
            ids_t = torch.from_numpy(ids).unsqueeze(0).to(torch.int32).to(device)

        afeats_t = None
        if use_action_emb:
            afeats = encoder.encode_actions(state)
            model_adim = model.action_encoder[0].in_features
            afeats_t = torch.from_numpy(afeats[:, :model_adim]).unsqueeze(0).to(device)

        # Model inference
        with torch.no_grad():
            if use_action_emb and is_lstm:
                logits, val, hh, hc = model(obs_t, mask_t, hh, hc, afeats_t, ids_t)
            elif use_action_emb:
                logits, val = model(obs_t, mask_t, afeats_t, ids_t)
            elif is_lstm:
                logits, val, hh, hc = model(obs_t, mask_t, hh, hc, ids_t)
            else:
                logits, val = model(obs_t, mask_t, ids_t)

            # Get probabilities
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            value = val.item()

            if args.temperature <= 0:
                action = int(logits.argmax(dim=-1).item())
            else:
                dist = torch.distributions.Categorical(logits=logits / args.temperature)
                action = int(dist.sample().item())

        if action >= len(moves):
            action = 0

        # Log decision
        log(format_decision(moves, action, probs, is_lstm))
        log(f"VALUE ESTIMATE: {value:.4f}")
        log("")

        # Submit
        gm.submit_action(game_name, 1, p1_key, moves[action]["params"])
        step += 1

    if args.output and out is not sys.stdout:
        out.close()
        log(f"\nFull diagnosis written to {args.output}")


if __name__ == "__main__":
    main()
