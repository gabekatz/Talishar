"""
scripts/play_vs_ai.py — Play against a trained model using the Talishar UI.

Creates a game with both players as "human", prints a URL you can open in the
Talishar-FE browser client, then drives P2 with the trained model.

Usage
-----
# 1. Make sure Docker backend is running (bash start.sh)
# 2. Make sure Talishar-FE is running (npm run dev in ../Talishar-FE)
# 3. Run this script:

python -m talishar_ai.scripts.play_vs_ai \
  --checkpoint checkpoints/model_final.pt \
  --p1-deck IraScarletRevenger \
  --p2-deck IraScarletRevenger

# 4. Open the printed URL in your browser
# 5. Play as P1 — the AI will respond as P2 automatically
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from talishar_ai.game_manager import GameManager
from talishar_ai.features import StateEncoder
from talishar_ai.models.network import ActorCritic
from talishar_ai.models.lstm_network import LSTMActorCritic
from talishar_ai.models.action_network import ActionEmbedActorCritic
from talishar_ai.models.action_lstm_network import LSTMActionEmbedActorCritic


def load_model(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    sd   = ckpt["model_state"]
    has_lstm       = any("lstm" in k for k in sd)
    has_action_emb = "action_encoder.0.weight" in sd
    has_emb        = "embedding.weight" in sd
    emb_dim    = sd["embedding.weight"].shape[1] if has_emb else 32
    vocab_size = sd["embedding.weight"].shape[0] if has_emb else 5000

    # Infer action_feat_dim from checkpoint (may differ from current ACTION_DIM)
    action_feat_dim = int(sd["action_encoder.0.weight"].shape[1]) if has_action_emb else 28

    if has_action_emb and has_lstm:
        model = LSTMActionEmbedActorCritic(
            use_embeddings=has_emb, emb_dim=emb_dim, vocab_size=vocab_size,
            action_feat_dim=action_feat_dim,
        ).to(device)
    elif has_action_emb:
        model = ActionEmbedActorCritic(
            use_embeddings=has_emb, emb_dim=emb_dim, vocab_size=vocab_size,
            action_feat_dim=action_feat_dim,
        ).to(device)
    elif has_lstm:
        model = LSTMActorCritic(
            use_embeddings=has_emb, emb_dim=emb_dim, vocab_size=vocab_size,
        ).to(device)
    else:
        model = ActorCritic(
            use_embeddings=has_emb, emb_dim=emb_dim, vocab_size=vocab_size,
        ).to(device)
    model.load_state_dict(sd)
    model.eval()
    step = ckpt.get("step", 0)
    print(
        f"Loaded {path}  (step {step:,}, lstm={has_lstm}, "
        f"action_embed={has_action_emb}, emb={has_emb})"
    )
    return model, has_lstm


def parse_args():
    p = argparse.ArgumentParser(description="Play against a trained AI in the Talishar UI.")
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    p.add_argument("--base-url",   default="http://localhost:8080", help="Backend URL")
    p.add_argument("--fe-url",     default="http://localhost:5173", help="Frontend URL")
    p.add_argument("--p1-deck",    default="IraScarletRevenger")
    p.add_argument("--p2-deck",    default="IraScarletRevenger")
    p.add_argument("--device",     default="cpu")
    p.add_argument("--temperature", type=float, default=0.5,
                   help="Sampling temperature (0 = greedy, higher = more random)")
    return p.parse_args()


def main():
    args    = parse_args()
    device  = torch.device(args.device)
    gm      = GameManager(base_url=args.base_url)
    encoder = StateEncoder()

    model, is_lstm = load_model(args.checkpoint, device)
    use_emb        = getattr(model, "use_embeddings", False)
    use_action_emb = getattr(model, "use_action_embed", False)

    # Create game — both players as "human" so the frontend can connect
    game_name, p1_key, p2_key = gm.create_game(
        p1_deck=args.p1_deck, p2_deck=args.p2_deck, p2_is_ai=False,
    )

    # Print the URL for the player
    url = f"{args.fe_url}/game/play?gameName={game_name}&playerID=1&authKey={p1_key}"
    print()
    print("=" * 60)
    print("  Open this URL in your browser to play:")
    print()
    print(f"  {url}")
    print()
    print(f"  Game: {game_name}  |  You: P1  |  AI: P2")
    print(f"  Temperature: {args.temperature}")
    print("=" * 60)
    print()
    print("Waiting for you to play... (Ctrl+C to quit)")
    print()

    # Init LSTM hidden state
    hh = hc = None
    if is_lstm:
        hh, hc = model.init_hidden(1, device)

    poll_interval = 0.3
    idle_count    = 0

    while True:
        try:
            # Check P2's state
            p2_state = gm.get_state(game_name, 2, p2_key)

            # Game over?
            phase = (p2_state.get("phase") or {}).get("turnPhase", "")
            if phase == "OVER":
                p2_hp = (p2_state.get("myState") or {}).get("health", 0)
                p1_hp = (p2_state.get("theirState") or {}).get("health", 0)
                if p2_hp <= 0:
                    print("Game over — You win!")
                elif p1_hp <= 0:
                    print("Game over — AI wins!")
                else:
                    print("Game over — Draw!")
                break

            # P2 has priority?
            if not p2_state.get("havePriority") or not p2_state.get("legalMoves"):
                idle_count += 1
                time.sleep(poll_interval)
                continue

            idle_count = 0
            moves = p2_state["legalMoves"]

            # Encode state
            obs  = encoder.encode(p2_state)
            mask = encoder.action_mask(p2_state)
            obs_t  = torch.from_numpy(obs).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)

            ids_t = None
            if use_emb:
                ids = encoder.card_ids(p2_state)
                ids_t = torch.from_numpy(ids).unsqueeze(0).to(torch.int32).to(device)

            afeats_t = None
            if use_action_emb:
                afeats = encoder.encode_actions(p2_state)
                # Truncate to model's expected dim (old checkpoints may use fewer features)
                model_adim = model.action_encoder[0].in_features
                afeats_t = torch.from_numpy(afeats[:, :model_adim]).unsqueeze(0).to(device)

            # Model inference
            with torch.no_grad():
                if use_action_emb and is_lstm:
                    logits, _, hh, hc = model(obs_t, mask_t, hh, hc, afeats_t, ids_t)
                elif use_action_emb:
                    logits, _ = model(obs_t, mask_t, afeats_t, ids_t)
                elif is_lstm:
                    logits, _, hh, hc = model(obs_t, mask_t, hh, hc, ids_t)
                else:
                    logits, _ = model(obs_t, mask_t, ids_t)

                if args.temperature <= 0:
                    action = int(logits.argmax(dim=-1).item())
                else:
                    dist = torch.distributions.Categorical(logits=logits / args.temperature)
                    action = int(dist.sample().item())

            # Clamp to valid range
            if action >= len(moves):
                action = 0

            move = moves[action]
            desc = move.get("description", f"mode={move.get('mode')}")
            print(f"  AI plays: {desc}")

            gm.submit_action(game_name, 2, p2_key, move["params"])

        except KeyboardInterrupt:
            print("\nQuitting.")
            break
        except Exception as exc:
            print(f"  [error] {exc}")
            time.sleep(1)


if __name__ == "__main__":
    main()
