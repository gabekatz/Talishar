"""
demo_server.py — Browser-based demo collection server.

A local web server that lets a human play as P1 against an AI opponent (P2),
recording every human decision as a training example for behavioral cloning.

Usage
-----
# Human vs random P2, save demos to demos/session1.jsonl:
uv run python -m scripts.demo_server --out demos/session1.jsonl

# Human vs a trained checkpoint:
uv run python -m scripts.demo_server --out demos/session2.jsonl --p2-checkpoint checkpoints/bc_final.pt

# Then open http://localhost:5000 in your browser.

Requirements
------------
pip install flask         (already in pyproject.toml extras)

How it works
------------
1. You open http://localhost:5000 — a new game is created automatically.
2. The page shows your health, hand, and the legal moves available.
3. You click a move button.
4. The server records your choice (obs, action index, legal mask, card ids)
   and drives P2 automatically until it's your turn again.
5. The page reloads with the new game state.
6. When the game ends, you see the result and can start a new one.
7. All your choices are appended to the --out file in real time.

The P2 AI acts greedily (argmax) if a checkpoint is provided, otherwise randomly.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

try:
    from flask import Flask, redirect, request, url_for
except ImportError:
    print("Flask is required: uv add flask  or  pip install flask")
    sys.exit(1)

from talishar_ai.game_manager import GameManager
from talishar_ai.env import TalisharEnv
from talishar_ai.features import StateEncoder
from talishar_ai.card_vocab import CardVocab
from talishar_ai.data.demo_dataset import DemoDataset

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Global server state (single game at a time — this is a local dev tool)
# ---------------------------------------------------------------------------

_gm:       GameManager | None  = None
_encoder:  StateEncoder | None = None
_p2_model: Any | None          = None      # ActorCritic / LSTMActorCritic or None
_p2_device: torch.device        = torch.device("cpu")
_p2_is_lstm: bool               = False
_p2_hidden_h: torch.Tensor | None = None
_p2_hidden_c: torch.Tensor | None = None

_dataset:    DemoDataset | None = None
_demo_path:  Path | None        = None
_args:       argparse.Namespace | None = None

# Active game
_game_name: str  = ""
_p1_key:    str  = ""
_p2_key:    str  = ""
_step:      int  = 0
_last_state: dict = {}
_message:   str  = ""   # status message shown to human

# ---------------------------------------------------------------------------
# P2 driving helpers
# ---------------------------------------------------------------------------

def _p2_action(state: dict) -> int:
    """Pick P2's action: greedy model if available, else random."""
    legal = state.get("legalMoves", [])
    if not legal:
        return 0
    if _p2_model is None:
        return int(np.random.randint(len(legal)))

    global _p2_hidden_h, _p2_hidden_c
    obs   = _encoder.encode(state)
    mask  = _encoder.action_mask(state)
    ids   = _encoder.card_ids(state)
    obs_t  = torch.from_numpy(obs).unsqueeze(0).to(_p2_device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(_p2_device)
    ids_t  = torch.from_numpy(ids).unsqueeze(0).to(_p2_device)

    with torch.no_grad():
        use_emb = _p2_model.use_embeddings
        if _p2_is_lstm:
            logits, _, _p2_hidden_h, _p2_hidden_c = _p2_model(
                obs_t, mask_t, _p2_hidden_h, _p2_hidden_c,
                ids_t if use_emb else None,
            )
        else:
            logits, _ = _p2_model(obs_t, mask_t, ids_t if use_emb else None)
    return int(logits.argmax(dim=-1).item())


def _drive_p2(state_after_p1: dict) -> dict:
    """Drive P2 until P1 has priority or game ends.  Returns P1's new state."""
    for _ in range(600):
        if TalisharEnv._is_terminal(state_after_p1):
            return _gm.get_state_blocking(_game_name, 1, _p1_key)

        p2_state = _gm.get_state(_game_name, 2, _p2_key)
        if TalisharEnv._is_terminal(p2_state):
            return _gm.get_state_blocking(_game_name, 1, _p1_key)

        if p2_state.get("havePriority") and p2_state.get("legalMoves"):
            # Filter undo/cancel modes — prefer non-undo moves when available,
            # fall back to undo/cancel only when it's the sole option.
            _UNDO_MODES = {10000, 10001, 10003}
            all_moves   = p2_state["legalMoves"]
            non_undo    = [m for m in all_moves if m.get("mode") not in _UNDO_MODES]
            valid_moves = non_undo if non_undo else all_moves
            p2_state_filtered = {**p2_state, "legalMoves": valid_moves}
            action = _p2_action(p2_state_filtered)
            move   = valid_moves[action]
            state_after_p1 = _gm.submit_action(_game_name, 2, _p2_key, move["params"])
            continue

        p1_state = _gm.get_state(_game_name, 1, _p1_key)
        if TalisharEnv._is_terminal(p1_state):
            return p1_state
        if p1_state.get("havePriority") and p1_state.get("legalMoves"):
            return p1_state

        time.sleep(0.1)

    # Timed out — return whatever state we have
    return _gm.get_state_blocking(_game_name, 1, _p1_key)


def _reset_p2_hidden() -> None:
    global _p2_hidden_h, _p2_hidden_c
    if _p2_is_lstm and _p2_model is not None:
        _p2_hidden_h, _p2_hidden_c = _p2_model.init_hidden(1, _p2_device)


def _new_game() -> None:
    """Create a new game and set global state."""
    global _game_name, _p1_key, _p2_key, _step, _last_state, _message
    name, p1k, p2k = _gm.create_game(
        p1_deck=_args.p1_deck,
        p2_deck=_args.p2_deck,
        p2_is_ai=False,
    )
    _game_name = name
    _p1_key    = p1k
    _p2_key    = p2k
    _step      = 0
    _reset_p2_hidden()
    _last_state = _gm.get_state_blocking(name, 1, p1k)
    _message    = f"New game #{name} started. You are P1."

# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

_STYLE = """
<style>
  body { font-family: monospace; background: #0d1117; color: #c9d1d9;
         padding: 20px; max-width: 960px; margin: auto; }
  h1   { color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 8px; }
  h2   { color: #79c0ff; margin-top: 20px; }
  .msg { background: #161b22; border: 1px solid #30363d; padding: 10px;
         border-radius: 6px; margin-bottom: 15px; color: #8b949e; }
  .hbar { display: inline-block; padding: 4px 12px; border-radius: 4px;
          font-size: 1.4em; font-weight: bold; }
  .p1hp { background: #1f4e2a; color: #56d364; }
  .p2hp { background: #4e1f1f; color: #f85149; }
  .phase { color: #d29922; font-size: 1.1em; }
  table { border-collapse: collapse; width: 100%; }
  td, th { border: 1px solid #30363d; padding: 6px 10px; text-align: left; }
  th { background: #161b22; color: #8b949e; }
  tr:hover td { background: #161b22; }
  .move-btn { display: block; background: #1c2128; border: 1px solid #388bfd;
              color: #c9d1d9; padding: 8px 14px; margin: 4px 0;
              cursor: pointer; border-radius: 6px; text-align: left;
              width: 100%; font-family: monospace; font-size: 0.95em; }
  .move-btn:hover { background: #388bfd; color: #0d1117; }
  .win  { color: #56d364; font-size: 1.3em; font-weight: bold; }
  .loss { color: #f85149; font-size: 1.3em; font-weight: bold; }
  .new-btn { background: #238636; border: none; color: white; padding: 10px 20px;
             border-radius: 6px; cursor: pointer; font-size: 1em; margin-top: 10px; }
  .new-btn:hover { background: #2ea043; }
  .card-tag { background: #1c2128; border: 1px solid #30363d; padding: 2px 8px;
              border-radius: 4px; display: inline-block; margin: 2px; font-size:0.9em;}
  .combat { background: #2d1b00; border: 1px solid #d29922;
            padding: 10px; border-radius: 6px; margin: 10px 0; }
</style>
"""


def _render_hand(hand: list) -> str:
    if not hand:
        return "<em style='color:#8b949e'>empty</em>"
    cards = []
    for c in hand:
        name  = c.get("cardNumber", "?")
        cost  = c.get("cost", "?")
        power = c.get("power", "")
        pitch = c.get("pitch", "")
        tip   = f"cost:{cost}"
        if power:
            tip += f" power:{power}"
        if pitch:
            tip += f" pitch:{pitch}"
        cards.append(f"<span class='card-tag' title='{tip}'>{name}</span>")
    return " ".join(cards)


def _render_state_page(state: dict, message: str = "", terminal: bool = False) -> str:
    my    = state.get("myState", {})
    their = state.get("theirState", {})
    phase = (state.get("phase") or {}).get("caption", "—")
    turn  = state.get("turnNumber", "?")
    legal = state.get("legalMoves", [])

    hand     = my.get("hand", [])
    arsenal  = my.get("arsenal", [])
    equip    = my.get("equipment", [])
    cc       = state.get("combatChain")
    stack    = state.get("stack")
    pending  = state.get("pendingDecision")
    my_hp    = my.get("health", "?")
    opp_hp   = their.get("health", "?")
    deck_my  = my.get("deckCount", "?")
    deck_opp = their.get("deckCount", "?")

    lines = [f"<!DOCTYPE html><html><head><title>Talishar Demo</title>{_STYLE}</head><body>"]
    lines.append(f"<h1>Talishar Demo Collector — Game #{_game_name} · Step {_step}</h1>")

    if message:
        lines.append(f"<div class='msg'>{message}</div>")

    # Health row
    lines.append(
        f"<div style='margin:12px 0'>"
        f"<span class='hbar p1hp'>♥ You: {my_hp}</span> &nbsp;&nbsp; "
        f"<span class='hbar p2hp'>♥ Opp: {opp_hp}</span> &nbsp;&nbsp; "
        f"<span class='phase'>Turn {turn} — {phase}</span>"
        f"</div>"
    )
    lines.append(f"<p style='color:#8b949e'>Deck: {deck_my} cards &nbsp;|&nbsp; Opp deck: {deck_opp} cards</p>")

    # Combat chain
    if cc:
        atk  = cc.get("attackingCard", {})
        pwr  = cc.get("totalPower", "?")
        dom  = "Dominate" if cc.get("dominate") else ""
        ga   = "Go Again" if cc.get("goAgain") else ""
        tags = " ".join(t for t in [dom, ga] if t)
        lines.append(
            f"<div class='combat'>⚔️ <b>Combat chain</b>: "
            f"{atk.get('cardNumber','?')} · Power {pwr}"
            + (f" · {tags}" if tags else "") +
            f"</div>"
        )

    # Hand / arsenal / equipment
    lines.append(f"<h2>Your hand ({len(hand)} cards)</h2><p>{_render_hand(hand)}</p>")
    if arsenal:
        lines.append(f"<p><b>Arsenal:</b> {_render_hand(arsenal)}</p>")
    if equip:
        lines.append(f"<p><b>Equipment:</b> {_render_hand(equip)}</p>")

    # Pending decision context
    if pending:
        ctx = pending.get("context", "")
        lines.append(f"<p class='phase'>📋 {ctx}</p>")

    # Terminal state
    if terminal:
        result = _determine_result(state)
        cls    = "win" if result == "win" else "loss"
        label  = "🏆 You win!" if result == "win" else "💀 You lose"
        lines.append(f"<p class='{cls}'>{label}</p>")
        lines.append(
            f"<form method='POST' action='/new_game'>"
            f"<button class='new-btn' type='submit'>Start new game</button>"
            f"</form>"
        )
        # Save demos to disk
        if _dataset and _demo_path:
            _dataset.save(_demo_path, append=True)
            _dataset._records.clear()
        lines.append("</body></html>")
        return "\n".join(lines)

    # Legal moves
    lines.append(f"<h2>Legal moves ({len(legal)})</h2>")
    if not legal:
        lines.append("<p style='color:#8b949e'>Waiting for your turn…</p>")
    else:
        for idx, move in enumerate(legal):
            desc = move.get("description", f"Move {idx}")
            lines.append(
                f"<form method='POST' action='/move/{idx}' style='margin:0'>"
                f"<button class='move-btn' type='submit'>"
                f"<b>[{idx}]</b> {desc}"
                f"</button>"
                f"</form>"
            )

    lines.append(f"<br><form method='POST' action='/new_game'>")
    lines.append(f"<button class='new-btn' type='submit'>Abandon &amp; start new game</button>")
    lines.append(f"</form>")
    lines.append("</body></html>")
    return "\n".join(lines)


def _determine_result(state: dict) -> str:
    my_hp  = int(state.get("myState",    {}).get("health", 0))
    opp_hp = int(state.get("theirState", {}).get("health", 0))
    if opp_hp <= 0 and my_hp > 0:
        return "win"
    if my_hp <= 0 and opp_hp > 0:
        return "loss"
    return "draw"

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    if not _game_name:
        _new_game()
    terminal = TalisharEnv._is_terminal(_last_state)
    return _render_state_page(_last_state, _message, terminal=terminal)


@app.route("/new_game", methods=["POST"])
def new_game():
    global _message
    _new_game()
    _message = f"Game #{_game_name} started."
    return redirect(url_for("index"))


@app.route("/move/<int:idx>", methods=["POST"])
def submit_move(idx: int):
    global _last_state, _step, _message

    legal = _last_state.get("legalMoves", [])
    if idx >= len(legal):
        _message = f"Invalid move index {idx} (only {len(legal)} legal moves)."
        return redirect(url_for("index"))

    # --- Record demo before submitting ---
    obs      = _encoder.encode(_last_state)
    mask     = _encoder.action_mask(_last_state)
    card_ids = _encoder.card_ids(_last_state)
    desc     = legal[idx].get("description", f"move {idx}")
    if _dataset is not None:
        _dataset.add(obs, idx, mask, card_ids, desc, _game_name, _step)

    # --- Submit P1's action ---
    params     = legal[idx]["params"]
    next_state = _gm.submit_action(_game_name, 1, _p1_key, params)
    _step     += 1

    # --- Drive P2 ---
    try:
        resolved = _drive_p2(next_state)
    except RuntimeError:
        resolved = _last_state   # fallback: show current state
        _message = "Game engine timed out — consider starting a new game."
        _last_state = resolved
        return redirect(url_for("index"))

    _last_state = resolved
    terminal    = TalisharEnv._is_terminal(resolved)

    if terminal:
        result   = _determine_result(resolved)
        _message = f"Game over — {'you win! 🏆' if result == 'win' else 'you lose. 💀'}"
        # Flush demos to disk
        if _dataset is not None and _demo_path is not None:
            _dataset.save(_demo_path, append=True)
            _dataset._records.clear()
    else:
        _message = f"Step {_step} — your turn."

    return redirect(url_for("index"))

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Talishar demo collection server.")
    p.add_argument("--base-url",       default="http://localhost:8080")
    p.add_argument("--p1-deck",        default="Ira")
    p.add_argument("--p2-deck",        default="Ira")
    p.add_argument("--out",            default="demos/session.jsonl",
                   help="Output .jsonl file for demo records.")
    p.add_argument("--p2-checkpoint",  default=None,
                   help="Checkpoint .pt to load for P2 AI (omit for random P2).")
    p.add_argument("--device",         default="cpu")
    p.add_argument("--port",           type=int, default=5000)
    return p.parse_args()


def main() -> None:
    global _gm, _encoder, _p2_model, _p2_device, _p2_is_lstm
    global _dataset, _demo_path, _args

    _args      = parse_args()
    _gm        = GameManager(base_url=_args.base_url)
    _encoder   = StateEncoder()
    _demo_path = Path(_args.out)
    _dataset   = DemoDataset()
    _p2_device = torch.device(_args.device)

    if _args.p2_checkpoint:
        ckpt = torch.load(_args.p2_checkpoint, map_location=_p2_device)
        # Detect model type and architecture from state dict keys
        from talishar_ai.models.network import ActorCritic
        from talishar_ai.models.lstm_network import LSTMActorCritic
        sd = ckpt["model_state"]
        has_lstm   = any("lstm" in k for k in sd)
        has_emb    = "embedding.weight" in sd
        emb_dim    = sd["embedding.weight"].shape[1] if has_emb else 32
        vocab_size = sd["embedding.weight"].shape[0] if has_emb else 5000
        if has_lstm:
            _p2_model   = LSTMActorCritic(
                use_embeddings=has_emb, emb_dim=emb_dim, vocab_size=vocab_size,
            ).to(_p2_device)
            _p2_is_lstm = True
        else:
            _p2_model = ActorCritic(
                use_embeddings=has_emb, emb_dim=emb_dim, vocab_size=vocab_size,
            ).to(_p2_device)
        _p2_model.load_state_dict(sd)
        _p2_model.eval()
        print(f"[demo_server] Loaded P2 checkpoint: {_args.p2_checkpoint} "
              f"(lstm={has_lstm}, embeddings={has_emb})")
    else:
        print("[demo_server] No P2 checkpoint — P2 will play randomly.")

    _new_game()

    print(f"[demo_server] Listening on http://localhost:{_args.port}")
    print(f"[demo_server] Demos will be saved to {_demo_path}")
    app.run(host="0.0.0.0", port=_args.port, debug=False)


if __name__ == "__main__":
    main()
