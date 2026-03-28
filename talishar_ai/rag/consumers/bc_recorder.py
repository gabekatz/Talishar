"""
bc_recorder.py — Records LLM decisions as behavioral cloning training data.

Wraps LLMActionConsumer to capture every decision in DemoDataset-compatible
JSONL format.  Run 50-200 games to generate expert demonstrations, then
train a PPO model via behavioral cloning (train_bc.py).

Output format (one JSON object per line)::

    {
        "obs": [0.5, 0.1, ...],       # float32 observation vector
        "action": 3,                   # action index chosen by Claude
        "legal_mask": [true, true, ...], # bool mask
        "card_ids": [42, 17, ...],     # card vocab indices
        "description": "Play Rising Knee Thrust Blue",
        "game": "game_12345",
        "step": 7,
        "llm_reasoning": "...",
        "llm_confidence": 0.85
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ...features import StateEncoder
from .llm_consumer import LLMActionConsumer


class BCRecorder:
    """
    Wraps LLMActionConsumer to record decisions as BC training data.

    Parameters
    ----------
    llm_consumer:
        The LLM consumer that makes decisions.
    encoder:
        StateEncoder for converting game state to observation vectors.
    output_path:
        Path to the output JSONL file.
    """

    def __init__(
        self,
        llm_consumer: LLMActionConsumer,
        encoder: StateEncoder,
        output_path: str | Path,
    ) -> None:
        self._consumer = llm_consumer
        self._encoder = encoder
        self._output_path = Path(output_path)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer: list[dict[str, Any]] = []
        self._flush_every = 50  # Flush to disk every N records

    def act_and_record(
        self,
        state: dict[str, Any],
        legal_moves: list[dict[str, Any]],
        game_name: str,
        step: int,
    ) -> int:
        """
        Ask LLM for action, record the decision, return action index.

        Parameters
        ----------
        state:
            Raw GetAIState JSON.
        legal_moves:
            Legal moves from state.
        game_name:
            Identifier for the current game.
        step:
            Step number within the game.

        Returns
        -------
        The chosen action index.
        """
        # Get LLM decision
        action_idx, decision = self._consumer.act(state, legal_moves)

        # Encode observation
        obs = self._encoder.encode(state)
        mask = self._encoder.action_mask(state)
        card_ids = self._encoder.card_ids(state)

        # Get action description
        description = ""
        if 0 <= action_idx < len(legal_moves):
            move = legal_moves[action_idx]
            description = move.get("description", "")
            if not description:
                description = f"{move.get('type', '')}: {move.get('cardID', '')}"

        # Build record
        record = {
            "obs": obs.tolist(),
            "action": action_idx,
            "legal_mask": mask.tolist(),
            "card_ids": card_ids.tolist(),
            "description": description,
            "game": game_name,
            "step": step,
            "llm_reasoning": decision.reasoning,
            "llm_confidence": decision.confidence,
        }

        self._buffer.append(record)

        if len(self._buffer) >= self._flush_every:
            self.flush()

        return action_idx

    def flush(self) -> None:
        """Write buffered records to disk."""
        if not self._buffer:
            return
        with open(self._output_path, "a") as f:
            for record in self._buffer:
                f.write(json.dumps(record) + "\n")
        self._buffer.clear()

    def close(self) -> None:
        """Flush and finalize."""
        self.flush()

    @property
    def records_written(self) -> int:
        """Total records flushed + buffered."""
        return self._consumer.decisions_made

    def __enter__(self) -> "BCRecorder":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
