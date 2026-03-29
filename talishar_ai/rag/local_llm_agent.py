"""
local_llm_agent.py — Local MLX model agent for game decisions.

Drop-in replacement for LLMAgent that uses a locally fine-tuned model
(via MLX LoRA) instead of the Claude API.  Zero marginal cost per game.

Usage
-----
    from rag.local_llm_agent import LocalLLMAgent
    agent = LocalLLMAgent(retriever, model_path="distill_model/fused")
    decision = agent.decide(state, legal_moves)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .llm_agent import LLMDecision, _SYSTEM_PROMPT
from .retriever import Retriever, RetrievalContext


# ---------------------------------------------------------------------------
# Compact system prompt for local model
# The model already learned the full rules during fine-tuning, so we only
# need a short reminder of the output format and key principles.
# This cuts ~800 tokens from every prompt → faster prefill.
# ---------------------------------------------------------------------------

_LOCAL_SYSTEM_PROMPT = """\
You are an expert Flesh and Blood player. Read the STATE DASHBOARD carefully \
before reasoning. Your response MUST be valid JSON with these fields:
{"action_index": <int>, "reasoning": "<2-3 sentences>", "confidence": <0-1>, "phase_strategy": "<1 sentence>"}

Key reminders:
- ACTION POINTS and RESOURCES in the dashboard are the ACTUAL current values. Trust them.
- Pitch cards from hand to generate resources. The engine handles pitching automatically.
- Equipment is a long-term resource. Almost never block with equipment.
- Turn 0: hand cards are FREE to block (redrawn). NEVER break equipment on turn 0.
- Go-again chains multiply value. Always prefer go-again attacks before non-go-again closers.
- Going wide (many attacks) beats going tall (few big attacks) when opponent has few cards.
- Block the minimum needed to prevent on-hit effects. Check ON-HIT EFFECT in the dashboard.
- Do NOT pass when you have playable cards and action points.
"""


class LocalLLMAgent:
    """
    Local MLX model agent — same interface as LLMAgent.

    Loads a fine-tuned MLX model (fused or base+adapter) and uses it
    to make game decisions.  Produces LLMDecision objects identical
    to what the Claude-backed LLMAgent returns.

    Parameters
    ----------
    retriever:
        Built Retriever instance with card indices.
    model_path:
        Path to fused model directory, or HuggingFace model name.
    adapter_path:
        Optional path to LoRA adapter directory (if not using fused model).
    max_tokens:
        Maximum generation tokens.
    distill_log:
        Optional path to log decisions for further distillation.
    """

    def __init__(
        self,
        retriever: Retriever,
        model_path: str = "distill_model/fused",
        adapter_path: str | None = None,
        max_tokens: int = 256,
        distill_log: str | Path | None = None,
    ) -> None:
        self._retriever = retriever
        self._max_tokens = max_tokens

        # Lazy-load MLX model
        try:
            from mlx_lm import load
        except ImportError:
            raise ImportError(
                "mlx-lm not installed. Run: pip install mlx-lm"
            )

        print(f"[LocalLLM] Loading model: {model_path}")
        if adapter_path:
            print(f"[LocalLLM] With adapter: {adapter_path}")
        load_start = time.time()

        self._model, self._tokenizer = load(
            model_path,
            adapter_path=adapter_path,
        )
        print(f"[LocalLLM] Model loaded in {time.time() - load_start:.1f}s")

        # Build a greedy sampler (low temperature for deterministic play)
        import mlx.core as mx
        self._sampler = lambda logits: mx.argmax(logits / 0.1, axis=-1)

        # Pre-tokenize the system prompt prefix for KV cache reuse.
        # Every call shares the same system prompt, so we cache its KV
        # state and only process the (shorter) user prompt each time.
        self._system_prefix = self._tokenizer.apply_chat_template(
            [{"role": "system", "content": _LOCAL_SYSTEM_PROMPT}],
            tokenize=False,
            add_generation_prompt=False,
        )

        # Distillation logging
        self._distill_path: Path | None = None
        if distill_log:
            self._distill_path = Path(distill_log)
            self._distill_path.parent.mkdir(parents=True, exist_ok=True)

        # Usage tracking
        self._total_decisions = 0
        self._api_calls = 0  # Named for compat; means "model calls"
        self._skipped_trivial = 0
        self._total_gen_time = 0.0
        self._total_gen_tokens = 0

    def decide(
        self,
        state: dict[str, Any],
        legal_moves: list[dict[str, Any]],
    ) -> LLMDecision:
        """
        Given raw game state and legal moves, ask the local model for an action.

        Same interface as LLMAgent.decide().
        """
        self._total_decisions += 1

        # -- Trivial decision: 0 or 1 legal moves --
        if len(legal_moves) <= 1:
            self._skipped_trivial += 1
            return LLMDecision(
                action_index=0,
                reasoning="Only one legal move available.",
                confidence=1.0,
                phase_strategy="forced",
            )

        # -- Non-trivial: run local model --
        self._api_calls += 1

        # 1. Retrieve context
        ctx = self._retriever.retrieve(state)

        # 2. Build prompt
        user_prompt = Retriever.render_for_llm(ctx, legal_moves)

        # 3. Format as chat messages (compact local prompt)
        messages = [
            {"role": "system", "content": _LOCAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 4. Generate
        from mlx_lm import generate

        gen_start = time.time()
        response_text = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self._max_tokens,
            sampler=self._sampler,
        )
        gen_time = time.time() - gen_start
        self._total_gen_time += gen_time
        self._total_gen_tokens += len(response_text.split())

        # 5. Log for distillation
        if self._distill_path is not None:
            self._log_distill(user_prompt, response_text)

        # 6. Parse response
        return self._parse_response(response_text, len(legal_moves))

    def _parse_response(
        self,
        response_text: str,
        n_legal_moves: int,
    ) -> LLMDecision:
        """Parse JSON response from local model into LLMDecision."""
        # Try to extract JSON from the response
        json_str = self._extract_json(response_text)

        if json_str:
            try:
                parsed = json.loads(json_str)
                action_idx = int(parsed.get("action_index", 0))
                action_idx = max(0, min(action_idx, n_legal_moves - 1))

                return LLMDecision(
                    action_index=action_idx,
                    reasoning=parsed.get("reasoning", ""),
                    confidence=float(parsed.get("confidence", 0.5)),
                    phase_strategy=parsed.get("phase_strategy", ""),
                )
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback: try to find just an action_index number
        match = re.search(r'"action_index"\s*:\s*(\d+)', response_text)
        if match:
            action_idx = int(match.group(1))
            action_idx = max(0, min(action_idx, n_legal_moves - 1))
            return LLMDecision(
                action_index=action_idx,
                reasoning="(parsed from partial response)",
                confidence=0.3,
                phase_strategy="",
            )

        # Complete fallback
        return LLMDecision(
            action_index=0,
            reasoning="Failed to parse local model response; defaulting to first move.",
            confidence=0.0,
            phase_strategy="",
        )

    @staticmethod
    def _extract_json(text: str) -> str | None:
        """Extract the first JSON object from text, handling markdown fences."""
        # Try the raw text first
        text = text.strip()
        if text.startswith("{"):
            # Find the matching closing brace
            depth = 0
            for i, ch in enumerate(text):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[: i + 1]

        # Try inside markdown code fence
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)

        # Try any JSON-like block
        match = re.search(r"\{[^{}]*\"action_index\"[^{}]*\}", text, re.DOTALL)
        if match:
            return match.group(0)

        return None

    def _log_distill(self, user_prompt: str, response_text: str) -> None:
        """Log decision for further distillation rounds."""
        record = {
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": response_text.strip()},
            ],
        }
        with open(self._distill_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    @property
    def stats(self) -> dict[str, Any]:
        """Usage statistics (compatible with LLMAgent.stats interface)."""
        avg_time = (
            self._total_gen_time / self._api_calls if self._api_calls > 0 else 0
        )
        return {
            "total_decisions": self._total_decisions,
            "api_calls": self._api_calls,
            "skipped_trivial": self._skipped_trivial,
            "input_tokens": 0,  # Local model — no API tokens
            "output_tokens": self._total_gen_tokens,
            "total_tokens": self._total_gen_tokens,
            "trivial_skip_rate": (
                self._skipped_trivial / self._total_decisions
                if self._total_decisions > 0
                else 0.0
            ),
            "total_gen_time": self._total_gen_time,
            "avg_gen_time": avg_time,
        }
