"""
llm_agent.py — Claude API integration for strategic game decisions.

Uses the Retriever to assemble phase-aware context, formats structured
prompts with FaB strategic knowledge, and gets deterministic action
selections via Claude's tool_use capability.

The LLM receives:
1. A static system prompt encoding FaB rules and strategic principles
2. A dynamic user prompt with current game state, card metadata, and legal moves
3. A tool schema constraining the output to a valid action index + reasoning

This is the highest-value consumer: Claude directly plays the game.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import anthropic

from .retriever import Retriever, RetrievalContext


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


@dataclass
class LLMDecision:
    """Structured output from Claude."""

    action_index: int  # Index into legalMoves
    reasoning: str  # Brief strategic reasoning
    confidence: float  # 0.0-1.0 self-reported confidence
    phase_strategy: str  # Summary of the phase-level plan


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert Flesh and Blood (FaB) card game player making in-game decisions.

## Core Rules
- Each player has a hero with a health total. Reduce the opponent to 0 to win.
- On your turn: play attack actions, use weapons, activate abilities.
- On defense: assign cards from hand and equipment to block incoming attacks.
- Cards have cost (resources to play), power (damage), defense (block value), and pitch (resources when pitched).
- Pitch: discard a card to generate resources equal to its pitch value (Red=1, Yellow=2, Blue=3).

## Rate System
- Baseline value is 3. A card that attacks for 6 power is on-rate (6/2 = 3 value).
- Keywords add value: go-again (+1), dominate (+1.5), piercing (+0.5).
- Cards below rate 3 are under-rate and generally poor plays.

## Strategic Principles
1. **Go-again chains multiply value**: A 3-power go-again into a 4-power closer = 7 total damage from 2 cards.
2. **Dominate forces suboptimal blocks**: The opponent can only block with one card, making the rest of their hand useless.
3. **Equipment is a long-term resource**: Breaking equipment for marginal block value is almost never correct.
4. **Arsenal carries over**: Only arsenal cards with play effects. Pure block/pitch cards are dead in arsenal.
5. **Turn 0 is special**: Both players draw back to intellect at end of turn 0. Hand cards used to block are FREE (replaced). Equipment is permanent — never break equipment on turn 0. Never activate equipment abilities while defending.
6. **Block efficiently**: When defending, exhaust hand cards before touching equipment. Block the minimum needed to prevent on-hit effects.
7. **Pitch ordering matters**: Cards pitched go to the bottom of your deck. Balance colors for your second cycle.
8. **Read the opponent**: Track opponent's hand size, deck count, and visible equipment. Low hand = less defense.

## Decision Framework
For each decision:
1. Identify the phase (offense, defense, arsenal, etc.)
2. Evaluate all legal moves against the current game state
3. Consider sequencing (go-again chains, resource planning)
4. Factor in opponent's likely response
5. Pick the action that maximizes expected value

You will receive the current game state with full card metadata and a list of legal moves.
Select the best action by calling the choose_action tool.
"""


# ---------------------------------------------------------------------------
# LLMAgent
# ---------------------------------------------------------------------------


class LLMAgent:
    """
    Claude-powered game agent.

    Uses Retriever for context, formats phase-specific prompts, and
    parses structured tool_use responses.

    Parameters
    ----------
    retriever:
        Built Retriever instance with card/combo indices.
    model:
        Claude model to use.
    api_key:
        Anthropic API key.  Falls back to ANTHROPIC_API_KEY env var.
    temperature:
        Sampling temperature.  Low for consistency.
    max_tokens:
        Maximum response tokens.
    """

    def __init__(
        self,
        retriever: Retriever,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        self._retriever = retriever
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "No API key provided.  Set ANTHROPIC_API_KEY or pass api_key=."
            )
        self._client = anthropic.Anthropic(api_key=key)

        # Usage tracking
        self._total_decisions = 0
        self._api_calls = 0
        self._skipped_trivial = 0
        self._input_tokens = 0
        self._output_tokens = 0

    def decide(
        self,
        state: dict[str, Any],
        legal_moves: list[dict[str, Any]],
    ) -> LLMDecision:
        """
        Given raw game state and legal moves, ask Claude for an action.

        Trivial decisions (0 or 1 legal moves) are auto-resolved without
        an API call.  This saves 10-30% of calls in a typical game.

        Flow:
        1. Short-circuit if trivial
        2. Retriever.retrieve(state) -> RetrievalContext
        3. Format phase-specific prompt + legal moves
        4. Call Claude API with tool_use
        5. Parse response into LLMDecision
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

        # -- Non-trivial: call Claude --
        self._api_calls += 1

        # 1. Retrieve context
        ctx = self._retriever.retrieve(state)

        # 2. Build prompt
        user_prompt = Retriever.render_for_llm(ctx, legal_moves)

        # 3. Build tool schema
        tool = self._build_tool_schema(len(legal_moves))

        # 4. Call Claude
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=_SYSTEM_PROMPT,
            tools=[tool],
            tool_choice={"type": "tool", "name": "choose_action"},
            messages=[{"role": "user", "content": user_prompt}],
        )

        # Track token usage
        if hasattr(response, "usage"):
            self._input_tokens += response.usage.input_tokens
            self._output_tokens += response.usage.output_tokens

        # 5. Parse response
        return self._parse_response(response, len(legal_moves))

    def _build_tool_schema(self, n_legal_moves: int) -> dict[str, Any]:
        """
        Build the tool_use schema that constrains Claude's output.

        The action_index is bounded to [0, n_legal_moves - 1].
        """
        return {
            "name": "choose_action",
            "description": (
                "Select the best legal game action. "
                "action_index must be a valid index into the LEGAL MOVES list."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "action_index": {
                        "type": "integer",
                        "description": (
                            "Index of the chosen action from the LEGAL MOVES list. "
                            f"Must be between 0 and {n_legal_moves - 1}."
                        ),
                        "minimum": 0,
                        "maximum": max(0, n_legal_moves - 1),
                    },
                    "reasoning": {
                        "type": "string",
                        "description": (
                            "Brief strategic reasoning for this choice "
                            "(2-3 sentences max)."
                        ),
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence in this choice (0.0 to 1.0).",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "phase_strategy": {
                        "type": "string",
                        "description": (
                            "One-sentence summary of the overall strategy "
                            "for this phase/turn."
                        ),
                    },
                },
                "required": ["action_index", "reasoning"],
            },
        }

    def _parse_response(
        self,
        response: Any,
        n_legal_moves: int,
    ) -> LLMDecision:
        """Extract structured decision from Claude's tool_use response."""
        # Find the tool_use block
        for block in response.content:
            if block.type == "tool_use" and block.name == "choose_action":
                inp = block.input
                action_idx = int(inp.get("action_index", 0))

                # Clamp to valid range
                action_idx = max(0, min(action_idx, n_legal_moves - 1))

                return LLMDecision(
                    action_index=action_idx,
                    reasoning=inp.get("reasoning", ""),
                    confidence=float(inp.get("confidence", 0.5)),
                    phase_strategy=inp.get("phase_strategy", ""),
                )

        # Fallback: if no tool_use block found, pick action 0
        return LLMDecision(
            action_index=0,
            reasoning="No tool_use response from Claude; defaulting to first legal move.",
            confidence=0.0,
            phase_strategy="",
        )

    # ------------------------------------------------------------------
    # Usage stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        """Return usage statistics for monitoring API costs."""
        return {
            "total_decisions": self._total_decisions,
            "api_calls": self._api_calls,
            "skipped_trivial": self._skipped_trivial,
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "total_tokens": self._input_tokens + self._output_tokens,
            "trivial_skip_rate": (
                self._skipped_trivial / self._total_decisions
                if self._total_decisions > 0
                else 0.0
            ),
        }
