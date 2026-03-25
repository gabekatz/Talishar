"""
llm_consumer.py — Drop-in action selector powered by Claude.

Replaces the neural network's ``model.act()`` in the game loop.
Given a game state, asks Claude for the best action and returns
the action index.
"""

from __future__ import annotations

from typing import Any

from ..llm_agent import LLMAgent, LLMDecision


class LLMActionConsumer:
    """
    Drop-in replacement for model.act() — Claude picks the action.

    Usage::

        consumer = LLMActionConsumer(llm_agent)
        action_idx, decision = consumer.act(state, state["legalMoves"])
    """

    def __init__(self, llm_agent: LLMAgent) -> None:
        self._agent = llm_agent
        self._decision_count = 0
        self._last_decision: LLMDecision | None = None

    def act(
        self,
        state: dict[str, Any],
        legal_moves: list[dict[str, Any]],
    ) -> tuple[int, LLMDecision]:
        """
        Ask Claude to pick an action.

        Parameters
        ----------
        state:
            Raw GetAIState JSON response.
        legal_moves:
            The legalMoves list from the state.

        Returns
        -------
        (action_index, decision_metadata)
        """
        decision = self._agent.decide(state, legal_moves)
        self._decision_count += 1
        self._last_decision = decision
        return decision.action_index, decision

    @property
    def last_decision(self) -> LLMDecision | None:
        """Most recent LLMDecision (useful for verbose logging in callers)."""
        return self._last_decision

    @property
    def decisions_made(self) -> int:
        return self._decision_count
