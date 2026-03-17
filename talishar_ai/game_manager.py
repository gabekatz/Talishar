"""
game_manager.py — HTTP wrapper around the Talishar PHP engine.

All communication with the game server goes through this class so the rest
of the codebase never touches raw HTTP.
"""

from __future__ import annotations

import time
from typing import Any

import requests


class GameManager:
    """
    Thin client for the three AI-facing PHP endpoints.

    Parameters
    ----------
    base_url:
        Root URL of the running Talishar server, e.g. "http://localhost:8080".
    timeout:
        Per-request HTTP timeout in seconds.
    poll_interval:
        Seconds to wait between state-poll retries when blocking for priority.
    max_poll:
        Maximum number of poll retries before raising a RuntimeError.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 10.0,
        poll_interval: float = 0.25,
        max_poll: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.max_poll = max_poll
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Game lifecycle
    # ------------------------------------------------------------------

    def create_game(
        self,
        p1_deck: str = "Ira",
        p2_deck: str = "Ira",
        p2_is_ai: bool = True,
        p1_is_ai: bool = False,
        format: str = "cc",
    ) -> tuple[str, str, str]:
        """
        Create a new training game.

        Returns
        -------
        (game_name, p1_auth_key, p2_auth_key)
        """
        resp = self._post(
            "/game/APIs/CreateTrainingGame.php",
            {
                "p1_deck": p1_deck,
                "p2_deck": p2_deck,
                "p2_is_ai": p2_is_ai,
                "p1_is_ai": p1_is_ai,
                "format": format,
            },
        )
        if "error" in resp:
            raise RuntimeError(f"CreateTrainingGame failed: {resp['error']}")
        return resp["gameName"], resp["p1AuthKey"], resp["p2AuthKey"]

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_state(
        self, game_name: str, player_id: int, auth_key: str
    ) -> dict[str, Any]:
        """Fetch the current AI game state for *player_id*."""
        resp = self._get(
            "/game/GetAIState.php",
            {"gameName": game_name, "playerID": player_id, "authKey": auth_key},
        )
        if "error" in resp:
            raise RuntimeError(f"GetAIState failed: {resp['error']}")
        return resp

    def get_state_blocking(
        self, game_name: str, player_id: int, auth_key: str
    ) -> dict[str, Any]:
        """
        Poll GetAIState until the game is over OR *player_id* has priority with
        at least one legal move available.

        Returns the state dict.  Raises RuntimeError after *max_poll* attempts.
        """
        for _ in range(self.max_poll):
            state = self.get_state(game_name, player_id, auth_key)
            if self._is_terminal(state):
                return state
            if state.get("havePriority") and state.get("legalMoves"):
                return state
            time.sleep(self.poll_interval)
        raise RuntimeError(
            f"Timed out waiting for priority in game {game_name} (player {player_id})"
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def submit_action(
        self,
        game_name: str,
        player_id: int,
        auth_key: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Submit a move.

        *params* is the ``params`` dict from a ``legalMoves`` entry, e.g.::

            {"mode": 27, "cardID": "0"}
            {"mode": 17, "buttonInput": "Option_A"}
            {"mode": 99}

        Returns the updated AI state after the move (and any P2 AI responses).
        """
        body = {
            "gameName": game_name,
            "playerID": player_id,
            "authKey": auth_key,
            **params,
        }
        resp = self._post("/game/SubmitAIAction.php", body)
        if "error" in resp:
            raise RuntimeError(f"SubmitAIAction failed: {resp['error']}")
        return resp

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_terminal(state: dict[str, Any]) -> bool:
        return (state.get("phase") or {}).get("turnPhase") == "OVER"

    def _get(self, path: str, params: dict) -> dict[str, Any]:
        url = self.base_url + path
        r = self._session.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict) -> dict[str, Any]:
        url = self.base_url + path
        r = self._session.post(url, json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()
