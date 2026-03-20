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
        max_poll: int = 40,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.max_poll = max_poll
        self._max_retries = 5
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
        fill_from_inventory: bool = False,
        target_deck_size: int = 60,
    ) -> tuple[str, str, str]:
        """
        Create a new training game.

        Parameters
        ----------
        fill_from_inventory:
            When True, the PHP backend randomly fills the main deck from the
            inventory section of the deck file until it reaches target_deck_size.
            Creates per-game deck variety during training.
        target_deck_size:
            Target main deck size when fill_from_inventory is True (default 60).

        Returns
        -------
        (game_name, p1_auth_key, p2_auth_key)
        """
        last_err: Exception | None = None
        body: dict = {
            "p1_deck": p1_deck,
            "p2_deck": p2_deck,
            "p2_is_ai": p2_is_ai,
            "p1_is_ai": p1_is_ai,
            "format": format,
        }
        if fill_from_inventory:
            body["fill_from_inventory"] = True
            body["target_deck_size"] = target_deck_size
        for attempt in range(self._max_retries):
            try:
                resp = self._post("/game/APIs/CreateTrainingGame.php", body)
                if "error" in resp:
                    raise RuntimeError(f"CreateTrainingGame failed: {resp['error']}")
                return resp["gameName"], resp["p1AuthKey"], resp["p2AuthKey"]
            except RuntimeError as exc:
                last_err = exc
                if attempt < self._max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))
        raise last_err  # type: ignore[misc]

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_state(
        self, game_name: str, player_id: int, auth_key: str
    ) -> dict[str, Any]:
        """Fetch the current AI game state for *player_id*."""
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = self._get(
                    "/game/GetAIState.php",
                    {"gameName": game_name, "playerID": player_id, "authKey": auth_key},
                )
                if "error" in resp:
                    raise RuntimeError(f"GetAIState failed: {resp['error']}")
                return resp
            except RuntimeError as exc:
                last_err = exc
                if attempt < self._max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))
        raise last_err  # type: ignore[misc]

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
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = self._post("/game/SubmitAIAction.php", body)
                if "error" in resp:
                    raise RuntimeError(f"SubmitAIAction failed: {resp['error']}")
                return resp
            except RuntimeError as exc:
                last_err = exc
                if attempt < self._max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))
        raise last_err  # type: ignore[misc]

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
        try:
            return r.json()
        except Exception as exc:
            # Empty or non-JSON response — game likely ended or server errored.
            raise RuntimeError(
                f"Invalid JSON from GET {path} (status {r.status_code}, "
                f"body={r.text[:120]!r})"
            ) from exc

    def _post(self, path: str, body: dict) -> dict[str, Any]:
        url = self.base_url + path
        r = self._session.post(url, json=body, timeout=self.timeout)
        r.raise_for_status()
        try:
            return r.json()
        except Exception as exc:
            raise RuntimeError(
                f"Invalid JSON from POST {path} (status {r.status_code}, "
                f"body={r.text[:120]!r})"
            ) from exc
