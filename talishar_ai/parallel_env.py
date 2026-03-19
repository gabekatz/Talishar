"""
parallel_env.py — Run N TalisharEnv instances concurrently using threads.

Since the bottleneck is HTTP round-trips to the PHP engine (I/O-bound),
threading gives near-linear throughput scaling without the overhead of
multiprocessing or the GIL issues that would affect CPU-bound work.

Each env creates its own independent game on the PHP server (different
game IDs, different auth keys), so there are no shared-state conflicts
between workers.

Usage
-----
>>> from talishar_ai.game_manager import GameManager
>>> from talishar_ai.env import TalisharEnv
>>> from talishar_ai.parallel_env import ParallelEnvManager
>>>
>>> def make_env():
...     return TalisharEnv(GameManager(), p1_deck="Ira", p2_deck="Ira")
>>>
>>> mgr = ParallelEnvManager([make_env] * 8)
>>> obs_list, info_list = mgr.reset()           # 8 games in parallel
>>> obs_list, rews, terms, truncs, infos = mgr.step(actions)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable

import numpy as np

from .env import TalisharEnv


class ParallelEnvManager:
    """
    Manages N TalisharEnv instances and steps them in parallel via a thread pool.

    Parameters
    ----------
    env_fns:
        List of zero-argument callables, each returning a freshly-constructed
        TalisharEnv.  One env is created per function.
    """

    def __init__(self, env_fns: list[Callable[[], TalisharEnv]]) -> None:
        self.envs   = [fn() for fn in env_fns]
        self.n_envs = len(self.envs)
        self._pool  = ThreadPoolExecutor(max_workers=self.n_envs)

    # ------------------------------------------------------------------
    # Core API — mirrors gymnasium.vector conventions but returns lists
    # ------------------------------------------------------------------

    def reset(self) -> tuple[list[np.ndarray], list[dict]]:
        """Reset all envs in parallel.  Returns (obs_list, info_list)."""
        futures: list[Future] = [self._pool.submit(env.reset) for env in self.envs]
        results = [f.result() for f in futures]
        return [r[0] for r in results], [r[1] for r in results]

    def step(
        self, actions: list[int]
    ) -> tuple[list[np.ndarray], list[float], list[bool], list[bool], list[dict]]:
        """
        Step all envs in parallel.

        Parameters
        ----------
        actions:
            One integer action per env (length must equal n_envs).

        Returns
        -------
        obs_list, reward_list, terminated_list, truncated_list, info_list
        """
        futures: list[Future] = [
            self._pool.submit(env.step, a)
            for env, a in zip(self.envs, actions)
        ]
        results = [f.result() for f in futures]
        return (
            [r[0] for r in results],  # next obs
            [r[1] for r in results],  # rewards
            [r[2] for r in results],  # terminateds
            [r[3] for r in results],  # truncateds
            [r[4] for r in results],  # infos
        )

    def reset_single(self, idx: int) -> tuple[np.ndarray, dict]:
        """
        Reset one env by index (called when it reaches a terminal state
        mid-rollout so the remaining envs are not blocked).
        """
        return self.envs[idx].reset()

    def close(self) -> None:
        """Shut down the thread pool. Call when training is done."""
        self._pool.shutdown(wait=False)