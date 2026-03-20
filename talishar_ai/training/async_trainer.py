"""
async_trainer.py — Asynchronous PPO training loop.

Unlike the synchronous Trainer (trainer.py) which waits for ALL N envs to
complete each step before doing the next model inference, AsyncTrainer gives
each env its own worker thread that loops independently:

    infer(obs) → env.step(action) → store transition → repeat

Workers only synchronise at a threading.Barrier when their rollout buffers
are full.  The main thread then runs the PPO update and releases workers for
the next rollout.

This eliminates the "wait for the slowest env" bottleneck.  In self-play,
each env.step() includes a variable-length P2 driving loop (3–20 HTTP
round-trips).  With synchronous stepping, fast envs idle while the slowest
env finishes.  With async workers, fast envs immediately start their next
step.

Model inference during rollout collection runs on CPU (sub-millisecond for
batch=1) so that worker threads don't contend on MPS's single Metal command
queue.  The PPO update still runs on the configured device (MPS/CUDA/CPU).
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.distributions import Categorical

from ..features import StateEncoder, N_CARD_SLOTS
from ..models.network import ActorCritic
from .rollout import RolloutBuffer
from .ppo import PPOTrainer
from .tb_logger import TBLogger


class _EnvWorker:
    """Per-env worker that collects rollout data in its own thread."""

    def __init__(
        self,
        idx:            int,
        env,                        # TalisharEnv or SelfPlayEnv
        model:          ActorCritic,
        buffer:         RolloutBuffer,
        encoder:        StateEncoder,
        use_embeddings: bool,
        use_lstm:       bool,
        barrier:        threading.Barrier,
        infer_lock:     threading.Lock,
    ) -> None:
        self.idx            = idx
        self.env            = env
        self.model          = model
        self.buffer         = buffer
        self.encoder        = encoder
        self.use_embeddings = use_embeddings
        self.use_lstm       = use_lstm
        self._barrier       = barrier
        self._infer_lock    = infer_lock

        # Per-worker state
        self.obs:       np.ndarray | None = None
        self.mask:      np.ndarray | None = None
        self.card_ids:  np.ndarray | None = None
        self.episode_start: bool = True

        # LSTM hidden state (n_layers, 1, lstm_hidden) — owned by this worker
        self.hidden_h: torch.Tensor | None = None
        self.hidden_c: torch.Tensor | None = None

        # Metrics collected during rollout (read by main thread after barrier)
        self.ep_reward:     float = 0.0
        self.ep_rewards:    list[float] = []
        self.ep_results:    list[str]   = []
        self.ep_game_stats: list[dict]  = []
        self.steps_done:    int   = 0
        self.games_done:    int   = 0

        # Bootstrap value (computed by worker when buffer is full)
        self.last_value: float = 0.0

        # Control
        self._stop = False

    def init_obs(self) -> None:
        """Reset the env and populate initial obs/mask/card_ids."""
        obs, info = self.env.reset()
        self.obs      = obs
        self.mask     = info["legal_mask"]
        self.card_ids = self.encoder.card_ids(info["raw_state"])
        self.episode_start = True

    def run(self) -> None:
        """Worker loop: fill buffer → barrier → wait for PPO update → repeat."""
        while not self._stop:
            try:
                self._collect_rollout()
            except Exception as exc:
                # Connection errors, server crashes, etc. should not kill the
                # worker thread.  Reset the env and pad the buffer so the
                # barrier is reached cleanly.
                print(
                    f"[env-{self.idx}] ERROR in rollout: {exc!r} — "
                    f"resetting env and padding buffer"
                )
                try:
                    self.init_obs()
                    if self.use_lstm:
                        self.hidden_h = torch.zeros_like(self.hidden_h)
                        self.hidden_c = torch.zeros_like(self.hidden_c)
                except Exception:
                    pass  # init_obs can also fail; we'll retry next rollout
                # Pad remaining buffer slots with zero-reward no-ops so the
                # barrier can proceed without deadlock.
                self.buffer.pad_remaining()

            # Compute bootstrap value before hitting barrier
            self._compute_bootstrap()

            # Signal: my buffer is full
            self._barrier.wait()

            # Wait for main thread to finish PPO update + buffer reset
            self._barrier.wait()

            if self._stop:
                break

    def _collect_rollout(self) -> None:
        """Fill this worker's buffer with rollout_steps transitions."""
        self.ep_rewards.clear()
        self.ep_results.clear()
        self.ep_game_stats.clear()
        self.steps_done = 0
        self.games_done = 0

        while not self.buffer.is_full():
            # --- Model inference on CPU (thread-safe, no MPS contention) ---
            obs_t      = torch.from_numpy(self.obs).unsqueeze(0)
            mask_t     = torch.from_numpy(self.mask).unsqueeze(0)
            card_ids_t = torch.from_numpy(self.card_ids).unsqueeze(0).to(torch.int32)

            with torch.no_grad():
                if self.use_lstm:
                    logits, value, new_h, new_c = self.model(
                        obs_t, mask_t,
                        self.hidden_h, self.hidden_c,
                        card_ids_t if self.use_embeddings else None,
                    )
                    self.hidden_h = new_h
                    self.hidden_c = new_c
                else:
                    logits, value = self.model(
                        obs_t, mask_t,
                        card_ids_t if self.use_embeddings else None,
                    )

                dist     = Categorical(logits=logits)
                action_t = dist.sample()
                log_prob = dist.log_prob(action_t)

            action   = int(action_t.item())
            lp       = float(log_prob.item())
            val      = float(value.item())

            # --- Env step (blocking HTTP — this is where time is spent) ---
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Store transition
            self.buffer.add(
                obs           = self.obs,
                action        = action,
                log_prob      = lp,
                value         = val,
                reward        = float(reward),
                done          = done,
                action_mask   = self.mask,
                card_ids      = self.card_ids,
                episode_start = self.episode_start,
            )

            self.ep_reward  += float(reward)
            self.steps_done += 1

            if done:
                self.ep_rewards.append(self.ep_reward)
                self.games_done += 1
                if info.get("result"):
                    self.ep_results.append(info["result"])
                if info.get("game_stats"):
                    gs = info["game_stats"]
                    self.ep_game_stats.append(
                        gs.to_dict() if hasattr(gs, "to_dict") else gs
                    )
                self.ep_reward = 0.0

                # Auto-reset
                next_obs, info = self.env.reset()
                self.episode_start = True

                # LSTM: wipe hidden state for new episode
                if self.use_lstm:
                    self.hidden_h = torch.zeros_like(self.hidden_h)
                    self.hidden_c = torch.zeros_like(self.hidden_c)
            else:
                self.episode_start = False

            # Advance
            self.obs      = next_obs
            self.mask     = info["legal_mask"]
            self.card_ids = self.encoder.card_ids(info["raw_state"])

    def _compute_bootstrap(self) -> None:
        """Compute V(s) for the state after the last buffered transition."""
        obs_t      = torch.from_numpy(self.obs).unsqueeze(0)
        mask_t     = torch.from_numpy(self.mask).unsqueeze(0)
        card_ids_t = torch.from_numpy(self.card_ids).unsqueeze(0).to(torch.int32)

        with torch.no_grad():
            if self.use_lstm:
                _, value, _, _ = self.model(
                    obs_t, mask_t,
                    self.hidden_h, self.hidden_c,
                    card_ids_t if self.use_embeddings else None,
                )
            else:
                _, value = self.model(
                    obs_t, mask_t,
                    card_ids_t if self.use_embeddings else None,
                )
        self.last_value = float(value.item())


class AsyncTrainer:
    """
    Asynchronous PPO trainer — each env runs in its own thread.

    Drop-in replacement for Trainer with the same constructor signature.
    """

    def __init__(
        self,
        envs:             list,          # list of TalisharEnv / SelfPlayEnv
        model:            ActorCritic,
        ppo:              PPOTrainer,
        rollout_steps:    int                          = 512,
        checkpoint_dir:   str                          = "checkpoints",
        checkpoint_freq:  int                          = 50_000,
        log_freq:         int                          = 10,
        device:           torch.device | None          = None,
        encoder:          StateEncoder | None          = None,
        post_update_fn:   Callable[[int, dict], None] | None = None,
        tb_logger:        TBLogger | None              = None,
    ) -> None:
        self.model           = model
        self.ppo             = ppo
        self.rollout_steps   = rollout_steps
        self.checkpoint_dir  = Path(checkpoint_dir)
        self.checkpoint_freq = checkpoint_freq
        self._project_root   = Path(__file__).resolve().parents[2]
        self.log_freq        = log_freq
        self.device          = device or torch.device("cpu")
        self._encoder        = encoder or StateEncoder()
        self.use_embeddings  = model.use_embeddings
        self._post_update_fn = post_update_fn
        self._tb             = tb_logger
        self.use_lstm        = getattr(model, "use_lstm", False)
        self.n_envs          = len(envs)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create a CPU copy of the model for rollout-collection inference.
        # Workers use this (read-only, no_grad) so they don't contend on MPS.
        self._infer_model = _cpu_inference_copy(model)

        # Barrier: N workers + 1 main thread.  Used twice per rollout:
        # 1) workers signal "buffer full" → main runs PPO update
        # 2) main signals "update done" → workers resume
        self._barrier    = threading.Barrier(self.n_envs + 1)
        self._infer_lock = threading.Lock()  # reserved for future use

        # Build workers
        self._buffers = [
            RolloutBuffer(capacity=rollout_steps, device=self.device)
            for _ in range(self.n_envs)
        ]
        self._workers = [
            _EnvWorker(
                idx            = i,
                env            = envs[i],
                model          = self._infer_model,
                buffer         = self._buffers[i],
                encoder        = self._encoder,
                use_embeddings = self.use_embeddings,
                use_lstm       = self.use_lstm,
                barrier        = self._barrier,
                infer_lock     = self._infer_lock,
            )
            for i in range(self.n_envs)
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, total_steps: int, start_step: int = 0) -> None:
        """Run the async training loop for *total_steps* environment steps."""
        model = self.model.to(self.device)
        if self.use_lstm and self.device.type == "mps":
            model.lstm = model.lstm.cpu()

        # Sync the CPU inference model with the latest weights
        self._sync_infer_model()

        # Initialise all workers (reset envs, set up LSTM hidden states)
        for w in self._workers:
            w.init_obs()
            if self.use_lstm:
                w.hidden_h, w.hidden_c = self._infer_model.init_hidden(1, torch.device("cpu"))
                w.buffer.initial_hidden_h = w.hidden_h.squeeze(1).numpy()
                w.buffer.initial_hidden_c = w.hidden_c.squeeze(1).numpy()

        # Start worker threads
        threads: list[threading.Thread] = []
        for w in self._workers:
            t = threading.Thread(target=w.run, daemon=True, name=f"env-{w.idx}")
            threads.append(t)
            t.start()

        global_step     = start_step
        target_step     = start_step + total_steps
        update_count    = 0
        last_checkpoint = global_step
        t0              = time.time()

        all_ep_rewards: list[float] = []
        all_ep_results: list[str]   = []

        # Heartbeat
        _hb_time     = time.time()
        _hb_steps    = 0
        _hb_games    = 0
        _HB_INTERVAL = 60.0

        # Docker health tracking
        _consecutive_dead_rounds = 0
        _DEAD_THRESHOLD = 2  # restart after this many consecutive dead rounds
        _MIN_HEALTHY_STEPS = self.rollout_steps  # expect at least 1 env's worth

        transitions_per_update = self.rollout_steps * self.n_envs
        print(
            f"[AsyncTrainer] Starting training for {total_steps:,} steps "
            f"(start={start_step:,}, target={target_step:,}, n_envs={self.n_envs})"
        )
        print(
            f"[AsyncTrainer] Device: {self.device} | "
            f"Rollout: {self.rollout_steps} steps × {self.n_envs} envs "
            f"= {transitions_per_update:,} transitions/update"
        )

        try:
            while global_step < target_step:
                # ---- Wait for all workers to fill their buffers ----
                self._barrier.wait()

                # Collect metrics from workers
                for w in self._workers:
                    global_step += w.steps_done
                    _hb_steps   += w.steps_done
                    _hb_games   += w.games_done
                    all_ep_rewards.extend(w.ep_rewards)
                    all_ep_results.extend(w.ep_results)
                    if self._tb:
                        for gs in w.ep_game_stats:
                            self._tb.log_game(global_step, gs)

                # ---- Docker health check ----
                round_steps = sum(w.steps_done for w in self._workers)
                if round_steps < _MIN_HEALTHY_STEPS:
                    _consecutive_dead_rounds += 1
                    if _consecutive_dead_rounds >= _DEAD_THRESHOLD:
                        print(
                            f"[AsyncTrainer] {_consecutive_dead_rounds} consecutive "
                            f"dead rounds ({round_steps} steps) — triggering Docker restart"
                        )
                        self._restart_docker()
                        for w in self._workers:
                            w.buffer.reset()
                        self._reinit_workers()
                        _consecutive_dead_rounds = 0
                        # Release workers to collect fresh rollouts
                        self._barrier.wait()
                        continue
                else:
                    _consecutive_dead_rounds = 0

                # ---- Heartbeat ----
                now = time.time()
                if now - _hb_time >= _HB_INTERVAL:
                    elapsed_min = (now - _hb_time) / 60.0
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(
                        f"[{ts}] [heartbeat] "
                        f"{_hb_steps:,} steps | "
                        f"{_hb_games} games | "
                        f"{_hb_steps / elapsed_min:,.0f} steps/min | "
                        f"global={global_step:,}/{target_step:,}"
                    )
                    _hb_time  = now
                    _hb_steps = 0
                    _hb_games = 0

                # ---- Compute GAE per buffer ----
                normalize_per_env = (not self.use_lstm) and (self.n_envs == 1)
                for w in self._workers:
                    w.buffer.compute_returns_and_advantages(
                        w.last_value, normalize=normalize_per_env
                    )

                # ---- PPO update ----
                if self.use_lstm:
                    stats = self.ppo.update_lstm(self._buffers)
                else:
                    merged = RolloutBuffer.merge(self._buffers, self.device)
                    stats  = self.ppo.update(merged)

                # ---- Reset buffers and sync inference model ----
                for w in self._workers:
                    w.buffer.reset()
                    if self.use_lstm:
                        w.buffer.initial_hidden_h = w.hidden_h.squeeze(1).numpy()
                        w.buffer.initial_hidden_c = w.hidden_c.squeeze(1).numpy()

                update_count += 1
                self._sync_infer_model()

                # ---- Post-update callback (opponent rotation etc.) ----
                if self._post_update_fn is not None:
                    self._post_update_fn(update_count, self.model.state_dict())

                # ---- Logging ----
                if update_count % self.log_freq == 0:
                    self._log(
                        global_step, target_step, stats,
                        all_ep_rewards, all_ep_results, t0,
                    )
                    all_ep_rewards.clear()
                    all_ep_results.clear()

                # ---- Checkpoint ----
                if global_step - last_checkpoint >= self.checkpoint_freq:
                    self._save(global_step)
                    last_checkpoint = global_step

                # Stop workers if we've reached the target
                if global_step >= target_step:
                    for w in self._workers:
                        w._stop = True

                # ---- Release workers for next rollout ----
                self._barrier.wait()

        except KeyboardInterrupt:
            print("\n[AsyncTrainer] Interrupted — saving checkpoint...")
            for w in self._workers:
                w._stop = True
            # Release workers so threads can exit
            try:
                self._barrier.wait(timeout=1)
            except threading.BrokenBarrierError:
                pass

        self._save(global_step, final=True)
        if self._tb:
            self._tb.close()
        print(f"[AsyncTrainer] Training complete ({global_step:,} steps).")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _restart_docker(self) -> None:
        """Stop Docker, clean game files, restart, and wait until healthy."""
        root = self._project_root
        games_dir = root / "Games"

        print("[AsyncTrainer] Docker unhealthy — restarting...")

        # 1. Stop containers
        subprocess.run(["bash", "stop.sh"], cwd=root, timeout=60,
                        capture_output=True)

        # 2. Remove accumulated game directories (the cause of resource exhaustion)
        if games_dir.exists():
            shutil.rmtree(games_dir, ignore_errors=True)
            games_dir.mkdir(mode=0o777, exist_ok=True)

        # 3. Start containers
        subprocess.run(["bash", "start.sh"], cwd=root, timeout=120,
                        capture_output=True)

        # 4. Wait for PHP to be ready (poll the server)
        import requests
        env0 = self._workers[0].env
        # SelfPlayEnv wraps TalisharEnv as _env; TalisharEnv has .gm directly
        inner = getattr(env0, "_env", env0)
        base_url = getattr(getattr(inner, "gm", None), "base_url", "http://localhost:8080")
        for i in range(30):
            try:
                r = requests.get(f"{base_url}/game/GetAIState.php",
                                 params={"gameName": "0", "playerID": "1",
                                         "authKey": "test"},
                                 timeout=3)
                if r.status_code == 200 and r.text.strip():
                    break
            except Exception:
                pass
            time.sleep(2)

        print("[AsyncTrainer] Docker restarted successfully.")

    def _reinit_workers(self) -> None:
        """Re-initialise all worker envs after a Docker restart."""
        for w in self._workers:
            try:
                w.init_obs()
                if self.use_lstm:
                    w.hidden_h, w.hidden_c = self._infer_model.init_hidden(
                        1, torch.device("cpu")
                    )
                    w.buffer.initial_hidden_h = w.hidden_h.squeeze(1).numpy()
                    w.buffer.initial_hidden_c = w.hidden_c.squeeze(1).numpy()
            except Exception as exc:
                print(f"[AsyncTrainer] WARNING: worker {w.idx} re-init failed: {exc}")

    def _sync_infer_model(self) -> None:
        """Copy the training model's weights to the CPU inference model."""
        state = self.model.state_dict()
        self._infer_model.load_state_dict(
            {k: v.cpu() for k, v in state.items()}
        )

    def _log(
        self,
        step:       int,
        total:      int,
        stats:      dict[str, float],
        ep_rewards: list[float],
        ep_results: list[str],
        t0:         float,
    ) -> None:
        elapsed  = time.time() - t0
        sps      = step / elapsed
        mean_rew = np.mean(ep_rewards) if ep_rewards else float("nan")
        # Exclude opponent-truncated games from win% — they don't reflect
        # P1's actual play quality.
        real_results = [r for r in ep_results if r != "truncated"]
        n_ep     = len(real_results)
        win_rate = real_results.count("win") / n_ep if n_ep else float("nan")
        n_trunc  = ep_results.count("truncated")

        now = datetime.now().strftime("%H:%M:%S")
        trunc_str = f" trunc={n_trunc}" if n_trunc else ""
        print(
            f"[{now}] [{step:>8,}/{total:,}] "
            f"sps={sps:,.0f} | "
            f"ep_rew={mean_rew:+.3f} | "
            f"win%={win_rate:.1%} ({n_ep} ep){trunc_str} | "
            f"π={stats['policy_loss']:+.4f} "
            f"V={stats['value_loss']:.4f} "
            f"H={stats['entropy']:.4f}"
        )

        if self._tb:
            self._tb.log_training(step, stats, ep_rewards, ep_results, sps)

    def _save(self, step: int, final: bool = False) -> None:
        tag  = "final" if final else f"{step}"
        path = self.checkpoint_dir / f"model_{tag}.pt"
        torch.save(
            {
                "step":        step,
                "model_state": self.model.state_dict(),
                "optim_state": self.ppo.optimizer.state_dict(),
            },
            path,
        )
        print(f"[AsyncTrainer] Saved checkpoint → {path}")


def _cpu_inference_copy(model: ActorCritic) -> ActorCritic:
    """Create a CPU copy of the model for thread-safe rollout inference."""
    import copy
    cpu_model = copy.deepcopy(model).cpu()
    cpu_model.eval()
    for p in cpu_model.parameters():
        p.requires_grad_(False)
    return cpu_model
