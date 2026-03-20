# Engineering Notes: Talishar AI — Design & Reasoning

This document explains every decision made when building the AI layer on top of
the Talishar game engine, written for a developer learning ML who wants to
understand *why* each piece exists, not just *what* it does.

---

## 1. Problem framing

Flesh and Blood (FaB) is a two-player, imperfect-information card game. Each
player holds a hidden hand, makes sequential decisions, and the game ends when
one hero's life total reaches 0.

We frame this as a **Markov Decision Process (MDP)**:

| MDP component | FaB equivalent |
|---|---|
| State *s* | Visible game state (health, zones, phase, CC) |
| Action *a* | One legal move from the current player's options |
| Reward *r* | +1 win, −1 loss; shaped by per-step health delta |
| Transition *T* | The game engine's ProcessInput() |
| Episode | One complete game |

Because we cannot observe the opponent's hand, this is strictly a **Partially
Observable MDP (POMDP)** — the agent never sees the full state of the world.
We tackle this at two levels:
1. **Card identity embeddings** (feature #2): give the model a richer description
   of *what* it can see.
2. **LSTM recurrent policy** (feature #5): give the model *memory* so it can
   infer hidden information from the history of what it has observed.

---

## 2. Why PPO?

Several RL algorithms could work here. We chose **Proximal Policy
Optimisation (PPO-Clip)** for these reasons:

| Criterion | PPO | DQN | MCTS |
|---|---|---|---|
| Discrete, variable action space | ✓ (with masking) | ✓ | ✓ |
| No environment model needed | ✓ | ✓ | needs model |
| Stable training | ✓ (clip) | ✗ (TD instability) | ✓ |
| Sample efficiency | moderate | low | high but slow |
| Implementation complexity | low | medium | high |

PPO's **clip** objective `min(r·A, clip(r,1-ε,1+ε)·A)` prevents the policy
from taking steps so large that the new policy collapses. This is the main
source of stability compared to vanilla policy gradients (REINFORCE).

**The ratio `r`** is `π_new(a|s) / π_old(a|s)` — how much more or less likely
the new policy is to take the same action the old policy took. Clipping it to
`[1-ε, 1+ε]` means "don't change your probabilities by more than ε in one
update." This is the insight that makes PPO work: big policy updates are the
main cause of training instability in RL, so we explicitly forbid them.

---

## 3. The observation vector (features.py)

### Why a flat vector instead of a graph or sequence?

A graph neural network (GNN) would be more principled — each card is a node,
board state is a graph — but it adds significant complexity for a first version.
A flat feature vector works well for MLP policies and is fast to compute.

### Card encoding (14 floats per card)

```
[cost/10, power/10, defense/10, pitch/3, type_onehot×8, counters/5, tapped]
```

- **Normalisation** (dividing by max): keeps all inputs in [0,1] so the network
  doesn't have to learn different weight scales per feature.
- **Type one-hot**: card type (AA, R, DR, …) is categorical; one-hot encoding
  is the standard way to feed categorical variables to an MLP.
- **Zero-padding**: zones like hand (max 7 cards) are zero-padded when they
  contain fewer cards. The network learns "empty slot = zeros".

### Fixed vs. variable dimension

FaB hands can hold 1–7 cards; equipment slots are fixed. We allocate the
*maximum* for each zone and pad. This makes the observation space
`gymnasium.spaces.Box` with a fixed shape — required by most RL frameworks.

---

## 4. Action masking (why it matters)

FaB has a **variable action space**: the number of legal moves changes every
step (0–50+). Vanilla PPO would assign non-zero probability to every action
slot and occasionally sample an illegal move. Two problems with that:

1. The environment would crash or return an error.
2. The policy wastes capacity learning not-to-do illegal things.

**Action masking** sets the logit of every illegal action to −∞ before
softmax, so their probability is exactly 0. The gradient never flows through
masked slots.

Implementation in `network.py`:
```python
logits = logits.masked_fill(~action_mask, -1e9)
```

The mask is a `bool` tensor of shape `(MAX_ACTIONS,)`. It's stored in the
rollout buffer alongside the observation so we can reconstruct it during the
PPO update.

---

## 5. The Actor-Critic architecture (network.py)

```
obs (413) → Linear(413,256) → LayerNorm → ReLU
          → Linear(256,256) → LayerNorm → ReLU
          ↓                            ↓
  actor: Linear(256,64)      critic: Linear(256,1)
  → mask → Categorical dist  → V(s)
```

### Shared trunk

Policy and value function share the first two layers. This is standard in
PPO because:
- Both need similar game-state representations.
- Fewer total parameters → faster training.
- The critic regularises the shared layers, which helps policy generalisation.

### LayerNorm instead of BatchNorm

Batch Normalisation computes statistics over a mini-batch. But during rollout
collection we process one step at a time (batch size = 1), so batch statistics
are meaningless. Layer Normalisation computes statistics per-sample, which is
stable at any batch size.

### Orthogonal initialisation

`nn.init.orthogonal_` initialises weights so the matrix is orthogonal (rows
are orthonormal). This is the recommended init for deep RL (from OpenAI's
baselines) because it preserves gradient magnitude through many layers.

---

## 6. GAE — Generalised Advantage Estimation (rollout.py)

The **advantage** A(s,a) = Q(s,a) − V(s) measures "how much better is this
action than average?". We need it to be low-variance for stable PPO updates.

**GAE** interpolates between:
- λ=0: one-step TD error (low variance, high bias)
- λ=1: Monte Carlo return (high variance, zero bias)

```
δₜ  = rₜ + γ·V(sₜ₊₁)·(1−done) − V(sₜ)    # TD error
Aₜ  = δₜ + (γλ)·Aₜ₊₁·(1−done)             # recursive GAE
Rₜ  = Aₜ + V(sₜ)                           # discounted return
```

We then normalise advantages: `A ← (A − mean) / (std + ε)`. This keeps the
gradient scale stable regardless of the reward magnitude.

---

## 7. Reward design

### Sparse terminal reward

```
+1  if P1 wins
-1  if P1 loses
 0  draw
```

Pure sparse rewards are hard to learn from: the agent might play hundreds of
steps before seeing any signal.

### Dense shaping

```
shaped = 0.01 × (opp_health_drop − my_health_drop)
```

This gives small positive rewards for dealing damage and small negative rewards
for taking damage, providing gradient signal *every step*.

**Potential-based shaping** (Ng et al. 1999) guarantees that shaping doesn't
change the optimal policy if `shaped_reward = γ·Φ(s') − Φ(s)` where Φ is the
potential function. Our reward uses health totals as the potential, which is
approximately potential-based.

### Clipping to [−1, 1]

Prevents extreme reward values (e.g. first turn vs. long game) from causing
large gradient updates.

---

## 8. The training loop (trainer.py)

```
LOOP:
  1. act(obs, mask) → action, log_prob, value
  2. env.step(action) → next_obs, reward, done, info
  3. buffer.add(...)
  WHEN buffer full (every 512 steps):
  4. bootstrap last_value = V(next_obs)
  5. buffer.compute_returns_and_advantages(last_value)
  6. ppo.update(buffer)   # 4 epochs × mini-batches of 64
  7. buffer.reset()
```

**Why 512-step rollouts?** Shorter than a full game (typically 50–150 turns),
which means more frequent updates and better exploration. The bootstrap value
approximates the return for the unfinished episode.

**Why 4 epochs?** Standard PPO uses 4–10 epochs per rollout. More epochs
squeeze more gradient signal from each batch of data; too many epochs violates
the "on-policy" assumption (the data was collected under the *old* policy).

---

## 9. The PHP layer (CreateTrainingGame.php)

The game engine requires:
1. A lobby file `GameFile.txt` (player info, auth keys, deck links)
2. A game state file `gamestate.txt` (zones, hands, turn state)

`CreateTrainingGame.php` combines what the normal UI flow does in 3 steps
(CreateGame → JoinGame → Start) into a single API call. It:

1. Calls `GetGameCounter()` for a unique numeric game ID
2. Copies deck files from `Assets/`
3. Generates auth keys with `hash('sha256', rand())`
4. Calls `WriteGameFile()` (from `MenuFiles/WriteGamefile.php`)
5. Writes `gamestate.txt` directly (mirrors `Start.php`)
6. Calls `initializePlayerState()` for each player (reads deck → writes zone data)
7. Caches the gamestate with `WriteGamestateCache()`
8. Runs `ParseGamestate.php` + `StartEffects.php` (hero start-of-game effects)
9. Returns `{gameName, p1AuthKey, p2AuthKey}`

Auth keys are how the engine authenticates actions. The Python AI stores both
and uses `p1AuthKey` for all P1 moves.

---

## 10. Feature: Parallel Environments (parallel_env.py)

### The bottleneck problem

Profiling the original single-env training revealed that the GPU/CPU sat idle
most of the time. The bottleneck was not computation — it was the HTTP
round-trips to the PHP game engine. Each `env.step()` call takes ~50–200 ms
waiting for the server.

With one environment, the timeline looks like:
```
[HTTP wait]──[model inference]──[HTTP wait]──[model inference]──...
             (instant)                       (instant)
```

### The solution: thread-based parallelism

Because the bottleneck is I/O (waiting for HTTP responses), not CPU, we can
use **threads** rather than processes. Multiple threads can all be waiting on
HTTP simultaneously without blocking each other.

```
env 0: [HTTP wait]──────────────[HTTP wait]──────────────...
env 1: ──[HTTP wait]──────────────[HTTP wait]─────────────...
env 2: ────[HTTP wait]──────────────[HTTP wait]───────────...
env 3: ──────[HTTP wait]──────────────[HTTP wait]─────────...
       ↑ all waiting at the same time
       model runs one batched forward pass when all results arrive
```

`ThreadPoolExecutor` dispatches all `env.step()` calls concurrently. The model
then processes all N observations in a single batched forward pass.

### Key design decisions

**Each env gets its own `GameManager` (its own `requests.Session`)**. HTTP
sessions are not thread-safe to share. One session per env prevents race
conditions in the connection pool.

**Per-env `RolloutBuffer`**: each environment's trajectory must be kept
separate until PPO update time. This is because **GAE is computed over
contiguous trajectories** — if you mixed steps from different envs, the
"next value" calculation would be wrong (env 2's next state is not a
continuation of env 1's current state).

**Global advantage normalisation**: after computing GAE per-env, we merge all
N buffers and normalise advantages *across all envs together* before the PPO
update. Normalising per-env first would destroy the relative scale of
advantages across different games.

---

## 11. Feature: Card Identity Embeddings (card_vocab.py, features.py, network.py)

### The problem with stat-only encoding

The original observation encodes each card as 14 floats (cost, power, type,
etc.). Two different cards with the same stats look identical to the model.
In FaB, card *identity* matters enormously — a `Timesnap Potion` and a
`Surging Strike` might have similar stats but completely different strategic
implications.

### Embeddings: learning a card's "personality"

An **embedding table** (`nn.Embedding`) maps each unique card ID to a learned
vector of floats. Think of it as a lookup table where each row is a
"personality vector" for that card, and the network learns what those
personalities should be during training.

```
card_id: "timesnap_potion" → index 3847 → embedding[3847] → [0.3, -0.1, 0.7, ...]
card_id: "surging_strike"  → index 1204 → embedding[1204] → [-0.2, 0.8, 0.1, ...]
```

These vectors are learned end-to-end: the network figures out which aspects
of card identity matter for decision-making.

### Architecture: concatenate, don't replace

We **concatenate** the embedding block to the existing stat features rather
than replacing them. This is intentional:

```
[stat features (413 floats) | card embeddings (27 slots × 32 dims = 864 floats)]
→ Linear(1277, 256) → ...
```

The stat features still carry useful information (cost, power, etc.). The
embeddings add *identity* on top. The network can use both.

### `padding_idx=0`: the empty slot trick

```python
nn.Embedding(vocab_size, emb_dim, padding_idx=0)
```

Empty card slots (zero-padded in the observation) map to index 0. With
`padding_idx=0`, the embedding at index 0 is always zero and its gradient
is suppressed. This means "no card here" always produces a zero vector,
which is exactly the right inductive bias.

### Small initialisation (`std=0.01`)

Embeddings are initialised with a very small normal distribution (standard
deviation 0.01). The stat features start with typical orthogonal init, so
their initial scale is much larger. The small embedding init prevents the
fresh, untrained embeddings from overwhelming the already-meaningful stat
features at the start of training.

### Building the vocab (card_vocab.py)

The vocab is built by parsing `GeneratedCardDictionaries.php` with a regex
to extract all card IDs. The result is a JSON file mapping each card name
to an integer index, which is checked in alongside the code. This is
intentional: the vocab should be stable between runs so checkpoint
embeddings remain interpretable.

---

## 12. Feature: Self-Play (training/self_play.py)

### Why self-play?

Training against `EncounterAI` (the server's built-in AI) has a fundamental
limitation: **it's a fixed, weak opponent**. Once the policy consistently
beats EncounterAI, the training signal degenerates. The agent stops learning
because it has already found a strategy that wins against this specific
opponent — but that strategy may not generalise.

This is called **overfitting to the opponent**. The model learns to exploit
EncounterAI's specific weaknesses rather than learning to play FaB.

### The auto-curriculum intuition

Self-play solves this by making the policy train against a frozen copy of
*itself*. As the active policy improves, so does the frozen opponent (when
rotated). This creates an **auto-curriculum**: the game is always
approximately as hard as the agent's current skill level.

This is the same technique used in AlphaGo, OpenAI Five (Dota), and most
modern game-playing AI systems.

### Implementation: frozen opponent

```
SelfPlayEnv wraps TalisharEnv(p2_is_ai=False)

P1: active (learning) policy  ←── gets gradients
P2: frozen copy of the policy ←── no gradients, weights updated periodically
```

`copy.deepcopy(model)` creates a completely independent copy of the model.
`requires_grad_(False)` tells PyTorch not to track gradients through it —
we're only using it for inference.

### Weight rotation

`SelfPlayManager.maybe_rotate()` is called after every PPO update. When
`update_count % update_freq == 0`, it calls `update_opponent(state_dict)` on
every SelfPlayEnv, replacing the frozen weights with the current active
weights.

**Why not rotate every update?** A too-quickly-moving target makes training
unstable — the agent can't learn against an opponent that keeps changing.
Rotating every 20–50 updates gives the policy enough time to improve before
the opponent catches up.

**`{k: v.clone()}`**: each tensor is explicitly cloned rather than just
loading by reference. This ensures the frozen model has no shared tensor
storage with the active model — if we updated the active model's weights,
the frozen model would silently update too without the clone.

### P2 driving loop

`_drive_p2()` polls the game state after each P1 action and submits P2's
responses using the frozen policy. It runs in a loop until P1 regains
priority (or the game ends). The loop includes a sleep for when neither
player has priority (the engine is resolving triggers/effects).

---

## 13. Feature: Evaluation Framework (evaluation/)

### Why a separate evaluation system?

The training loop optimises the policy but doesn't tell you if it's actually
*getting better*. You need a separate, unbiased measurement system.

### Game statistics (game_stats.py)

`GameStatsCollector` tracks per-game metrics beyond win/loss:
- Damage dealt and taken
- Deck cards remaining (a proxy for resource efficiency)
- Number of turns
- Whether the game was truncated (hit the step limit)

These give you diagnostic information. If your agent is winning but always
running the opponent out of deck rather than reducing health, that tells you
something about the strategy it's found.

### Elo rating (elo.py)

**Elo** is the rating system used in chess, originally developed by Arpad Elo.
The key insight is that ratings should be *relative* to opponents, not
absolute scores.

```
expected_win_probability = 1 / (1 + 10^((rating_B - rating_A) / 400))
```

After a game:
```
new_rating = old_rating + K × (actual_result - expected_result)
```

Where K=32 is how much a single game can change your rating. A win against
a much stronger opponent moves your rating a lot; a win against a much weaker
opponent moves it very little.

We use Elo to track checkpoint quality over time. The EncounterAI is treated
as a fixed reference opponent (constant Elo), so checkpoint Elo values are
comparable across training runs.

### Why not just track win rate?

Win rate against a fixed opponent has a ceiling: once you win 90%+ of games,
the signal is noise. Elo continues to differentiate between policies even
when both beat the reference opponent, because it accounts for *margin* and
*consistency*. More importantly, Elo lets you compare any two checkpoints
directly via `head_to_head.py`.

---

## 14. Feature: Recurrent Policy / LSTM (models/lstm_network.py)

### The partial observability problem, revisited

In FaB, the opponent's hand is hidden. When you see your opponent pass
priority without playing a reaction, that's information — maybe they don't
have one. When they pitch a card, that's information about their hand. A
policy that only sees the current state has to make decisions without this
context.

An **LSTM (Long Short-Term Memory)** network has a **hidden state** — a
vector that persists across timesteps — that acts as a learned memory. The
network decides what to store and what to forget at each step.

### Architecture

```
obs + [optional card embeddings]
    ↓
Linear(obs_dim, hidden) → LayerNorm → ReLU    ← pre-LSTM encoder
    ↓
LSTM(hidden, lstm_hidden)                      ← recurrent memory
    ↓
    ├── actor:  Linear(lstm_hidden, MAX_ACTIONS) → masked Categorical
    └── critic: Linear(lstm_hidden, 1) → V(s, h)
```

The pre-LSTM encoder compresses the raw observation into a meaningful
representation before feeding it to the LSTM. The LSTM then updates its
hidden state `(h, c)` at each timestep. Both `h` (hidden) and `c` (cell)
are vectors of size `lstm_hidden` — they carry information forward.

### The hidden state management problem

This is where LSTM complicates the training loop significantly.

**During rollout collection**: the hidden state `(h, c)` for each of the N
parallel envs must be tracked and updated every step. When an episode ends
(game over), that env's hidden state must be reset to zeros so the next
game starts with a blank memory.

**During PPO update**: we can't just shuffle transitions randomly anymore.
The LSTM's re-evaluation of step 200 requires that it has seen steps 0–199
first. Shuffling would give it a random hidden state at the start of each
mini-batch, making the loss calculation wrong.

### Sequence processing in the PPO update

Instead of random mini-batches, we process each env's full rollout as one
**sequence** per epoch:

```python
for epoch in range(n_epochs):
    for buffer in per_env_buffers:
        log_probs, values, entropy = model.evaluate_sequence(
            buffer.obs,          # (T, obs_dim) — T steps in order
            buffer.action_masks,
            h0, c0,              # stored initial hidden state
            episode_starts,      # where to reset h,c within the sequence
            buffer.actions,
        )
        # compute PPO loss, backprop through time
```

This is called **BPTT (Backpropagation Through Time)** — gradients flow
backwards through the sequence.

### Episode boundary handling

A 512-step rollout from one env will often contain multiple complete games
(each game is ~50–150 turns). At each game boundary, the hidden state must
be reset to zeros. We track this with an `episode_starts` boolean array
stored in the rollout buffer.

`evaluate_sequence` processes the rollout by splitting it at episode
boundaries and processing each segment as a separate LSTM call, which is
both correct and efficient (a single LSTM call can process a long segment
using CUDA kernels):

```
rollout: [game 1: t=0..73] [game 2: t=74..201] [game 3: t=202..511]
                           ↑                    ↑
                       episode_starts[74]=True  episode_starts[202]=True

process: LSTM(game1, h0=zeros) → LSTM(game2, h0=zeros) → LSTM(game3, h0=zeros)
```

### Why store the initial hidden state per buffer?

We store `buf.initial_hidden_h / initial_hidden_c` — the hidden state at the
very start of each rollout. During the PPO update (which happens after the
rollout is collected), we replay the sequence from this stored starting point
to get the exact same hidden-state trajectory as during collection. Without
this, the re-evaluated log probabilities would be computed with a different
hidden state than the original, making the PPO importance ratio `r = π_new/π_old`
incorrect.

---

## 15. How to run with all features enabled

```bash
# Full command: LSTM + embeddings + 4 parallel envs + self-play
uv run python -m scripts.train \
  --use-lstm       --lstm-hidden 256  --lstm-layers 1 \
  --use-embeddings --emb-dim 32 \
  --n-envs 4 \
  --self-play      --opponent-update-freq 20 \
  --total-steps 2_000_000 \
  --checkpoint-dir checkpoints/

# Evaluate a checkpoint (produces Elo + game stats)
uv run python -m scripts.evaluate \
  --checkpoint checkpoints/model_final.pt \
  --n-games 20 \
  --log-dir eval_logs/

# Head-to-head: compare two checkpoints directly
uv run python -m scripts.head_to_head \
  --checkpoint-a checkpoints/model_500000.pt \
  --checkpoint-b checkpoints/model_final.pt \
  --n-games 10
```

### Flag reference

| Flag | Feature | Effect when omitted |
|---|---|---|
| `--n-envs N` | Parallel envs | 1 env (slower) |
| `--use-embeddings` | Card ID embeddings | Float stats only |
| `--emb-dim D` | Embedding width | 32 (default) |
| `--self-play` | Self-play | Trains vs EncounterAI |
| `--opponent-update-freq N` | Rotation cadence | 20 updates |
| `--use-lstm` | Recurrent policy | MLP policy |
| `--lstm-hidden H` | LSTM hidden size | 256 |
| `--lstm-layers L` | Stacked LSTM layers | 1 |

---

## 16. Design decisions that recur across features

A few principles show up in multiple places and are worth naming explicitly:

### "Clone, don't reference"
Whenever we make a copy of model weights (frozen opponent, checkpoint), we
use `.clone()` or `copy.deepcopy()`. In PyTorch, assignment does not copy
tensors — it creates a new reference to the same storage. Without cloning,
"updating the active model" silently updates the frozen opponent too.

### "Normalise, but only once"
Advantage normalisation (`A = (A - mean) / std`) appears in multiple places.
The rule is: normalise exactly once, and across the broadest possible scope.
For multi-env training, that means normalising across all envs after merging,
not per-env before merging (which would hide relative differences between
games).

### "Detect at runtime via duck typing"
Rather than a class hierarchy (`LSTMActorCritic extends ActorCritic`), we
detect model capabilities at runtime: `getattr(model, 'use_lstm', False)`.
This keeps Trainer and PPOTrainer generic — they work with any model that
exposes the right attributes and methods. Adding a new policy architecture
doesn't require changing Trainer.

### "Per-env buffers, merge for MLP, sequence for LSTM"
The rollout buffer architecture (one buffer per env) works for both training
modes. MLP: merge all buffers → shuffle → mini-batch update. LSTM: keep
buffers separate → process each as a sequence → no shuffling. The same
buffer class serves both paths.

---

## 17. MPS (Apple Metal) Compatibility Workarounds

Training on Apple Silicon (M1/M2/M3) via PyTorch's MPS backend exposes several
Metal-specific bugs. These workarounds are applied across the codebase so that
`--device mps` works out of the box.

### Problem 1: int64 tensor corruption

MPS can corrupt `int64` values during CPU→GPU transfer, producing garbage
indices (e.g. `704374636706` instead of `42`). When these indices hit
`nn.Embedding`, PyTorch crashes with `subRange.start` errors.

**Fix — int32 casting**: all `card_ids` tensors are cast to `int32` before
moving to MPS. This is done in every location that creates card ID tensors:

| File | Location |
|---|---|
| `training/trainer.py` | Rollout collection + bootstrap (2 sites) |
| `training/rollout.py` | `get_batches()` |
| `training/ppo.py` | `update_lstm()` |
| `training/self_play.py` | P2 frozen-policy inference |

```python
card_ids_t = torch.from_numpy(card_ids_arr).to(torch.int32).to(self.device)
```

**Fix — embedding index clamping**: as a safety net, both `ActorCritic` and
`LSTMActorCritic` clamp card IDs to valid range before the embedding lookup:

```python
card_ids = card_ids.clamp(0, self.embedding.num_embeddings - 1)
```

### Problem 2: nn.LSTM Metal gate dimension mismatch

MPS has a known bug in its `nn.LSTM` implementation that produces dimension
errors like `subRange.start (255) not less than length of dimension[2] (1)`.
This is an internal Metal shader issue, not a shape problem in user code.

**Fix — CPU routing**: `LSTMActorCritic._lstm_forward()` detects MPS and routes
all LSTM computation through CPU, moving inputs to CPU, running the LSTM, and
moving outputs back to MPS:

```python
def _lstm_forward(self, inp, h, c):
    orig_device = inp.device
    if orig_device.type == "mps":
        out, (h_new, c_new) = self.lstm(inp.cpu(), (h.cpu(), c.cpu()))
        return out.to(orig_device), h_new.to(orig_device), c_new.to(orig_device)
    out, (h_new, c_new) = self.lstm(inp, (h, c))
    return out, h_new, c_new
```

Since the bottleneck is HTTP I/O to the PHP engine (~50–200 ms per step), the
CPU detour for LSTM (microseconds) has zero measurable impact on throughput.

### Problem 3: LSTM device pinning across the lifecycle

The LSTM must stay on CPU while the rest of the model lives on MPS. Several
code paths move the entire model to MPS, undoing the CPU pin:

1. **Model creation** (`scripts/train.py`): `model.to(device)` moves everything
   to MPS. Immediately followed by `model.lstm = model.lstm.cpu()`.

2. **Checkpoint resume** (`scripts/train.py`): `model.load_state_dict()` puts
   all weights on the `map_location` device (MPS). Re-pin LSTM to CPU, then
   rebuild the optimizer so its param groups reference the correct devices.
   Adam buffers (`exp_avg`, `exp_avg_sq`) are also fixed up per-param:
   ```python
   for group in optimizer.param_groups:
       for p in group["params"]:
           for k, v in optimizer.state.get(p, {}).items():
               if isinstance(v, torch.Tensor):
                   state[k] = v.to(p.device)
   ```

3. **Trainer.train()** (`training/trainer.py`): `self.model.to(self.device)` at
   the top of the training loop re-pins after the move.

The LSTM must be pinned to CPU **before** optimizer creation so that the
optimizer's param groups are consistent with the actual param devices from the
start.

---

## 18. Tool: Deck Downloader (scripts/download_deck.py)

Interactive CLI to browse and download competitive decks from
[fabrary.net](https://fabrary.net/most-played-decks) for use in AI training.

### Motivation

Training against diverse decks requires a library of decks in Talishar's
`Assets/*.txt` format. Manually transcribing decks from competitive sites is
tedious and error-prone. This tool automates the process.

### How it works

1. **Scraping**: Uses Playwright (headless Chromium) to render fabrary.net's
   JS-rendered pages. Hero names are extracted from hero image URLs in deck
   listing entries (`content.fabrary.net/heroes/<slug>.webp`).

2. **Card resolution**: fabrary uses set-specific card codes (e.g. `WTR215`,
   `PEN319`). These are mapped to Talishar card IDs using
   `GeneratedCardDictionaries.php`:
   - **Direct lookup**: `GeneratedSetIDtoCardID` (4,600+ set-code mappings)
   - **Name fallback**: For newer sets not yet in the dictionary, falls back to
     `GeneratedCardName` reverse lookup with pitch-color heuristics

3. **Deck format**: Output matches Talishar's `Assets/*.txt` format:
   - Line 1: hero + equipment (space-separated)
   - Line 2: main deck cards (space-separated)
   - Lines 3+: sideboard cards (one per line)

### Usage

```bash
uv sync --dev && uv run playwright install chromium

# List most-played decks
uv run python -m scripts.download_deck

# Filter by hero
uv run python -m scripts.download_deck --hero "Dorinthea"
```

### Limitations

- Newer card sets (SKA, SDO, SFA) may not be fully mapped in
  `GeneratedCardDictionaries.php` — the tool warns about unmapped cards.
- Pitch-color heuristic for cross-set reprints can pick the wrong variant.
- Depends on fabrary.net's current page structure; site redesigns will break
  the scraper selectors.

---

## 19. Card Metadata Pipeline (scripts/generate_card_metadata.py)

### The problem: the model can't read card text

The observation vector encodes raw stats (cost, power, defense, pitch) but knows
nothing about what a card *does*.  Two cards with identical stats but completely
different abilities — one creates 3 Runechants, the other gains 3 life — look
the same to the model.  The model has to discover every card's strategic value
purely through gameplay experience, which is extremely sample-inefficient.

### Solution: a three-stage metadata pipeline

`generate_card_metadata.py` builds `card_metadata.json` with structured strategic
information for every card, computed *before* training starts.

#### Stage 1: PHP Parsing (always runs, no API key needed)

Parses `GeneratedCardDictionaries.php` to extract:
- **Stats**: name, type, subtype, class, talent, cost, power, defense, pitch
- **40+ keywords**: goAgain, dominate, overpower, intimidate, phantasm, crush,
  piercing, bloodDebt, battleworn, temper, bladebreak, guardwell, boost, charge,
  arcaneBarrier, spellvoid, ward, combo, reprise, ambush, channel, surge, etc.
- **Amount keywords**: arcaneBarrierAmount, spellvoidAmount, quellAmount, etc.

**Parser details**: uses two-pass regex matching — `_MATCH_QUOTED_RE` handles
values with commas (e.g., "Ira, Crimson Haze"), `_MATCH_UNQUOTED_RE` handles
numeric values.  Boolean keywords use `_parse_bool_match_block()` which looks
for `=> true` patterns.  The PHP function `GeneratedCardType` defaults to `"AA"`
for cards not explicitly listed — the parser mirrors this with
`types.get(cid, "AA")`.

**Output**: 4,645 cards with stats and keyword flags.

#### Stage 2: Deterministic Valuation (always runs)

Applies the **FaB rate system** where 3 = on-rate (a card that attacks for 3,
defends for 3, or pitches for 3 is baseline).  Computed fields:

| Field | Formula |
|---|---|
| `attack_value` | power + keyword adjustments (dominate +1, overpower +1, intimidate +1, phantasm −1, crush +1, piercing +1, bloodDebt −1) − cost/3 |
| `block_value` | defense (direct) |
| `pitch_value` | pitch (direct) |
| `best_use_value` | max(attack_value, block_value, pitch_value) |
| `rate_delta` | best_use_value − 3 (positive = above rate) |
| `block_willingness` | Equipment only: guardwell=9, battleworn=5, temper=4, bladebreak=2, no keyword=7. Adjusted by defense. |
| `arsenal_value` | 0-10, how useful this card is in arsenal. Resources/gems=0 (can only pitch/block from hand), AA scaled by attack_value, A=4-10 based on cost, I=3-10, AR=2-8, DR=1-6, equipment/weapons=0 (own zones). |

**Why arsenal_value?** In FaB, you can only pitch and block from hand, not from
arsenal.  A resource card stuck in arsenal is dead weight — it can't pitch, can't
block, and has no play effect.  This feature teaches the model to never arsenal
cards that can't be played out of it.

#### Stage 3: LLM Enrichment (optional, requires `--enrich` + API key)

Uses Claude API to score properties that can't be derived from stats or keywords.
Each card type gets a tailored prompt:

| Card type | LLM fields |
|---|---|
| Equipment (E) | `equipment_utility` 0-10 (activated abilities value) |
| Weapons (W) | `equipment_utility` 0-10 |
| Attack actions (AA) | `on_hit_value` 0-5, `has_on_hit`, `conditional_cost`, `token_generation` 0-3 |
| Non-attack actions (A) | `conditional_cost`, `token_generation` 0-3, `pump_value` 0-5, `disruption_value` 0-5, `effect_value` 0-10 |
| Reactions (AR, DR) | `conditional_cost`, `token_generation` 0-3, `pump_value` 0-5 |
| Instants (I) | `conditional_cost`, `token_generation` 0-3, `disruption_value` 0-5, `effect_value` 0-10 |

**`effect_value`** is critical for non-attack actions whose value comes from
effects, not stats.  Deadwood Dirge Red (power=0, defense=2, pitch=0) looks
below-rate deterministically (`best_use_value=2`), but creating 3 Runechants
is actually on-rate (`effect_value=3`).  When the LLM returns an `effect_value`
higher than the deterministic `best_use_value`, the metadata is updated.

#### Stage 4: Hero-Conditioned Synergy (optional, `--heroes`)

Scores how each hero's specific ability changes a card's value.  See §21 below.

### Usage

```bash
# Stage 1+2 only (no API key needed)
python3 -m talishar_ai.scripts.generate_card_metadata

# Full pipeline: Stage 1+2+3
python3 -m talishar_ai.scripts.generate_card_metadata --enrich

# Full pipeline + hero synergy for all heroes
python3 -m talishar_ai.scripts.generate_card_metadata --enrich --heroes all

# Auto-generate hero ability descriptions (one-time setup)
python3 -m talishar_ai.scripts.generate_card_metadata --generate-hero-abilities
```

---

## 20. Observation Space Expansion (features.py)

### Card feature vector: 14 → 18 floats

The original 14-float card vector encoded only raw stats and type.  It now
includes strategic metadata from `card_metadata.json`:

```
[0]  cost/10        [1]  power/10       [2]  defense/10     [3]  pitch/3
[4–11] type one-hot (AA, I, A, E, DR, R, T, C)
[12] counters/5     [13] tapped
[14] equipment_utility/10    — how valuable to preserve (LLM-scored)
[15] block_willingness/10    — should we block with this? (deterministic)
[16] arsenal_value/10        — is this worth arsenaling? (deterministic)
[17] hero_synergy/10         — hero-specific card value (LLM-scored)
```

### New aggregated features

Beyond the per-card vector expansion, several new observation blocks were added:

**Combat chain on-hit awareness (CC_DIM: 5 → 8)**:
- `active_on_hits`: whether the attacking card has an active on-hit trigger
- `on_hit_value/5`: how valuable the on-hit effect is (0-5 from metadata)
- `effective_attack_value/20`: raw damage + on-hit value (true cost of letting
  the attack through)

**Hand attack planning (HAND_PLAN_DIM = 7)**:
A greedy algorithm computes the optimal attack sequence from the current hand,
considering go-again chaining, pitch costs, and external go-again sources
(agility tokens in auras).  Gives the model a "hand report":
- `best_attack_line`: max damage from optimal attack sequence
- `attack_actions`: count of AA cards in hand
- `go_again_sources`: go-again from hand cards + aura tokens
- `surplus_cards`: cards left over after best attack plan (free to block)
- `surplus_block_value`: total defense of surplus cards
- `attack_reactions`: instant/reaction pumps in hand
- `can_multi_attack`: 1 if can play 2+ attacks

**Equipment & tempo (EQUIP_TEMPO_DIM = 7)**:
- `turn_number/30`: game progression
- Equipment counts (mine and opponent's)
- Equipment defense total
- `is_first_turn`: turn 0 flag (both players redraw to intellect)
- `is_defending`: in defense phase
- `hand_is_free`: turn 0 AND defending (hand cards are "free" to block with)

**On-hit prevention**: `can_prevent_on_hit` — 1 if total hand defense can fully
block the incoming attack when there's an active on-hit trigger.

### Total observation dimension: 535

```
27 card slots × 18 features    = 486 (zone cards)
7 global + 8 phase + 8 combat  =  23
1 stack + 4 opponent            =   5
7 hand_agg + 7 hand_plan        =  14
7 equip_tempo                   =   7
                                 ───
                          Total = 535
```

---

## 21. Hero-Conditioned Card Metadata

### The problem: card value depends on who's playing

Deadwood Dirge creates 3 Runechants — good for any Runeblade, but *exceptional*
for Vynnset whose hero ability converts Runechants into direct arcane damage.
Ash-generating cards are worthless outside Dromai.  Combo cards only matter for
Katsu.  A universal card valuation misses these synergies.

### Solution: hero_synergy scores

For each hero, the LLM evaluates every compatible card and assigns a
`hero_synergy` score (0-10) measuring how much the hero's specific ability
enhances that card compared to a generic hero of the same class.

**Class/talent filtering**: only cards playable by the hero are evaluated.
A card is compatible if its class matches any of the hero's classes (handling
multi-class heroes like Marlynn = PIRATE,RANGER) AND its talent matches or is
empty.  This prevents wasting API calls evaluating Pirate Ranger cards for Lexi.

**Deduplication**: heroes with identical `(ability_text, class, talent)` tuples
are evaluated only once.  Young/adult variants of the same hero that share an
ability get shared scores.  Different abilities (e.g., Ira Crimson Haze vs Ira
Scarlet Revenger) are evaluated separately.  The dedup key is the ability text
itself, not the hero name — this correctly handles cases like Arakni variants
that share a name but have completely different abilities.

**Hero abilities file**: `hero_abilities.json` maps hero card IDs to their
ability descriptions, intellect, and life.  Can be auto-populated with
`--generate-hero-abilities` (uses Claude to write ability text for all hero
cards not yet in the file).

**Observation encoding**: the encoder auto-detects the active hero from the
character zone in the game state (`myState.character[0].cardID`), then looks up
`hero_scores[hero_id].hero_synergy` for each card.  This flows into card feature
slot [17] (`hero_synergy/10`).

### Storage format

```json
"mask_of_momentum": {
  "name": "Mask of Momentum",
  "type": "E",
  "equipment_utility": 10,
  "hero_scores": {
    "ira_crimson_haze": {"hero_synergy": 10},
    "katsu": {"hero_synergy": 9},
    "fai": {"hero_synergy": 7}
  }
}
```

---

## 22. Equipment Preservation Reward Shaping (env.py)

### The problem

Early training showed the model was breaking equipment to block on turn 0 —
even premium equipment like Mask of Momentum.  This is a strategic mistake:
on turn 0 both players draw back to intellect (hand size), so hand cards used
to block are "free" (they'll be replaced), but equipment is permanent.

### Solution: utility-weighted equipment loss penalties

The reward function now tracks equipment changes between steps using card ID
diffing (via `collections.Counter` subtraction).  When equipment is lost, the
penalty is weighted by the card's `equipment_utility` from metadata:

```python
my_lost_ids = list((Counter(prev_my_equip_ids) - Counter(curr_my_ids)).elements())
my_lost_utility = sum(self._equip_utility(cid) for cid in my_lost_ids)
```

The utility weight normalises so that default-utility equipment (5) has weight
1.0, while premium equipment (utility 10, e.g., Mask of Momentum) has weight
2.0.  An early-game multiplier increases the penalty on turn 0 (3×) and turn 1
(2×), fading to 1× by turn 10.

Destroying the *opponent's* equipment is rewarded symmetrically, creating a
signal for aggressive equipment destruction strategies.

---

## 23. File map

```
talishar_ai/
  pyproject.toml              Python package + deps
  card_vocab.json             card_id → integer index (built once, checked in)
  card_metadata.json          Strategic card metadata (generated, 4,645 cards)
  hero_abilities.json         Hero ability descriptions for synergy scoring

  game_manager.py             HTTP client for the PHP engine
  features.py                 StateEncoder: JSON → float obs + card_ids (OBS_DIM=535)
  env.py                      TalisharEnv (Gymnasium) + equipment reward shaping
  parallel_env.py             ParallelEnvManager (ThreadPoolExecutor)

  card_vocab.py               CardVocab: parse PHP dicts → int lookup
  models/
    network.py                ActorCritic (MLP, use_embeddings opt.)
    lstm_network.py           LSTMActorCritic (LSTM, use_embeddings opt.)
  training/
    rollout.py                RolloutBuffer (GAE, episode_starts, LSTM init hidden)
    ppo.py                    PPOTrainer: update() [MLP] + update_lstm() [LSTM]
    trainer.py                Main collect→update loop (handles both models)
    self_play.py              SelfPlayEnv + SelfPlayManager
  evaluation/
    elo.py                    EloTracker (K=32, JSON persistence)
    game_stats.py             GameStatsCollector + GameStats dataclass
    logger.py                 EvalLogger (CSV + Elo JSON)
  scripts/
    train.py                  CLI: training entry-point
    evaluate.py               CLI: evaluate checkpoint vs EncounterAI
    head_to_head.py           CLI: checkpoint vs checkpoint Elo match
    download_deck.py          CLI: browse/download competitive decks from fabrary.net
    generate_card_metadata.py CLI: 4-stage metadata pipeline (parse → rate → LLM → heroes)

PHP (repo root / APIs/):
  GetAIState.php              Structured JSON state + legalMoves[] + activeOnHits
  SubmitAIAction.php          JSON POST action injection
  APIs/CreateTrainingGame.php Programmatic game creation
```

---

## 24. Quick-start checklist

```bash
# 1. Start the game engine
bash start.sh

# 2. Install Python deps
cd talishar_ai
uv sync

# 3. Verify game creation
curl -s -X POST http://localhost:8080/APIs/CreateTrainingGame.php \
  -H "Content-Type: application/json" \
  -d '{"p1_deck":"Ira","p2_deck":"Ira","p2_is_ai":true}' | python -m json.tool

# 4. Generate card metadata (deterministic only — no API key needed)
python3 -m talishar_ai.scripts.generate_card_metadata

# 4b. (Optional) Full metadata with LLM enrichment + hero synergy
export ANTHROPIC_API_KEY=sk-ant-...
python3 -m talishar_ai.scripts.generate_card_metadata --generate-hero-abilities
python3 -m talishar_ai.scripts.generate_card_metadata --enrich --heroes all

# 5. Quick smoke test (MLP, 1 env, no extras)
uv run python -m scripts.train \
  --total-steps 2000 --rollout-steps 128

# 6. Full-featured training run
uv run python -m scripts.train \
  --use-lstm --lstm-hidden 256 \
  --use-embeddings --emb-dim 32 \
  --n-envs 4 --self-play \
  --total-steps 2_000_000

# 7. Evaluate a checkpoint
uv run python -m scripts.evaluate \
  --checkpoint checkpoints/model_final.pt \
  --n-games 20 --log-dir eval_logs/
```
