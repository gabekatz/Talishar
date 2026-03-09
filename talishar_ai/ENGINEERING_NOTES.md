# Engineering Notes: Talishar AI — Design & Reasoning

This document explains every decision made when building the AI layer on top of
the Talishar game engine, written for an aspiring ML engineer who wants to
understand or replicate the work.

---

## 1. Problem framing

Flesh and Blood (FaB) is a two-player, imperfect-information card game.  Each
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
Observable MDP (POMDP)**.  We approximate it as an MDP by treating "opponent's
hand count" as the only hidden-state signal.  A future improvement would add
recurrent state (LSTM) to let the policy infer hand contents from history.

---

## 2. Why PPO?

Several RL algorithms could work here.  We chose **Proximal Policy
Optimisation (PPO-Clip)** for these reasons:

| Criterion | PPO | DQN | MCTS |
|---|---|---|---|
| Discrete, variable action space | ✓ (with masking) | ✓ | ✓ |
| No environment model needed | ✓ | ✓ | needs model |
| Stable training | ✓ (clip) | ✗ (TD instability) | ✓ |
| Sample efficiency | moderate | low | high but slow |
| Implementation complexity | low | medium | high |

PPO's **clip** objective `min(r·A, clip(r,1-ε,1+ε)·A)` prevents the policy
from taking steps so large that the new policy collapses.  This is the main
source of stability compared to vanilla policy gradients (REINFORCE).

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
  contain fewer cards.  The network learns "empty slot = zeros".

### Fixed vs. variable dimension

FaB hands can hold 1–7 cards; equipment slots are fixed.  We allocate the
*maximum* for each zone and pad.  This makes the observation space
`gymnasium.spaces.Box` with a fixed shape — required by most RL frameworks.

### What's *not* encoded (yet)

- Opponent's hand contents (hidden — only count is visible)
- Past chain links (history)
- Exact card identity (we use stats not card-name embeddings)

These are natural Phase 4 improvements.

---

## 4. Action masking (why it matters)

FaB has a **variable action space**: the number of legal moves changes every
step (0–50+).  Vanilla PPO would assign non-zero probability to every action
slot and occasionally sample an illegal move.  Two problems with that:

1. The environment would crash or return an error.
2. The policy wastes capacity learning not-to-do illegal things.

**Action masking** sets the logit of every illegal action to −∞ before
softmax, so their probability is exactly 0.  The gradient never flows through
masked slots.

Implementation in `network.py`:
```python
logits = logits.masked_fill(~action_mask, -1e9)
```

The mask is a `bool` tensor of shape `(MAX_ACTIONS,)`.  It's stored in the
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

Policy and value function share the first two layers.  This is standard in
PPO because:
- Both need similar game-state representations.
- Fewer total parameters → faster training.
- The critic regularises the shared layers, which helps policy generalisation.

### LayerNorm instead of BatchNorm

Batch Normalisation computes statistics over a mini-batch.  But during rollout
collection we process one step at a time (batch size = 1), so batch statistics
are meaningless.  Layer Normalisation computes statistics per-sample, which is
stable at any batch size.

### Orthogonal initialisation

`nn.init.orthogonal_` initialises weights so the matrix is orthogonal (rows
are orthonormal).  This is the recommended init for deep RL (from OpenAI's
baselines) because it preserves gradient magnitude through many layers.

---

## 6. GAE — Generalised Advantage Estimation (rollout.py)

The **advantage** A(s,a) = Q(s,a) − V(s) measures "how much better is this
action than average?".  We need it to be low-variance for stable PPO updates.

**GAE** interpolates between:
- λ=0: one-step TD error (low variance, high bias)
- λ=1: Monte Carlo return (high variance, zero bias)

```
δₜ  = rₜ + γ·V(sₜ₊₁)·(1−done) − V(sₜ)    # TD error
Aₜ  = δₜ + (γλ)·Aₜ₊₁·(1−done)             # recursive GAE
Rₜ  = Aₜ + V(sₜ)                           # discounted return
```

We then normalise advantages: `A ← (A − mean) / (std + ε)`.  This keeps the
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
potential function.  Our reward uses health totals as the potential, which is
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

**Why 512-step rollouts?**  Shorter than a full game (typically 50–150 turns),
which means more frequent updates and better exploration.  The bootstrap value
approximates the return for the unfinished episode.

**Why 4 epochs?**  Standard PPO uses 4–10 epochs per rollout.  More epochs
squeeze more gradient signal from each batch of data; too many epochs violates
the "on-policy" assumption (the data was collected under the *old* policy).

---

## 9. The PHP layer (CreateTrainingGame.php)

The game engine requires:
1. A lobby file `GameFile.txt` (player info, auth keys, deck links)
2. A game state file `gamestate.txt` (zones, hands, turn state)

`CreateTrainingGame.php` combines what the normal UI flow does in 3 steps
(CreateGame → JoinGame → Start) into a single API call.  It:

1. Calls `GetGameCounter()` for a unique numeric game ID
2. Copies deck files from `Assets/`
3. Generates auth keys with `hash('sha256', rand())`
4. Calls `WriteGameFile()` (from `MenuFiles/WriteGamefile.php`)
5. Writes `gamestate.txt` directly (mirrors `Start.php`)
6. Calls `initializePlayerState()` for each player (reads deck → writes zone data)
7. Caches the gamestate with `WriteGamestateCache()`
8. Runs `ParseGamestate.php` + `StartEffects.php` (hero start-of-game effects)
9. Returns `{gameName, p1AuthKey, p2AuthKey}`

Auth keys are how the engine authenticates actions.  The Python AI stores both
and uses `p1AuthKey` for all P1 moves.

---

## 10. How to extend this

### Better card representation
Replace the 14-float hand-crafted vector with a **card embedding table**:
- Build a `card_id → int_index` lookup from `CardDictionary.php`
- Use `nn.Embedding(num_cards, 64)` to learn a dense card representation
- Process the hand as a *set* with an attention mechanism (Transformer encoder)

### Self-play (Phase 4)
Set `p2_is_ai=False` in `CreateTrainingGame.php`.  Run two `TalisharEnv`
instances over the same `game_name` with alternating player IDs.  Periodically
copy a frozen checkpoint to be the "opponent" policy (population-based self-play).

### Recurrent policy (LSTM)
Replace the MLP trunk with an LSTM to track hidden state (opponent hand
contents, past chain links).  Pass `(h, c)` cell state across steps.

### Multi-hero generalisation
The current observation encodes card stats generically.  To generalise across
heroes, add a "hero one-hot" or hero embedding to the global scalars.

### Parallel environments
Wrap `TalisharEnv` in `gymnasium.vector.AsyncVectorEnv` with multiple worker
processes, each talking to its own game instance.  This multiplies sample
throughput without changing the algorithm.

---

## 11. File map

```
talishar_ai/
  requirements.txt          Python dependencies
  __init__.py
  game_manager.py           HTTP client for the PHP engine
  features.py               StateEncoder: JSON → numpy obs
  env.py                    TalisharEnv (Gymnasium)
  models/
    __init__.py
    network.py              ActorCritic (masked PPO policy)
  training/
    __init__.py
    rollout.py              RolloutBuffer (GAE)
    ppo.py                  PPOTrainer (clip + value + entropy)
    trainer.py              Main collect→update loop
  scripts/
    __init__.py
    train.py                CLI: python -m talishar_ai.scripts.train
    evaluate.py             CLI: python -m talishar_ai.scripts.evaluate
  ENGINEERING_NOTES.md      ← this file

PHP (repo root / APIs/):
  GetAIState.php            Structured JSON state + legalMoves[]
  SubmitAIAction.php        JSON POST action injection
  APIs/CreateTrainingGame.php  Programmatic game creation
```

---

## 12. Quick-start checklist

```bash
# 1. Start the game engine
bash start.sh

# 2. Install Python deps
cd talishar_ai
pip install -r requirements.txt

# 3. Verify game creation
curl -s -X POST http://localhost:8080/APIs/CreateTrainingGame.php \
  -H "Content-Type: application/json" \
  -d '{"p1_deck":"Ira","p2_deck":"Ira","p2_is_ai":true}' | python -m json.tool

# 4. Train (small test run)
python -m talishar_ai.scripts.train \
  --base-url http://localhost:8080 \
  --p1-deck Ira --p2-deck Ira \
  --total-steps 5000 --rollout-steps 128

# 5. Evaluate
python -m talishar_ai.scripts.evaluate \
  --checkpoint checkpoints/model_final.pt \
  --n-games 10
```
