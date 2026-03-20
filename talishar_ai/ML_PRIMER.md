# Machine Learning Primer: Core Principles via the Talishar AI

This document teaches fundamental ML/RL concepts by showing how each one
manifests in a real system — the Talishar FaB AI. Every section leads with
the principle, then grounds it in the concrete implementation.

---

## 1. The Markov Decision Process: Formalising "What Game Are We Playing?"

**Principle.** Before writing any ML code, you must define your problem as a
mathematical framework the algorithm can operate on. For sequential
decision-making, that framework is the **Markov Decision Process (MDP)**:

- **State** *s*: everything the agent can observe
- **Action** *a*: a choice the agent can make
- **Reward** *r*: a scalar signal after each action
- **Transition** *T(s,a) → s'*: the environment's response

The agent's goal is to learn a **policy** *π(a|s)* — a mapping from states to
actions — that maximises cumulative reward.

**In Talishar.** The PHP game engine *is* the environment. The state is the
JSON blob returned by `GetAIState.php` (health totals, cards in each zone,
combat chain, phase). The actions are the `legalMoves[]` array. The transition
is `SubmitAIAction.php` → engine logic → new game state. The reward is +1 for
winning, −1 for losing.

One game = one **episode**. The agent plays hundreds of thousands of episodes
to improve.

**Key insight.** The MDP framing forces you to answer three hard questions
up front: *what does the agent see?* (observation design), *what can it do?*
(action space), and *what does "good" mean?* (reward). Every other ML decision
flows from these answers.

---

## 2. Partial Observability: Working With Incomplete Information

**Principle.** A true MDP assumes the agent sees the full state. When it
can't — when there's hidden information — you have a **Partially Observable
MDP (POMDP)**. The agent must make decisions with incomplete knowledge, and
ideally *infer* hidden information from what it has observed over time.

**In Talishar.** The opponent's hand is hidden. You can't see what they
might block or react with. When they pass priority without playing a reaction,
that's evidence — maybe they don't have one. When they pitch a blue card,
that's information about their hand composition.

We address this at two levels:
1. **Rich observations** — encode enough visible information that the agent
   can make educated guesses (hand aggregates, equipment state, combat chain
   details).
2. **LSTM recurrent policy** — the network carries a hidden state across
   timesteps, giving it *memory*. It can learn temporal patterns: "opponent
   blocked aggressively early → probably low on cards now."

**Key insight.** Don't try to solve partial observability by cheating
(showing hidden info). Instead, give the model the tools to *infer* — rich
features for the present, and memory for the past.

---

## 3. Observation Design: The Feature Engineering–Learning Tradeoff

**Principle.** Raw data rarely goes directly into a neural network. You must
decide what to show the model (feature engineering) versus what to let it
discover on its own (representation learning). More features = faster learning
but more engineering effort. Fewer features = the model needs more data to
discover the same patterns.

The sweet spot is encoding things the model *can't easily learn from
experience* and letting it learn the rest.

**In Talishar.** The 535-float observation vector is a study in this tradeoff:

| What we encode explicitly | Why the model can't easily learn it |
|---|---|
| Card metadata (rate values, keywords) | Would take millions of games to discover that "dominate" is worth +1 power |
| Arsenal value (0 for resources/gems) | The *rule* "you can't pitch from arsenal" is invisible in gameplay — the model would just see arsenaled resources sit unused forever |
| Hero synergy scores | Vynnset's Runechant synergy requires understanding her hero ability text, which isn't in the observation |
| Equipment utility | Mask of Momentum's activated ability value can't be inferred from its stats alone |

Contrast with what we *don't* encode:
- "Should I attack or block?" → the model learns this from experience
- Sequencing of attacks → the model discovers optimal orderings through play
- Bluffing patterns → the LSTM learns these from temporal patterns

**Key insight.** Encode domain knowledge that would take too long to learn
from scratch. Let the network learn everything else. The card metadata
pipeline (`generate_card_metadata.py`) exists precisely because some FaB
knowledge — the rate system, keyword synergies, hero abilities — would take
millions of training episodes to rediscover.

---

## 4. Normalisation: Speaking the Network's Language

**Principle.** Neural networks learn fastest when all inputs are on a similar
scale, typically [0, 1] or [-1, 1]. If one feature ranges from 0–1000 and
another from 0–1, the network must learn very different weight scales for
each, slowing convergence.

**In Talishar.** Every feature is normalised by its natural maximum:

```python
vec[0] = min(cost, 10) / 10.0     # cost:    0-10 → 0-1
vec[1] = min(power, 10) / 10.0    # power:   0-10 → 0-1
vec[3] = min(pitch, 3) / 3.0      # pitch:   0-3  → 0-1
```

Categorical variables (card type) use **one-hot encoding** — 8 binary slots
where exactly one is 1.0 and the rest are 0.0. This prevents the network from
inferring false ordinal relationships (type 3 > type 2).

Aggregated features use their practical maximums:
```python
total_hand_power / 50     # a 7-card hand rarely exceeds 50 power
total_hand_defense / 35   # 7 × 5 defense = 35
turn_number / 30          # games rarely go past 30 turns
```

**Key insight.** Normalisation isn't just cosmetic — it directly affects
learning speed. Unnormalised features create steep, narrow loss landscapes
that gradient descent navigates poorly.

---

## 5. Action Masking: Constraining the Search Space

**Principle.** When the set of legal actions changes at each timestep, the
model must never select an illegal action. Naively training a fixed-size
output and hoping the network learns legality wastes enormous capacity.
**Action masking** forces illegal action probabilities to exactly zero by
setting their logits to −∞ before the softmax.

**In Talishar.** FaB's legal actions range from 1 (forced pass) to 50+
(complex hand decisions). The action space is fixed at MAX_ACTIONS=64, but a
boolean mask tells the network which slots are legal:

```python
logits = logits.masked_fill(~action_mask, -1e9)
probs = softmax(logits)  # illegal actions get probability ≈ 0
```

This has two effects:
1. The model never wastes a training step on "don't play that illegally."
2. Exploration is concentrated on *meaningful* choices, dramatically
   improving sample efficiency.

**Key insight.** Action masking is one of the highest-leverage techniques in
game AI. It transforms an intractable action space (64 slots, most illegal)
into a focused decision among 2–10 real options. Without it, PPO would need
orders of magnitude more data to learn basic legality.

---

## 6. Policy Gradient Methods: Learning by Trial and Error

**Principle.** In **policy gradient** methods, the model directly outputs a
probability distribution over actions. It tries actions, observes rewards,
and adjusts: increase probability of actions that led to good outcomes,
decrease for bad ones.

The fundamental update is:
```
∇θ J(θ) = E[∇θ log π(a|s) · A(s,a)]
```

Read this as: "nudge the policy parameters θ to make action *a* more likely
in state *s*, proportionally to how much better *a* was than average
(the advantage *A*)."

**In Talishar.** The actor head outputs logits for 64 action slots. After
masking and softmax, we sample an action. After collecting a rollout (512
steps), we compute advantages and update:

- Actions that led to winning → advantage is positive → probabilities increase
- Actions that led to losing → advantage is negative → probabilities decrease
- Actions that were "average" → advantage ≈ 0 → probabilities barely change

**Key insight.** Policy gradients learn from *relative* quality, not absolute.
The advantage function is the key: it asks "was this action better or worse
than what I'd normally do here?" This relative comparison is what makes
learning possible even with sparse, delayed rewards.

---

## 7. PPO-Clip: Stable Policy Updates

**Principle.** Vanilla policy gradients are unstable — a single bad update
can collapse the policy. **PPO-Clip** fixes this by limiting how much the
policy can change in one update:

```
L = min(r · A, clip(r, 1-ε, 1+ε) · A)
```

Where `r = π_new(a|s) / π_old(a|s)` is the probability ratio. If `r` drifts
outside `[1-ε, 1+ε]` (typically ε=0.2), the clipped version kicks in and
the gradient stops pushing further.

**In Talishar.** Without clipping, we observed training collapses where the
model would suddenly become deterministic (one action gets 99% probability)
and never recover. PPO-Clip prevents this by saying: "even if this action
looks amazing, don't make it more than 20% more likely in one update."

This is especially important in FaB because:
- Games are long (50–150 turns) → many decisions per episode
- Rewards are sparse (only at game end) → advantage estimates are noisy
- Action spaces vary wildly → the policy operates very differently across
  game phases

**Key insight.** PPO-Clip is essentially a trust region in probability space.
It says "I trust my advantage estimates enough to adjust slightly, but not
enough to make drastic changes." This conservatism is what makes PPO the
workhorse of modern RL.

---

## 8. The Actor-Critic Architecture: Two Heads, One Body

**Principle.** An **actor-critic** model combines two functions:
- **Actor** (policy): "what should I do?" → probability distribution over actions
- **Critic** (value): "how good is this state?" → scalar V(s)

They share early layers (the "trunk") because both need to understand the
game state. The critic's V(s) is used to compute advantages for the actor's
update, and the actor's exploration generates the data the critic learns from.

**In Talishar.**

```
obs (535) → Linear(535, 256) → LayerNorm → ReLU
          → Linear(256, 256) → LayerNorm → ReLU
          ↓                            ↓
  actor: Linear(256, 64)      critic: Linear(256, 1)
  → mask → Categorical dist   → V(s)
```

The shared trunk learns a compressed representation of "what's happening in
this game." The actor head translates that into action preferences. The
critic head translates it into a value estimate ("am I winning or losing?").

Design choices that matter:
- **LayerNorm** (not BatchNorm): during rollout collection, we process one
  observation at a time. BatchNorm needs batch statistics, which are
  meaningless at batch size 1. LayerNorm computes per-sample.
- **Orthogonal initialisation**: preserves gradient magnitude through layers,
  critical for deep RL where training signals are already noisy.

**Key insight.** The critic is the teacher; the actor is the student. The
critic learns to predict "how good is this position?" from game outcomes.
The actor uses those predictions to evaluate "was my action better or worse
than expected?" Neither can learn well without the other.

---

## 9. Generalised Advantage Estimation (GAE): Balancing Bias and Variance

**Principle.** The advantage A(s,a) = Q(s,a) − V(s) has a bias-variance
tradeoff:

- **Monte Carlo** (use actual returns): zero bias, high variance — every
  random event in the rest of the game affects the estimate
- **One-step TD** (use V(s') as a bootstrap): low variance, but biased by
  the critic's errors

**GAE** with parameter λ smoothly interpolates:

```
δₜ = rₜ + γ·V(sₜ₊₁) − V(sₜ)         # one-step TD error
Aₜ = δₜ + (γλ)·δₜ₊₁ + (γλ)²·δₜ₊₂ + ...  # exponentially-weighted sum
```

λ=0 gives pure TD (low variance, high bias). λ=1 gives pure Monte Carlo
(high variance, zero bias). λ=0.95 is the standard balance.

**In Talishar.** GAE is computed per-environment over contiguous trajectories
in `rollout.py`. With λ=0.95 and γ=0.99:

- The TD error δₜ gives immediate feedback: "this action caused me to take
  5 damage — that's bad relative to what the critic expected."
- The exponential weighting means events 20+ steps in the future barely
  affect the current advantage. This is critical in FaB where a game might
  last 100+ turns — without GAE, a turn-1 block decision would be evaluated
  based on the entire game's outcome.

**Advantage normalisation** (`A ← (A − mean) / std`) is applied once, across
all environments. This keeps gradient magnitudes stable regardless of whether
games are high-scoring or low-scoring.

**Key insight.** GAE lets you say "recent consequences matter more than
distant ones" without hard-coding a horizon. It's the temporal equivalent
of attention — focusing the learning signal on the most informative events.

---

## 10. Reward Shaping: Teaching Without Cheating

**Principle.** Sparse rewards (+1 for win, −1 for loss) contain the right
signal but are hard to learn from — the agent plays hundreds of actions
before getting any feedback. **Reward shaping** adds dense intermediate
signals that guide learning without changing the optimal policy.

The theoretical guarantee (Ng et al. 1999): if your shaping reward has the
form `γ·Φ(s') − Φ(s)` for some potential function Φ, the optimal policy is
unchanged. Shaping that violates this can introduce sub-optimal biases.

**In Talishar.** Three layers of reward shaping:

**1. Health delta (approximately potential-based):**
```python
shaped = 0.01 × (opponent_health_drop − my_health_drop)
```
Φ(s) = health difference → this is approximately potential-based. The agent
gets a small positive signal for dealing damage and negative for taking it,
every single step.

**2. Equipment preservation (utility-weighted):**
```python
penalty = −0.005 × utility_weight × early_game_multiplier
```
Losing Mask of Momentum (utility=10, weight=2.0) on turn 0 (multiplier=3×)
is penalised 6× more than losing generic equipment late-game. This encodes
the FaB insight that equipment is permanent but hand cards refresh every turn.

**3. Equipment destruction (symmetric reward):**
Destroying the opponent's equipment is rewarded, teaching aggressive equipment
targeting in matchups where it matters (e.g., assassins vs. temper gear).

**Key insight.** Reward shaping is where domain knowledge enters the learning
loop. The FaB rate system, equipment durability rules, and turn-0 mechanics
are all strategic knowledge that experienced players know instantly but an
agent would need millions of games to discover. Shaping lets you teach these
lessons explicitly while still letting the agent learn everything else
through experience.

---

## 11. Embeddings: Learning Representations for Discrete Objects

**Principle.** Neural networks work with continuous vectors, not discrete
labels. When you have a large set of discrete objects (words, cards, users),
an **embedding table** maps each object to a learned vector. The network
discovers what each vector should represent during training.

This is the same technique behind Word2Vec and modern language models —
except instead of learning that "king" − "man" + "woman" ≈ "queen", we're
learning that cards with similar strategic roles get similar vectors.

**In Talishar.** With 4,645 unique cards, many share identical stats but
have completely different abilities. The embedding table maps each card ID
to a 32-dimensional vector:

```python
nn.Embedding(vocab_size=4645, embedding_dim=32, padding_idx=0)
```

Key design decisions:
- **Concatenate, don't replace**: embeddings are appended to the stat features,
  not substituted. Stats provide reliable structure; embeddings add identity.
- **`padding_idx=0`**: empty card slots (no card present) always produce a zero
  vector, with no gradient. The network learns that all-zeros = "nothing here."
- **Small initialisation (std=0.01)**: fresh, untrained embeddings shouldn't
  overwhelm the already-meaningful stat features. They start near-zero and
  grow as the model learns which card identities matter.

**Key insight.** Embeddings solve the "two cards with same stats but different
abilities" problem. Timesnap Potion and Surging Strike might look identical
in raw stats, but their embeddings diverge as the model learns they play
completely different strategic roles.

---

## 12. Recurrent Networks: Learning to Remember

**Principle.** Standard feedforward networks (MLPs) are memoryless — each
decision is based only on the current observation. An **LSTM (Long Short-Term
Memory)** maintains a hidden state that persists across timesteps, acting as
a learned memory. It decides what information to store, what to forget, and
what to output at each step.

**In Talishar.** The LSTM tracks information across turns:
- When the opponent pitches a blue card, that's evidence of hand composition
- When they pass without reacting, that suggests a weak hand
- When they aggressively block, they may be running low on attack resources

The hidden state `(h, c)` carries this temporal context. The architecture:

```
obs → pre-LSTM encoder → LSTM → actor head + critic head
                           ↑
                     (h, c) from previous step
```

**The LSTM complicates training significantly:**
- **Rollout collection**: track and update (h, c) per environment; reset to
  zeros when an episode ends (new game starts with blank memory)
- **PPO update**: can't shuffle transitions randomly (step 200 needs the
  hidden state from steps 0–199). Process each environment's rollout as an
  ordered sequence
- **Episode boundaries**: a 512-step rollout may contain 3–5 complete games.
  Hidden state must reset at each game boundary

This is **Backpropagation Through Time (BPTT)** — gradients flow backwards
through the sequence, teaching the LSTM what to remember.

**Key insight.** Memory isn't free. The LSTM architecture doubles the
implementation complexity (per-env hidden states, sequential processing,
episode boundary handling). The decision to use it should be driven by
evidence: does the model perform measurably better with memory? In FaB,
the hidden hand makes temporal reasoning genuinely valuable, so the
complexity is justified.

---

## 13. Parallel Environments: Hiding Latency

**Principle.** The training loop alternates between data collection (acting
in the environment) and model updates (gradient descent). If the environment
is slow, the GPU sits idle waiting. Running multiple environments in parallel
lets you collect data from all of them simultaneously.

This is distinct from data parallelism (multiple GPUs) — it's *environment*
parallelism.

**In Talishar.** The bottleneck is HTTP round-trips to the PHP game engine
(~50–200ms each). With one environment:

```
[HTTP 150ms]──[inference 1ms]──[HTTP 150ms]──[inference 1ms]──...
```

GPU utilisation: ~0.7%. With 4 parallel environments using a thread pool:

```
env 0: [HTTP 150ms]─────────────────
env 1:  [HTTP 150ms]────────────────
env 2:   [HTTP 150ms]───────────────
env 3:    [HTTP 150ms]──────────────
         all complete → batched inference → next step
```

**Threads, not processes.** The bottleneck is I/O (waiting for HTTP), not
CPU. Python's GIL doesn't block I/O-bound threads. Each thread gets its own
HTTP session to prevent connection pool race conditions.

**Per-environment buffers.** Each environment's trajectory must stay separate
because **GAE requires contiguous trajectories**. Mixing steps from different
games would make the "next value" calculation meaningless. Advantages are
normalised *after* merging all buffers — per-env normalisation would destroy
the relative scale between easy and hard games.

**Key insight.** Profile before parallelising. Our bottleneck was I/O, not
compute, so threads (not processes, not GPUs) were the right solution.
4 environments gave ~4× throughput at minimal code complexity.

---

## 14. Self-Play: An Auto-Curriculum

**Principle.** Training against a fixed opponent leads to **overfitting to
the opponent** — the agent learns to exploit that specific opponent's
weaknesses rather than learning the game. **Self-play** solves this by
making the agent train against a frozen copy of itself. As the agent
improves, so does the opponent, creating a naturally escalating curriculum.

This is the technique behind AlphaGo, OpenAI Five, and most modern game AI.

**In Talishar.**

```
P1 (active policy):   learning, receives gradients
P2 (frozen opponent): deepcopy of P1's weights, no gradients
```

The frozen opponent is updated periodically (every ~20 PPO updates). Key
implementation details:

- **`copy.deepcopy(model)`** creates a fully independent copy. In PyTorch,
  simple assignment creates a reference — updating the active model would
  silently update the "frozen" opponent. `.clone()` on every tensor prevents
  this.

- **Rotation frequency matters.** Too fast → the opponent is a moving target,
  training is unstable. Too slow → the agent overfits to an outdated version
  of itself. Every 20–50 updates is standard.

- **P2 driving loop**: after each P1 action, the self-play wrapper checks if
  it's P2's turn and drives the frozen model's responses automatically.

**Key insight.** Self-play converts a two-player game into a single-agent
problem with an auto-scaling difficulty curve. The "curriculum" emerges
naturally — you never need to hand-design opponent difficulty levels.

---

## 15. Evaluation: Elo Rating and Why Win Rate Isn't Enough

**Principle.** Training optimises a loss function, but you need a separate,
unbiased measurement of agent quality. **Win rate** against a fixed opponent
has a ceiling — once you win 90%+ of games, additional improvement is noise.
**Elo rating** quantifies skill as a continuous number that keeps
differentiating even among strong players.

```
Expected win probability = 1 / (1 + 10^((rating_B - rating_A) / 400))
Rating update = old + K × (actual_result − expected_result)
```

**In Talishar.** We evaluate checkpoints by playing games against a fixed
reference opponent (EncounterAI, held at constant Elo). This lets us:

1. **Track improvement over training**: Elo should trend upward
2. **Compare checkpoints directly**: head-to-head matches between any two
   saved models
3. **Diagnose strategy**: `GameStatsCollector` tracks damage dealt, turns
   played, deck cards remaining — if the model wins by decking out the
   opponent rather than reducing health, that's a useful diagnostic

**Key insight.** Evaluation should measure *what you care about*, not just
what's easy to measure. Win rate is easy but saturates. Elo is harder to
implement but gives continuous signal. Game statistics tell you *how* the
agent is winning, which matters more than *whether* it wins.

---

## 16. Domain Knowledge Injection: The Metadata Pipeline

**Principle.** In ML, there's a constant tension between learning from data
and encoding prior knowledge. Pure end-to-end learning is elegant but
sample-inefficient. Encoding everything as features is fast but brittle and
doesn't generalise. The best systems do both: **encode what's hard to learn,
let the network learn the rest.**

**In Talishar.** The card metadata pipeline is a four-stage system that
injects FaB domain knowledge into the observation space:

**Stage 1 — Parse (deterministic).** Extract stats and 40+ keywords from
PHP source code. This is data the model could theoretically learn from
gameplay, but why wait? Keywords like `dominate` and `overpower` have
well-defined mechanical effects.

**Stage 2 — Rate valuation (deterministic).** Apply the FaB rate system
(3 = on-rate). A card that attacks for 4 with go-again is above-rate. A card
that costs 3 and attacks for 2 is below-rate. This encodes competitive
player knowledge about card quality.

**Stage 3 — LLM enrichment (requires API).** For properties that can't be
computed from stats — on-hit effects, token generation, effect values — use
an LLM to read card text and score it. Deadwood Dirge Red has power=0 but
creates 3 Runechants = effect_value 3 = on-rate. No amount of stat parsing
can capture this.

**Stage 4 — Hero synergy (requires API).** Same card, different hero =
different value. Runechant generation is good for Viserai, exceptional for
Vynnset. Score every card's synergy with every hero's ability. Deduplicate
by ability text so young/adult variants with identical abilities aren't
evaluated twice.

**Key insight.** The pipeline separates *what can always be computed* (stages
1–2) from *what requires understanding* (stages 3–4). Stages 1–2 run without
any API key and cover all 4,645 cards. Stages 3–4 add strategic depth but
are optional and incremental. This layered design means you can start
training immediately and add richer features later.

---

## 17. Representation Choices: Fixed-Size Vectors vs. Structured Input

**Principle.** The choice of input representation constrains what
architectures you can use:

| Representation | Compatible architectures | Complexity |
|---|---|---|
| Fixed-size flat vector | MLP, LSTM | Low |
| Variable-length sequences | Transformer, RNN | Medium |
| Graphs | GNN | High |
| Images | CNN | Medium |

Richer representations (graphs, sequences) can capture more structure but
add implementation complexity and often don't help until you have enough data
to exploit that structure.

**In Talishar.** We chose a fixed-size flat vector (535 floats):

- **Zero-padding** handles variable-size zones (hand holds 1–7 cards; we
  allocate 7 slots and pad with zeros)
- **Fixed slot assignment** means the model always knows "slot 0–6 = hand,
  slot 7–8 = arsenal, slot 9–13 = equipment" — it doesn't need to learn
  spatial structure
- Compatible with `gymnasium.spaces.Box`, which most RL frameworks require

The cost: position in a zone is somewhat arbitrary (card order in hand
shouldn't matter, but the network might learn positional biases). The benefit:
simplicity, speed, and compatibility with both MLP and LSTM architectures.

**Key insight.** Start with the simplest representation that works. A flat
vector with good features outperforms a sophisticated architecture with poor
features. You can always upgrade the representation later (to a Transformer
over card sequences, for example) — but get the features right first.

---

## 18. Training Stability: The Things That Go Wrong

**Principle.** RL training is notoriously unstable. Unlike supervised learning
where the data distribution is fixed, in RL the data distribution changes
as the policy changes (non-stationarity). Several techniques prevent
collapse:

**Gradient clipping.** Cap the gradient norm to prevent a single bad batch
from destroying the weights:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
```

**Entropy bonus.** Add the policy's entropy to the loss function, rewarding
exploration and preventing premature convergence to a deterministic policy:
```
L = L_policy + c₁·L_value − c₂·H(π)
```
In FaB, where optimal play requires mixing strategies (sometimes blocking,
sometimes taking damage to preserve resources), maintaining entropy is critical.

**Learning rate.** Too high → training oscillates or collapses. Too low →
training makes no progress. Standard PPO uses 3e-4, which works as a starting
point but may need tuning.

**Value function clipping.** The critic's loss is also clipped to prevent
the value estimates from swinging wildly, which would destabilise advantage
computation and cascade into unstable policy updates.

**In Talishar.** We encountered specific stability issues:

- **Deterministic collapse**: without entropy bonus, the model would discover
  one "good" action (e.g., always pass) and converge to 100% probability,
  never exploring alternatives. The entropy bonus (c₂=0.01) prevents this.
- **Advantage explosion**: without normalisation, advantages could reach
  ±100 in games with large health swings, causing enormous gradient updates.
  Normalising to zero-mean unit-variance solved this.

**Key insight.** RL training fails silently. The loss goes down, but the
policy gets worse. Always track game-level metrics (win rate, Elo, game
length) alongside training loss. If Elo stagnates while loss improves,
something is wrong.

---

## 19. The Explore-Exploit Tradeoff

**Principle.** The agent must balance **exploration** (trying new actions
to discover better strategies) with **exploitation** (using what it already
knows works). Too much exploitation → stuck in local optima. Too much
exploration → never converges.

In policy gradient methods, exploration comes from the stochasticity of the
policy itself — actions are *sampled* from the probability distribution,
not taken greedily. The entropy bonus (§18) keeps this distribution from
collapsing.

**In Talishar.** FaB has deep explore-exploit tensions:

- **Turn 0 blocking**: should the model explore different blocking strategies
  or exploit the equipment-preservation approach it's learned? Reward shaping
  guides early, but the entropy bonus ensures alternatives are tried.
- **Arsenal decisions**: should it explore arsenaling different card types or
  always arsenal the highest attack? The arsenal_value feature provides
  guidance, but the model still needs to discover *when* each strategy is
  optimal.
- **Self-play as exploration**: when the opponent (frozen copy) changes, the
  agent is forced to explore new counter-strategies. The rotating opponent
  prevents premature exploitation of a fixed opponent's weaknesses.

**Key insight.** In games, the explore-exploit tradeoff is never "solved" —
it's managed. Action masking focuses exploration on legal moves. Entropy
bonus prevents collapse. Self-play prevents overfitting. Reward shaping
accelerates early exploration. These mechanisms work together to keep
exploration productive without making it random.

---

## 20. Platform-Specific Challenges: When Hardware Breaks Your Assumptions

**Principle.** ML code that works perfectly on one platform may fail
silently on another. GPU backends (CUDA, MPS, ROCm) implement operations
differently, and bugs in these implementations can produce incorrect results
with no error message.

**In Talishar.** Training on Apple Silicon (MPS backend) exposed three bugs:

1. **int64 tensor corruption**: MPS silently corrupts int64 values during
   CPU→GPU transfer. A card index of 42 becomes 704374636706. The fix:
   cast all card_ids to int32 before GPU transfer.

2. **LSTM gate dimension mismatch**: MPS's LSTM implementation has a known
   Metal shader bug. The fix: route LSTM computation through CPU, moving
   inputs to CPU and outputs back to MPS. Since the real bottleneck is
   HTTP I/O (150ms), the CPU detour (microseconds) is invisible.

3. **Device pinning across lifecycle**: `model.to(device)` moves everything
   to MPS, undoing the LSTM CPU pin. Must re-pin after every `model.to()`
   call, including checkpoint loading and training loop initialisation.

**Key insight.** Always validate on your target hardware early. Silent data
corruption is worse than crashes — your model trains on garbage data and
you don't know until you evaluate. Embedding index clamping (a safety net)
and int32 casting (the fix) are both needed: defense in depth.

---

## 21. Putting It All Together: The Training Pipeline

Here's how all these principles connect in a single training step:

```
1. Parallel Envs [§13]         → 4 games run simultaneously
2. Observation Design [§3,4]   → PHP state → 535-float vector (normalised)
3. Card Embeddings [§11]       → card IDs → 32-dim learned vectors
4. LSTM Memory [§12]           → hidden state carries temporal context
5. Actor-Critic [§8]           → actor picks action, critic estimates value
6. Action Masking [§5]         → illegal moves set to probability 0
7. Environment Step            → PHP engine processes action, returns new state
8. Reward Shaping [§10]        → health delta + equipment preservation signal
9. Rollout Buffer              → store (s, a, r, V, logπ) for 512 steps
10. GAE [§9]                   → compute advantages with bias-variance tradeoff
11. PPO Update [§7]            → clipped policy gradient update, 4 epochs
12. Stability Measures [§18]   → gradient clipping, entropy bonus, value clipping
13. Self-Play Rotation [§14]   → update frozen opponent every 20 updates
14. Evaluation [§15]           → checkpoint → Elo rating vs reference opponent
```

Each principle exists because of a specific problem:
- The model can't read card text → metadata pipeline [§16]
- The opponent's hand is hidden → LSTM [§12]
- Training against one opponent overfits → self-play [§14]
- Win/loss-only reward is too sparse → reward shaping [§10]
- Legal moves change every step → action masking [§5]
- HTTP latency wastes compute → parallel environments [§13]
- Large policy updates collapse training → PPO clip [§7]

None of these are academic exercises. Each solves a real problem that blocked
progress during development. That's the core lesson: ML principles aren't
abstract theory — they're solutions to concrete engineering problems.
