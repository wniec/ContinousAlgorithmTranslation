# Continuous Algorithm Translation

Studying whether a neural network can **translate the internal state of one
metaheuristic into another** — turning a PSO swarm (positions + **velocities** +
personal bests) into an equivalent CMA-ES search distribution (mean +
**covariance** + step-size + evolution paths), and back.

When a dynamic-algorithm-selection controller switches optimizers mid-run, the
hand-off is normally *lossy*: the shared population is carried but every
algorithm-specific structure (velocities, covariance) is discarded. This project
**learns** that hand-off instead.

## Idea

Each optimizer exposes a warm-start state via `get_data()` / `set_data()`. That
state splits into:

- **shared** fields — positions, values, best-so-far — the fixed "circumstances"
  of a switch, carried verbatim and never translated;
- **specific** fields — the algorithm's own machinery, which a bidirectional
  translator learns to map between algorithms.

Three optimizers are implemented (`cat/optimizers/`): **PSO** (per-particle
velocities + personal bests), **CMA-ES** (mean + covariance + step-size +
evolution paths), and **MadDE** (Biswas et al., CEC 2021 — a self-adaptive DE
whose translated field is its **external archive**, a variable-size *point-set*).
The archive is a distinct `POINT_SET` field type: the encoder pools it
(set-equivariant), and the decoder regenerates it as `round(2.3·N)` points
seeded from the carried population. Any pair can be translated
(`PSO`/`CMAES`/`MADDE`).

A `TranslatorPair` holds an encoder + decoder head per algorithm. Translating
`A -> B` reuses A's population and only decodes B's specific fields. It is trained
with three objectives (`cat/losses.py`):

1. **Cycle-consistency** — with the population fixed, `A -> B -> A` must perturb
   A's own parameters as little as possible.
2. **Reconstruction** — each algorithm's encoder/decoder is an autoencoder
   identity (anchors the shared latent).
3. **Utility proxy** — the translated state must be a good warm-start for the
   target (CMA-ES: elites likely under the decoded distribution; PSO: velocities
   point down-hill), so the cycle can't be solved by a useless near-identity map.

The network is **set-equivariant**: permutation-invariant over particles
(DeepSets pooling) and permutation-equivariant over coordinate axes (attention
over per-dimension tokens with a covariance-derived bias). One model spans all
problem dimensions `D` and population sizes `N`. The covariance is decoded as
`L Lᵀ + diag(d)`, PSD by construction.

## Install

Python 3.11, dependencies via [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Logging (Weights & Biases)

Both trainers can log metrics to W&B. Copy the example env file and fill in your
credentials (the file is git-ignored and loaded automatically):

```bash
cp .env.example .env      # then edit WANDB_API_KEY / WANDB_PROJECT / WANDB_ENTITY
```

Then pass `--wandb` (optionally `--wandb-project` / `--wandb-run-name`) to
`train.py` or `train_rl.py`. Per-epoch (supervised) or per-update (PPO) metrics
are logged. Set `WANDB_MODE=offline` to sync later, or `disabled` to turn it off.
Without `--wandb` nothing W&B-related is touched.

## Train

```bash
python train.py PSO CMAES -d 2 3 5 --n-individuals 12 \
    --fe-multiplier 2000 --n-switches 8 --epochs 30 --seed 42
```

Runs BBOB problems while alternating between the two named algorithms, snapshots
their internal states, and trains the translator. Saves `models/PSO_CMAES.pt`.

Each episode is split into `--n-switches` segments whose boundaries are **not**
fixed checkpoints: they are sampled **log-uniformly over the FE budget**,
independently per problem. Because optimization progress is roughly linear in
`log(FE)`, this places more switches early (where the state changes fast) and
fewer late — and gives varied, randomized switch positions across the dataset.

Key flags: `-d/--dims`, `--split {easy,all}`, `--schedule {alternate,random}`,
`--n-switches` (segments per episode), `--n-individuals` (fix a shared
population size — recommended when pairing a swarm with an ES), and the loss
weights `--w-cycle/--w-recon/--w-utility`.

## Train with reinforcement learning (PPO)

The supervised utility term is only a *proxy* for what we actually want — that
the translated state makes the target optimizer optimize well, which is
non-differentiable. The RL trainer optimizes that true objective directly with
PPO over a Gymnasium environment, reusing the **same** set-equivariant network as
the policy:

```bash
python train_rl.py PSO CMAES -d 2 3 5 --n-individuals 12 \
    --n-switches 6 --fe-multiplier 2000 --updates 100 --rollout-steps 1024
```

- **Environment** (`cat/rl/env.py`, a `gymnasium.Env`): one episode is a BBOB run
  with log-sampled switch points; the observation is the source optimizer's
  state, the action is the target optimizer's state (the translation).
- **Reward** (`--reward-mode`, all scaled by the **initial gap to the global
  optimum** — warmup best-so-far minus the BBOB optimum):
  - `noswitch` **(default)** — the translation should make the switch
    *invisible*: each segment is run twice, once with the switch (target
    optimizer from the translated state) and once **without** (the source
    optimizer simply continuing), and the reward is the *similarity* of the two,
    `−|Δ best| / gap` (0 = the switched trajectory matches no-switch exactly).
  - `absolute` — log-scaled best-so-far improvement over the segment.
  - `relative` — log-scaled improvement over the lossy default hand-off.

  `noswitch` and `relative` run an extra counterfactual optimizer per step
  (~2× cost).
- **Action**: a Gaussian policy over the decoded target fields (covariance
  sampled in factor space → stays PSD).
- **Critic**: its own encoder (decoupled from the actor) with PPO value-clipping
  and a Huber loss, so a value-function spike can't swamp the policy gradient. It
  additionally consumes a side-input of the 2 progress scalars (running-normalized).
  This feeds **only the critic** (the actor/translator stays a function of the
  optimizer state alone, so `evaluate.py` and the supervised path are unaffected).
- **Normalization** (on by default): rewards are scaled by the running std of the
  discounted return, and the scalar context observation is standardized by its
  running mean/std — both persisted across updates so value targets and
  advantages stay at ~unit scale. The structured optimizer state is already
  per-sample normalized inside the network (`NormContext`). Disable with
  `--no-norm-reward` / `--no-norm-obs`. (Logged `return` is always the raw value.)
- **Cycle-consistency** enters as a differentiable penalty (`--lambda-cycle`),
  keeping the `A→B→A`-minimal condition in the objective.
- **PPO** (`cat/rl/ppo.py`): a custom loop (no SB3) so it handles variable `D`/`N`
  and the structured covariance action; minibatches are grouped by
  `(target_algo, D, N)`.

Saves `models/PSO_CMAES_rl.pt`, which `evaluate.py` consumes directly
(`--model models/PSO_CMAES_rl.pt`).

## Evaluate (the headline metric)

```bash
python evaluate.py PSO CMAES -d 2 3 5 --n-individuals 12 --max-problems 60
```

On the BBOB **test** split, warms up the source optimizer to a switch point
sampled log-uniformly over the budget (one switch per problem, drawn
independently), then switches to the target under three hand-offs and reports
the best objective reached by each:

- **translated** — source state run through the trained translator;
- **lossy** — carry only the shared population (current default);
- **cold** — fresh target, ignores the source.

Success = translated ≥ lossy on average. The mean cycle-consistency drift
(`A -> B -> A`) is reported alongside.

## Layout

```
cat/
├── optimizers/      # vendored SubOptimizer base + PSO family + CMA-ES
├── suite/           # IOH/BBOB problem loader + train/test splits
├── state/           # StateSpec schema + native<->canonical tensor conversion
├── data/            # episode collector (A/B switching) + grouped dataset
├── models/          # norm, set-equivariant layers, encoder, heads, translator
├── losses.py        # cycle + reconstruction + utility-proxy
├── train_loop.py    # supervised training orchestration + checkpoint io
└── rl/              # RL path: env (gymnasium), policy (actor-critic), ppo
train.py             # supervised:  python train.py <ALG_A> <ALG_B>
train_rl.py          # PPO:         python train_rl.py <ALG_A> <ALG_B>
evaluate.py          # translated vs lossy vs cold hand-off (loads either checkpoint)
tests/               # round-trip, equivariance/PSD, overfit, RL env + PPO
```

Adding a new algorithm to the study: see
`.claude/skills/add-translation-algorithm/SKILL.md`.

## Acknowledgements

The optimizer implementations and the BBOB run/switch loop are adapted from the
companion **DynamicAlgorithmSelection2** project.
