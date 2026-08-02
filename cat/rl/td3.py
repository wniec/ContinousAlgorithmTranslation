"""Twin Delayed DDPG (TD3), as an off-policy alternative to PPO for the same
``TranslationEnv`` (``cat/rl/env.py``) and the same *continuous* action: the
target optimizer's full decoded warm-start, produced by the set-equivariant
``TranslatorPair`` (``cat/models/translator.py``).

Unlike PPO's stochastic Gaussian policy (learnable global ``log_std``, on-policy
clipped surrogate objective), TD3's actor is **deterministic** — it outputs the
translator's mean action directly, with exploration coming from Gaussian noise
added in action space during rollout collection, and training coming from a
replay buffer (off-policy, more sample-efficient) rather than fresh on-policy
rollouts. Two independent critics (``TwinQCritic``) estimate Q(s, a) and the
*minimum* of the two is used for both the TD-target and (via ``target policy
smoothing``, noised target actions) to curb the value-overestimation bias a
single critic would suffer from continuous-action Q-learning; the actor is
updated only every ``policy_freq`` critic updates, using just one critic
(``q1``), and both target networks are Polyak-averaged rather than hard-synced.

The Q-network's structural challenge: the action is not a fixed-size vector
across the board — its dimension depends on ``(target_algo, D, N)`` (see
``StateDecoder.action_dim`` in ``cat/models/heads.py``), so a plain
``concat(state, action) -> MLP`` critic can't have a fixed input size. The fix
reuses the same trick that makes the *state* encoder shape-agnostic in the
first place: decode the action back into its structured per-algorithm fields
(``build_state``, which is entirely parameter-free — pure reshape + denormalize
+ the static ``CovFactorHead.assemble`` — so replaying an old, off-policy action
through the *current* decoder always reconstructs bit-identical structured
fields, no staleness concern) and encode *that* with the target algorithm's own
``StateEncoder``, exactly like encoding a real state. The critic then combines
the source state's embedding with this "action-as-state" embedding.
"""

from __future__ import annotations

import copy
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cat.data.dataset import collate
from cat.models.encoder import StateEncoder
from cat.models.layers import MLP
from cat.models.norm import NormContext
from cat.models.translator import TranslatorPair
from cat.rl.env import TranslationEnv
from cat.rl.normalize import RunningMeanStd, normalize_rewards
from cat.state.canonical import (
    CanonicalState,
    from_canonical,
    resample_population,
    to_canonical,
)

PROGRESS_DIM = 2  # kept in sync with cat.rl.policy.PROGRESS_DIM (no ELA side-input)
CONTEXT_DIM = PROGRESS_DIM


def _ctx(state: CanonicalState) -> NormContext:
    return NormContext.from_shared(state.positions, state.values)


def _action_state(
    actor: "TD3Actor", target_algo: str, action: Tensor, batch: CanonicalState, ctx
) -> CanonicalState:
    """Decode a flat action vector into a full (batched) ``CanonicalState`` on
    ``target_algo``, sharing ``batch``'s population (see module docstring: this
    reconstruction is parameter-free, so it's safe regardless of which actor
    instance/weights are passed — an ``actor`` argument is only needed because
    ``build_state`` lives on an ``nn.Module`` decoder head)."""
    specific = actor.translator.decoders[target_algo].build_state(action, ctx, batch.n)
    return CanonicalState(
        algo=target_algo,
        positions=batch.positions,
        values=batch.values,
        best_x=batch.best_x,
        best_y=batch.best_y,
        specific=specific,
    )


# --------------------------------------------------------------------------- #
# Actor                                                                        #
# --------------------------------------------------------------------------- #


@dataclass
class TD3Step:
    """Everything produced for one acted transition (used by the replay buffer)."""

    src_state: CanonicalState  # un-batched, on cpu
    target_algo: str
    action: Tensor  # (action_dim,) raw mean(+noise) action, on the SOURCE's own N
    native: dict  # target warm-start dict (resized to target's own N) for env.step


class TD3Actor(nn.Module):
    def __init__(
        self,
        algo_a: str,
        algo_b: str,
        hidden: int = 64,
        n_layers: int = 2,
        cov_rank: int = 4,
        resample_seed: int | None = None,
    ):
        super().__init__()
        # Drives the population-resizing jitter in _assemble_native, exactly
        # like PPO's ActorCritic (cat/rl/policy.py) — reproducible given a seed.
        self._resample_rng = np.random.default_rng(resample_seed)
        self.translator = TranslatorPair(
            algo_a, algo_b, hidden=hidden, n_layers=n_layers, cov_rank=cov_rank
        )
        self.algo_a = algo_a
        self.algo_b = algo_b

    def forward(self, target_algo: str, batch: CanonicalState, ctx) -> Tensor:
        z = self.translator.encode(batch, ctx)
        return self.translator.decoders[target_algo].action_mean(z, batch.positions, ctx)

    def obs_to_state(self, obs: dict, device="cpu") -> CanonicalState:
        return to_canonical(obs["source_algo"], obs["native"], device=device)

    @torch.no_grad()
    def act(
        self,
        obs: dict,
        expl_noise: float,
        rng: np.random.Generator,
        device: str = "cpu",
        context=None,
    ) -> TD3Step:
        src = self.obs_to_state(obs, device)
        target = obs["target_algo"]
        batch = collate([src])
        ctx = _ctx(batch)

        mean = self.forward(target, batch, ctx)
        if expl_noise > 0:
            noise = torch.as_tensor(
                rng.normal(scale=expl_noise, size=tuple(mean.shape)),
                dtype=mean.dtype,
                device=device,
            )
            action = mean + noise
        else:
            action = mean

        native = self._assemble_native(action, target, batch, ctx, obs.get("target_n"))
        return TD3Step(
            src_state=src.to("cpu"),
            target_algo=target,
            action=action.squeeze(0).cpu(),
            native=native,
        )

    def _assemble_native(
        self, action: Tensor, target: str, batch: CanonicalState, ctx, n_target=None
    ) -> dict:
        """Mirrors ``ActorCritic._assemble_native`` (cat/rl/policy.py): the
        action is always decoded onto the source's own population (keeping the
        replay buffer's action_dim a pure function of (target_algo, D, N) for
        homogeneous batching); only the assembled warm-start payload handed to
        env.step is resized to the target's own N."""
        tgt_state = _action_state(self, target, action, batch, ctx).index(0)
        if n_target is not None and n_target != tgt_state.n:
            tgt_state = resample_population(tgt_state, n_target, self._resample_rng)
        return from_canonical(tgt_state)

    def cycle_drift(self, batch: CanonicalState, ctx, cycle_mode: str = "field"):
        """Mean A->B->A drift for a batch; see ``TranslatorPair.cycle_drift``
        (shared with PPO's ``ActorCritic``)."""
        return self.translator.cycle_drift(batch, ctx, cycle_mode)


# --------------------------------------------------------------------------- #
# Twin critics                                                                 #
# --------------------------------------------------------------------------- #


class QCritic(nn.Module):
    """One of the two twins. Encodes the source state AND the proposed
    action-as-state (see module docstring), each with their own dedicated
    per-algorithm ``StateEncoder`` — kept separate from the actor's own
    encoder for the same reason PPO's critic does (cat/rl/policy.py): sharing
    would make the critic chase a representation the actor/cycle losses
    constantly reshape."""

    def __init__(self, algo_a: str, algo_b: str, hidden: int = 64, n_layers: int = 2):
        super().__init__()
        self.src_encoders = nn.ModuleDict(
            {a: StateEncoder(a, hidden, n_layers) for a in (algo_a, algo_b)}
        )
        self.act_encoders = nn.ModuleDict(
            {a: StateEncoder(a, hidden, n_layers) for a in (algo_a, algo_b)}
        )
        self.head = MLP([2 * hidden + CONTEXT_DIM, hidden, 1])

    def forward(
        self,
        source_algo: str,
        target_algo: str,
        src_batch: CanonicalState,
        act_batch: CanonicalState,
        ctx,
        context: Tensor,
    ) -> Tensor:
        zs = self.src_encoders[source_algo](src_batch, ctx)
        za = self.act_encoders[target_algo](act_batch, ctx)
        feats = torch.cat([zs.global_vec, za.global_vec, context], dim=-1)
        return self.head(feats).squeeze(-1)


class TwinQCritic(nn.Module):
    def __init__(self, algo_a: str, algo_b: str, hidden: int = 64, n_layers: int = 2):
        super().__init__()
        self.q1 = QCritic(algo_a, algo_b, hidden, n_layers)
        self.q2 = QCritic(algo_a, algo_b, hidden, n_layers)

    def forward(
        self, source_algo, target_algo, src_batch, act_batch, ctx, context
    ) -> tuple[Tensor, Tensor]:
        return (
            self.q1(source_algo, target_algo, src_batch, act_batch, ctx, context),
            self.q2(source_algo, target_algo, src_batch, act_batch, ctx, context),
        )


def _polyak_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, p in zip(target.parameters(), online.parameters()):
            tp.mul_(1.0 - tau).add_(p, alpha=tau)


# --------------------------------------------------------------------------- #
# Replay buffer                                                                #
# --------------------------------------------------------------------------- #


@dataclass
class TD3Transition:
    src_state: CanonicalState  # s, un-batched, cpu
    source_algo: str
    action: Tensor  # (action_dim,), cpu — raw mean(+noise) action taken
    reward: float
    context: np.ndarray  # normalized context at s
    # Always a well-formed CanonicalState (see collect_rollout's guard) with
    # algo == the deterministic partner algorithm; `done` masks its
    # contribution to the TD-target rather than needing a None special case.
    next_state: CanonicalState
    next_context: np.ndarray
    done: bool


class ReplayBuffer:
    """Sliding-window replay memory (FIFO once ``capacity`` is exceeded)."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buf: "deque[TD3Transition]" = deque(maxlen=capacity)

    def extend(self, transitions: list[TD3Transition]) -> None:
        self._buf.extend(transitions)

    def as_list(self) -> list[TD3Transition]:
        return list(self._buf)

    def __len__(self) -> int:
        return len(self._buf)


@dataclass
class TD3Config:
    updates: int = 500
    rollout_steps: int = 1024  # env steps collected (noisy actor) per update
    gradient_steps: int = 64  # critic updates performed per update
    minibatch_size: int = 256
    buffer_capacity: int = 20_000
    gamma: float = 0.8
    actor_lr: float = 8e-5
    critic_lr: float = 3e-4
    expl_noise: float = 0.1  # std of Gaussian exploration noise (action space)
    policy_noise: float = 0.2  # std of target policy smoothing noise
    noise_clip: float = 0.5  # clip range for target policy smoothing noise
    policy_freq: int = 2  # delayed actor/target update frequency, in gradient steps
    tau: float = 0.005  # Polyak averaging coefficient for target networks
    max_grad_norm: float = 1.0
    lambda_cycle: float = 0.0  # optional cycle-consistency penalty on the actor loss
    cycle_mode: str = "field"
    norm_reward: bool = True
    norm_obs: bool = True
    device: str = "cpu"
    seed: int = 42
    log_every: int = 1


@dataclass
class TD3Log:
    history: list[dict] = field(default_factory=list)
    ret_rms: "RunningMeanStd | None" = None
    obs_rms: "RunningMeanStd | None" = None


def collect_rollout(
    actor: TD3Actor,
    envs,
    cfg: TD3Config,
    ep_rng: np.random.Generator,
    ret_rms: RunningMeanStd | None = None,
    obs_rms: RunningMeanStd | None = None,
) -> tuple[list[TD3Transition], list[float], dict[str, list[float]]]:
    """Run whole episodes (actor + Gaussian exploration noise) until at least
    rollout_steps transitions are gathered. Mirrors ``cat.rl.ppo.collect_rollout``
    (same env, same reward/obs normalization). Unlike PPO's GAE (which only needs the
    reward/value sequence), TD3 needs a well-formed ``next_state`` right away to
    bootstrap from, so a transition whose resulting state is degenerate is
    dropped and the rest of the episode abandoned.

    The third return value maps each switch type (``"<source>-><target>"``) to
    its raw per-step rewards, under the ``relative`` and ``noswitch`` reward
    modes (see ``cat.rl.ppo.collect_rollout``)."""
    transitions: list[TD3Transition] = []
    ep_returns: list[float] = []
    switch_rewards: dict[str, list[float]] = defaultdict(list)
    while len(transitions) < cfg.rollout_steps:
        env: TranslationEnv = envs[ep_rng.integers(len(envs))]
        obs, _ = env.reset()
        ep: list[TD3Transition] = []
        rewards: list[float] = []
        done = False
        while not done:
            if not env.source_has_full_state():
                break  # degenerate source; abandon this episode
            raw_ctx = np.asarray(obs["context"], dtype=np.float32)
            if obs_rms is not None:
                obs_rms.update(raw_ctx[None])
                ctx = obs_rms.normalize(raw_ctx)
            else:
                ctx = raw_ctx

            switch = f"{obs['source_algo']}->{obs['target_algo']}"
            step = actor.act(
                obs, expl_noise=cfg.expl_noise, rng=ep_rng, device=cfg.device, context=ctx
            )
            next_obs, reward, term, trunc, info = env.step(step.native)
            done = term or trunc

            if not env.source_has_full_state():
                break  # resulting state degenerate; drop this transition too

            next_raw_ctx = np.asarray(next_obs["context"], dtype=np.float32)
            if obs_rms is not None:
                obs_rms.update(next_raw_ctx[None])
                next_ctx = obs_rms.normalize(next_raw_ctx)
            else:
                next_ctx = next_raw_ctx
            next_state = to_canonical(next_obs["source_algo"], next_obs["native"]).to(
                "cpu"
            )

            ep.append(
                TD3Transition(
                    src_state=step.src_state,
                    source_algo=obs["source_algo"],
                    action=step.action,
                    reward=reward,
                    context=ctx,
                    next_state=next_state,
                    next_context=next_ctx,
                    done=done,
                )
            )
            rewards.append(reward)
            if env.reward_mode in ("relative", "noswitch", "mixed"):
                switch_rewards[switch].append(reward)  # raw, per switch direction
            # In 'mixed' mode also break out the two additive components so each
            # part's mean is logged (reward/mixed_noswitch, reward/mixed_relative).
            for part, value in info.get("reward_parts", {}).items():
                switch_rewards[part].append(value)
            obs = next_obs
        if not ep:
            continue
        ep_returns.append(float(sum(rewards)))  # report the RAW return
        train_rewards = (
            normalize_rewards(rewards, cfg.gamma, ret_rms)
            if ret_rms is not None
            else rewards
        )
        for tr, r in zip(ep, train_rewards):
            tr.reward = r
        transitions.extend(ep)
    return transitions, ep_returns, switch_rewards


def _group_key(tr: TD3Transition):
    return (tr.source_algo, tr.src_state.d, tr.src_state.n)


def update(
    actor: TD3Actor,
    target_actor: TD3Actor,
    critic: TwinQCritic,
    target_critic: TwinQCritic,
    actor_opt,
    critic_opt,
    transitions: list[TD3Transition],
    cfg: TD3Config,
    rng: np.random.Generator,
    grad_step: int,
) -> tuple[dict, int]:
    """``gradient_steps`` critic updates (every one), each on a shape-
    homogeneous minibatch sampled (with replacement) from one random
    ``(source_algo, D, N)`` group of the replay buffer — same grouping trick
    PPO uses, since the encoders need uniform shapes within a batch, and
    since the target algorithm is a deterministic function of the source
    algorithm here too (env alternates strictly between ``algo_a``/``algo_b``,
    each with its own fixed configured population size), every transition's
    ``next_state`` in a group shares the same algo/D/N as well. The actor and
    both target networks are updated only every ``policy_freq`` critic steps
    (TD3's delayed policy update)."""
    device = torch.device(cfg.device)
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, tr in enumerate(transitions):
        groups[_group_key(tr)].append(i)
    group_keys = list(groups.keys())

    metrics = {"critic_loss": 0.0, "q_value": 0.0, "actor_loss": 0.0, "cycle": 0.0}
    n_steps = 0
    actor_updates = 0
    for _ in range(cfg.gradient_steps):
        gi = int(rng.integers(len(group_keys)))
        idxs = groups[group_keys[gi]]
        size = min(cfg.minibatch_size, len(idxs))
        pick = rng.choice(len(idxs), size=size, replace=len(idxs) < cfg.minibatch_size)
        mb = [idxs[i] for i in pick]
        source_algo = transitions[mb[0]].source_algo
        target_algo = transitions[mb[0]].next_state.algo
        # Deterministic pairing: s' is encoded as target_algo, and a' decodes
        # back onto source_algo.
        next_source_algo, next_target_algo = target_algo, source_algo

        batch = collate([transitions[i].src_state.to(device) for i in mb])
        ctx = _ctx(batch)
        contexts = torch.tensor(
            np.stack([transitions[i].context for i in mb]),
            dtype=torch.float32,
            device=device,
        )
        actions = torch.stack([transitions[i].action for i in mb]).to(device)
        rewards = torch.tensor(
            [transitions[i].reward for i in mb], dtype=torch.float32, device=device
        )
        dones = torch.tensor(
            [transitions[i].done for i in mb], dtype=torch.float32, device=device
        )

        next_batch = collate([transitions[i].next_state.to(device) for i in mb])
        next_ctx = _ctx(next_batch)
        next_contexts = torch.tensor(
            np.stack([transitions[i].next_context for i in mb]),
            dtype=torch.float32,
            device=device,
        )

        # ---- critic update ----
        with torch.no_grad():
            a2_mean = target_actor(next_target_algo, next_batch, next_ctx)
            noise = (torch.randn_like(a2_mean) * cfg.policy_noise).clamp(
                -cfg.noise_clip, cfg.noise_clip
            )
            a2 = a2_mean + noise
            act_state2 = _action_state(actor, next_target_algo, a2, next_batch, next_ctx)
            q1_t, q2_t = target_critic(
                next_source_algo, next_target_algo, next_batch, act_state2, next_ctx,
                next_contexts,
            )
            q_next = torch.min(q1_t, q2_t)
            y = rewards + cfg.gamma * (1.0 - dones) * q_next

        act_state = _action_state(actor, target_algo, actions, batch, ctx)
        q1, q2 = critic(source_algo, target_algo, batch, act_state, ctx, contexts)
        critic_loss = F.smooth_l1_loss(q1, y) + F.smooth_l1_loss(q2, y)

        critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
        critic_opt.step()

        metrics["critic_loss"] += float(critic_loss.detach())
        metrics["q_value"] += float(q1.mean().detach())
        n_steps += 1
        grad_step += 1

        # ---- delayed actor + target updates ----
        if grad_step % cfg.policy_freq == 0:
            a_new = actor(target_algo, batch, ctx)
            act_state_new = _action_state(actor, target_algo, a_new, batch, ctx)
            q1_new = critic.q1(source_algo, target_algo, batch, act_state_new, ctx, contexts)
            actor_loss = -q1_new.mean()
            if cfg.lambda_cycle != 0.0:
                cycle = actor.cycle_drift(batch, ctx, cycle_mode=cfg.cycle_mode)
                actor_loss = actor_loss + cfg.lambda_cycle * cycle
                metrics["cycle"] += float(cycle.detach())

            actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
            actor_opt.step()

            _polyak_update(target_actor, actor, cfg.tau)
            _polyak_update(target_critic, critic, cfg.tau)

            metrics["actor_loss"] += float(actor_loss.detach())
            actor_updates += 1

    n_steps = max(n_steps, 1)
    actor_updates = max(actor_updates, 1)
    out = {
        "critic_loss": metrics["critic_loss"] / n_steps,
        "q_value": metrics["q_value"] / n_steps,
        "actor_loss": metrics["actor_loss"] / actor_updates,
        "cycle": metrics["cycle"] / actor_updates,
    }
    return out, grad_step


def train_td3(
    actor: TD3Actor, critic: TwinQCritic, envs, cfg: TD3Config, log_fn=None
) -> TD3Log:
    """Run TD3. ``log_fn(metrics)`` is called once per update (for W&B;
    optional). Mirrors ``cat.rl.ppo.train_ppo``'s outer loop/logging shape so
    the two algorithms' runs are directly comparable."""
    actor.to(cfg.device)
    critic.to(cfg.device)
    target_actor = copy.deepcopy(actor).to(cfg.device)
    target_critic = copy.deepcopy(critic).to(cfg.device)
    for net in (target_actor, target_critic):
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    ep_rng = np.random.default_rng(cfg.seed)
    train_rng = np.random.default_rng(cfg.seed + 1)
    log = TD3Log()
    ret_rms = RunningMeanStd(()) if cfg.norm_reward else None
    obs_rms = RunningMeanStd((CONTEXT_DIM,)) if cfg.norm_obs else None
    log.ret_rms, log.obs_rms = ret_rms, obs_rms

    buffer = ReplayBuffer(cfg.buffer_capacity)
    grad_step = 0

    for u in range(cfg.updates):
        fresh, ep_returns, switch_rewards = collect_rollout(
            actor, envs, cfg, ep_rng, ret_rms, obs_rms
        )
        buffer.extend(fresh)
        m, grad_step = update(
            actor,
            target_actor,
            critic,
            target_critic,
            actor_opt,
            critic_opt,
            buffer.as_list(),
            cfg,
            train_rng,
            grad_step,
        )
        m["update"] = u
        m["mean_return"] = float(np.mean(ep_returns)) if ep_returns else 0.0
        # Per-switch-direction mean reward (relative/noswitch modes; else empty).
        for switch, rs in switch_rewards.items():
            m[f"reward/{switch}"] = float(np.mean(rs))
        m["n_transitions"] = len(buffer)
        m["n_fresh"] = len(fresh)
        log.history.append(m)
        if log_fn is not None:
            log_fn(m)
        if cfg.log_every and u % cfg.log_every == 0:
            print(
                f"update {u:3d} | return {m['mean_return']:.4f} "
                f"| critic {m['critic_loss']:.4f} | actor {m['actor_loss']:.4f} "
                f"| q {m['q_value']:.4f}"
            )
    return log
