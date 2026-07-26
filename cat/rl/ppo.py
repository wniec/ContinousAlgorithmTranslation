"""Custom PPO loop for the state translator.

Standard clipped-surrogate PPO with GAE, specialized in two ways:

* transitions carry **structured, variable-shape** observations and actions
  (the optimizer states), so the update groups them by ``(target_algo, D, N)``
  — same idea as ``cat/data/dataset.py`` — and each minibatch is drawn from one
  shape-homogeneous group;
* the actor objective adds a differentiable **cycle-consistency penalty**
  (``lambda_cycle · A->B->A drift``), realizing the "improvement + cycle penalty"
  objective more sample-efficiently than routing a differentiable quantity
  through the scalar reward. The drift is scored via ``cycle_mode`` — "field"
  (decoded fields, default) or "latent" (re-encoded latents only), mirroring
  ``cat.losses.batch_losses`` — and skipped entirely when ``lambda_cycle == 0``.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from cat.rl.env import TranslationEnv
from cat.rl.normalize import RunningMeanStd, normalize_rewards
from cat.rl.policy import CONTEXT_DIM, ActorCritic, Step


@dataclass
class PPOConfig:
    updates: int = 50
    rollout_steps: int = 1024
    ppo_epochs: int = 4
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.25
    lambda_cycle: float = 0.05
    cycle_mode: str = "field"  # "field" (decoded fields) or "latent" (re-encoded latents)
    lr: float = 3e-4
    # log_std is a single global scalar (see ActorCritic.log_std): at the shared
    # network lr it can only drift by ~lr per opt.step(), which is too slow to
    # track a moving optimum over the handful of steps in one update. Give it
    # its own Adam param group at lr * std_lr_mult instead.
    std_lr_mult: float = 20.0
    max_grad_norm: float = 1.0
    # Sliding-window rollout buffer: each update collects ~rollout_steps fresh
    # transitions, appends them to a deque of this capacity, and trains on the
    # whole window — so each record is reused across roughly
    # ``buffer_capacity / rollout_steps`` updates. ``None`` keeps only the latest
    # rollout (textbook on-policy PPO). Clamped up to rollout_steps so a full
    # fresh rollout always fits.
    buffer_capacity: int | None = None
    norm_reward: bool = True
    norm_obs: bool = True
    device: str = "cpu"
    seed: int = 42
    log_every: int = 1


@dataclass
class Transition:
    step: Step
    reward: float
    context: np.ndarray
    adv: float = 0.0
    ret: float = 0.0


@dataclass
class PPOLog:
    history: list[dict] = field(default_factory=list)
    ret_rms: "RunningMeanStd | None" = None
    obs_rms: "RunningMeanStd | None" = None


class RolloutBuffer:
    """Sliding-window store of transitions, retained across PPO updates.

    Mirrors ``DynamicAlgorithmSelection``'s ``RolloutBuffer``: rather than a
    fresh buffer per update (textbook on-policy PPO), it keeps the most recent
    ``capacity`` transitions so each record is reused over several updates. The
    clipped PPO ratio (``exp(logp - old_logp)``) tolerates the mild
    off-policyness of older records still inside the window.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buf: "deque[Transition]" = deque(maxlen=capacity)

    def clear(self) -> None:
        self._buf.clear()

    def extend(self, transitions: list[Transition]) -> None:
        # deque(maxlen) evicts the oldest transitions once capacity is exceeded.
        self._buf.extend(transitions)

    def as_list(self) -> list[Transition]:
        return list(self._buf)

    def __len__(self) -> int:
        return len(self._buf)


def _gae(rewards, values, gamma, lam):
    T = len(rewards)
    adv = [0.0] * T
    last = 0.0
    for t in reversed(range(T)):
        next_v = values[t + 1] if t + 1 < T else 0.0  # episodes terminate -> 0
        delta = rewards[t] + gamma * next_v - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    returns = [adv[t] + values[t] for t in range(T)]
    return adv, returns


def collect_rollout(
    ac: ActorCritic,
    envs,
    cfg: PPOConfig,
    ep_rng,
    ret_rms: RunningMeanStd | None = None,
    obs_rms: RunningMeanStd | None = None,
) -> tuple[list[Transition], list[float], dict[str, list[float]]]:
    """Run whole episodes until at least rollout_steps transitions are gathered.

    Reward normalization (``ret_rms``) and context/observation normalization
    (``obs_rms``) use running statistics that persist across updates; the
    normalized context is stored in each transition so the PPO update is
    consistent with what the policy saw at collection time.

    The third return value maps each switch type (``"<source>-><target>"``) to
    its raw per-step rewards, but only under the ``relative`` reward mode — the
    one mode whose reward is directly comparable across the two hand-off
    directions (each is an improvement over that same direction's lossy default).
    """
    transitions: list[Transition] = []
    ep_returns: list[float] = []
    switch_rewards: dict[str, list[float]] = defaultdict(list)
    while len(transitions) < cfg.rollout_steps:
        env: TranslationEnv = envs[ep_rng.integers(len(envs))]
        obs, _ = env.reset()
        ep: list[Transition] = []
        rewards, values, contexts = [], [], []
        done = False
        while not done:
            if not env.source_has_full_state():
                break  # degenerate source; abandon this episode
            # Critic side-input: progress scalars.
            raw_ctx = np.asarray(obs["context"], dtype=np.float32)
            if obs_rms is not None:
                obs_rms.update(raw_ctx[None])
                ctx = obs_rms.normalize(raw_ctx)
            else:
                ctx = raw_ctx
            switch = f"{obs['source_algo']}->{obs['target_algo']}"
            step = ac.act(obs, device=cfg.device, context=ctx)
            next_obs, reward, term, trunc, _ = env.step(step.native)
            ep.append(Transition(step=step, reward=reward, context=ctx))
            rewards.append(reward)
            values.append(step.value)
            contexts.append(ctx)
            if env.reward_mode == "relative":
                switch_rewards[switch].append(reward)  # raw, per switch direction
            done = term or trunc
            obs = next_obs
        if not ep:
            continue
        ep_returns.append(float(sum(rewards)))  # report the RAW return
        train_rewards = (
            normalize_rewards(rewards, cfg.gamma, ret_rms)
            if ret_rms is not None
            else rewards
        )
        adv, ret = _gae(train_rewards, values, cfg.gamma, cfg.gae_lambda)
        for tr, a, r in zip(ep, adv, ret):
            tr.adv, tr.ret = a, r
        transitions.extend(ep)
    return transitions, ep_returns, switch_rewards


def _group_key(tr: Transition):
    s = tr.step.src_state
    return (tr.step.target_algo, s.d, s.n)


def update(ac: ActorCritic, opt, transitions: list[Transition], cfg: PPOConfig) -> dict:
    device = torch.device(cfg.device)
    # Normalize advantages into a local array indexed by position — never mutate
    # Transition.adv, since records persist across updates (sliding-window
    # buffer) and the stored raw GAE must survive for the next update.
    advs = torch.tensor([t.adv for t in transitions], dtype=torch.float32)
    norm_advs = ((advs - advs.mean()) / (advs.std() + 1e-8)).tolist()

    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, tr in enumerate(transitions):
        groups[_group_key(tr)].append(i)

    gen = torch.Generator().manual_seed(cfg.seed)
    metrics = defaultdict(float)
    policy_stds: list[float] = []
    n_batches = 0

    for _ in range(cfg.ppo_epochs):
        for (target_algo, _, _), idxs in groups.items():
            order = torch.randperm(len(idxs), generator=gen).tolist()
            idxs = [idxs[o] for o in order]
            for start in range(0, len(idxs), cfg.minibatch_size):
                mb = idxs[start : start + cfg.minibatch_size]
                states = [transitions[i].step.src_state.to(device) for i in mb]
                source_algo = states[0].algo  # constant within a target-keyed group
                actions = torch.stack([transitions[i].step.action for i in mb]).to(
                    device
                )
                contexts = torch.tensor(
                    np.stack([transitions[i].context for i in mb]),
                    dtype=torch.float32,
                    device=device,
                )
                old_logp = torch.tensor(
                    [transitions[i].step.log_prob for i in mb],
                    dtype=torch.float32,
                    device=device,
                )
                old_value = torch.tensor(
                    [transitions[i].step.value for i in mb],
                    dtype=torch.float32,
                    device=device,
                )
                adv = torch.tensor(
                    [norm_advs[i] for i in mb], dtype=torch.float32, device=device
                )
                ret = torch.tensor(
                    [transitions[i].ret for i in mb], dtype=torch.float32, device=device
                )

                logp, entropy, value, batch, ctx = ac.evaluate_actions(
                    states, source_algo, target_algo, actions, contexts
                )
                ratio = (logp - old_logp).exp()
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                # PPO value-clipping + Huber: bounds the value gradient so a
                # critic spike can't dominate the (globally clipped) gradient.
                v_clipped = old_value + (value - old_value).clamp(-cfg.clip, cfg.clip)
                vl_unclipped = F.smooth_l1_loss(value, ret, reduction="none")
                vl_clipped = F.smooth_l1_loss(v_clipped, ret, reduction="none")
                value_loss = torch.max(vl_unclipped, vl_clipped).mean()
                ent = entropy.mean()
                # Skip the A->B->A round trip entirely when it wouldn't affect
                # the loss anyway, rather than computing it just to multiply by 0.
                if cfg.lambda_cycle == 0.0:
                    cycle = value.new_zeros(())
                else:
                    cycle = ac.cycle_drift(batch, ctx, cycle_mode=cfg.cycle_mode)

                loss = (
                    policy_loss
                    + cfg.vf_coef * value_loss
                    - cfg.ent_coef * ent
                    + cfg.lambda_cycle * cycle
                )
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ac.parameters(), cfg.max_grad_norm)
                opt.step()

                metrics["policy_loss"] += float(policy_loss.detach())
                metrics["value_loss"] += float(value_loss.detach())
                metrics["cycle"] += float(cycle.detach())
                n_batches += 1

    n_batches = max(n_batches, 1)
    out = {k: v / n_batches for k, v in metrics.items()}
    # Log the exploration scale (policy std) rather than the summed differential
    # entropy: std is sign-stable and independent of the action dimension, which
    # varies across (PSO/CMA-ES)-target groups. The summary spread ranges over
    # the per-segment stds (ActorCritic.log_std) at the end of the update — the
    # segments are what can actually diverge from one another; each also gets its
    # own named series so the individual field kinds are visible.
    with torch.no_grad():
        for algo, p in ac.log_std.items():
            seg_std = p.exp()
            for name, s in zip(
                ac.translator.decoders[algo].segment_names(), seg_std.tolist()
            ):
                out[f"policy_std/{algo}.{name}"] = s
            policy_stds.extend(seg_std.tolist())
    stds = np.asarray(policy_stds) if policy_stds else np.zeros(1)
    out["policy_std"] = float(stds.mean())
    out["policy_std_median"] = float(np.median(stds))
    out["policy_std_min"] = float(stds.min())
    out["policy_std_max"] = float(stds.max())
    return out


def train_ppo(ac: ActorCritic, envs, cfg: PPOConfig, log_fn=None) -> PPOLog:
    """Run PPO. ``log_fn(metrics)`` is called once per update (for W&B; optional).

    The reward/observation normalizers (running statistics) are attached to the
    returned log as ``log.ret_rms`` / ``log.obs_rms`` so they can be checkpointed.
    """
    ac.to(cfg.device)
    other_params = [
        p for n, p in ac.named_parameters() if not n.startswith("log_std.")
    ]
    opt = torch.optim.Adam(
        [
            {"params": other_params},
            {"params": list(ac.log_std.values()), "lr": cfg.lr * cfg.std_lr_mult},
        ],
        lr=cfg.lr,
    )
    ep_rng = np.random.default_rng(cfg.seed)
    log = PPOLog()
    ret_rms = RunningMeanStd(()) if cfg.norm_reward else None
    obs_rms = RunningMeanStd((CONTEXT_DIM,)) if cfg.norm_obs else None
    log.ret_rms, log.obs_rms = ret_rms, obs_rms

    # Persistent sliding-window buffer: each record lives for several updates.
    # Clamp capacity up so at least one full fresh rollout always fits.
    capacity = max(cfg.buffer_capacity or cfg.rollout_steps, cfg.rollout_steps)
    buffer = RolloutBuffer(capacity)

    for u in range(cfg.updates):
        fresh, ep_returns, switch_rewards = collect_rollout(
            ac, envs, cfg, ep_rng, ret_rms, obs_rms
        )
        buffer.extend(fresh)
        m = update(ac, opt, buffer.as_list(), cfg)
        m["update"] = u
        # Report the raw return of the freshly collected episodes only, so the
        # learning curve reflects the current policy (not stale window records).
        m["mean_return"] = float(np.mean(ep_returns)) if ep_returns else 0.0
        # Per-switch-direction mean reward (relative mode only; empty otherwise).
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
                f"| pi {m['policy_loss']:.4f} | vf {m['value_loss']:.4f} "
                f"| std {m['policy_std']:.4f} | cycle {m['cycle']:.4f}"
            )
    return log
