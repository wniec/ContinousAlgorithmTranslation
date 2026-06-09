"""Optional Weights & Biases logging.

Credentials and defaults are read from the environment; ``.env`` (git-ignored,
see ``.env.example``) is loaded automatically. ``wandb`` is imported lazily so
the dependency is only touched when logging is actually enabled.

Usage::

    logger = WandbLogger(enabled=args.wandb, project=..., run_name=..., config=vars(args))
    logger.log({"loss": 0.1}, step=epoch)
    logger.finish()

When ``enabled`` is False every method is a no-op, so call sites stay clean.
"""

from __future__ import annotations

import os

try:  # loading .env is best-effort; absence of python-dotenv must not break runs
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

DEFAULT_PROJECT = "continuous-algorithm-translation"


class WandbLogger:
    def __init__(
        self,
        enabled: bool = False,
        *,
        project: str | None = None,
        run_name: str | None = None,
        group: str | None = None,
        config: dict | None = None,
        x_axis: str = "step",
    ):
        self.run = None
        self.x_axis = x_axis
        if not enabled:
            return
        import wandb  # lazy: only required when --wandb is passed

        self.run = wandb.init(
            project=project or os.getenv("WANDB_PROJECT", DEFAULT_PROJECT),
            entity=os.getenv("WANDB_ENTITY") or None,
            name=run_name,
            group=group,
            mode=os.getenv("WANDB_MODE", "online"),
            config=config or {},
        )
        # Make the dashboards default to a meaningful x-axis (e.g. "update" /
        # "epoch") instead of wandb's internal "_step", and avoid charts that
        # read empty on the "_step" axis.
        wandb.define_metric(x_axis)
        wandb.define_metric("*", step_metric=x_axis)

    def log(self, metrics: dict, step: int | None = None) -> None:
        if self.run is None:
            return
        import wandb

        payload = dict(metrics)
        if step is not None:
            payload.setdefault(self.x_axis, step)
        # No explicit step= : let wandb auto-increment _step while the chosen
        # x_axis field carries the semantic step. Both axes then have data.
        wandb.log(payload)

    def finish(self) -> None:
        if self.run is not None:
            import wandb

            wandb.finish()
            self.run = None
