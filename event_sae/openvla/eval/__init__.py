from event_sae.openvla.eval.config import (
    EnvConfig,
    LoggingConfig,
    ModelConfig,
    RunConfig,
    SAECollectConfig,
    load_config,
    parse_overrides,
    resolve_task_ids,
    resolve_trial_indices,
)


def __getattr__(name):
    if name in {"EvalResult", "eval_libero"}:
        from event_sae.openvla.eval import runner
        return getattr(runner, name)
    raise AttributeError(name)

__all__ = [
    "EnvConfig",
    "EvalResult",
    "LoggingConfig",
    "ModelConfig",
    "RunConfig",
    "SAECollectConfig",
    "eval_libero",
    "load_config",
    "parse_overrides",
    "resolve_task_ids",
    "resolve_trial_indices",
]
