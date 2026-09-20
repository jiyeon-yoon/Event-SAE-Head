"""OpenVLA exports without eagerly importing torch or the simulator."""


def __getattr__(name):
    if name in __all__:
        from event_sae.openvla import activations
        return getattr(activations, name)
    raise AttributeError(name)

__all__ = [
    "ActivationCollectHandle",
    "apply_collect_hooks",
    "apply_sae_topk_collect_hooks",
]
