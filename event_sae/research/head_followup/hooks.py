"""Native-compatible hooks for follow-up kernels; no original hook is patched.

Phase conditions require an explicit verified pre-action phase provider. The
legacy runner alone does not supply that information. Errors abort evaluation
through SAEHookError rather than becoming artificial robot failures.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from event_sae.openvla.intervene import SAEHookError
from .interventions import apply_edit, phase_gate


class FollowupHook:
    def __init__(self, handle, stream, counters):
        self.handle, self.stream, self.counters = handle, stream, counters
        self.removed = False

    def remove(self):
        if not self.removed:
            self.handle.remove()
            self.stream.close()
            self.removed = True

    def summary(self):
        return dict(self.counters)


def register_followup_hook(model, sae, *, layer_idx, feature_id, alpha, evidence_path,
                           direction=None, target_phase=None, phase_provider=None):
    """Attach to the final decoder layer of an already loaded, frozen policy.

``phase_provider(context)`` must return the CURRENT observation's phase, not
use a future action/outcome or replay the baseline episode's timing. Adapter
validation is an external prerequisite, not certified by accepting a callable.
"""
    layers = model.language_model.model.layers
    if layer_idx != len(layers) - 1:
        raise ValueError("follow-up targets the final decoder layer only")
    if target_phase is not None:
        phase_gate(target_phase, "other")
        if not callable(phase_provider):
            raise ValueError("phase intervention requires a validated pre-action phase provider")
    if getattr(sae, "training", False):
        raise ValueError("SAE must be frozen in eval mode")
    file = Path(evidence_path)
    file.parent.mkdir(parents=True, exist_ok=True)
    stream = file.open("x", encoding="utf-8")
    counters = {"num_forwards": 0, "enabled_forwards": 0, "edited_rows": 0,
                "max_cast_norm_abs_error": 0., "phase_provider_runtime_validated": False}
    previous_query, current_phase = None, "all"

    @torch.inference_mode()
    def hook(module, inputs, output):
        nonlocal previous_query, current_phase
        del module, inputs
        try:
            context = getattr(model, "_sae_hook_context", None)
            required = ("task_id", "task_episode_idx", "step_in_episode")
            if not isinstance(context, dict) or any(
                isinstance(context.get(k), bool) or not isinstance(context.get(k), int) or context[k] < 0 for k in required
            ):
                raise ValueError("complete current episode/step context required")
            query = tuple(context[k] for k in required)
            if query != previous_query:
                current_phase = phase_provider(dict(context)) if target_phase is not None else "all"
                previous_query = query
            enabled = target_phase is None or phase_gate(target_phase, current_phase)
            if not isinstance(output, tuple) or not output or not isinstance(output[0], torch.Tensor) or output[0].ndim != 3:
                raise ValueError("expected decoder output tuple with [batch, tokens, hidden]")
            hidden = output[0]
            if hidden.shape[0] != 1 or not torch.isfinite(hidden).all():
                raise ValueError("finite batch-size-one forward required")
            flat = hidden.reshape(-1, hidden.shape[-1])
            counters["num_forwards"] += 1
            evidence = {**{k: context[k] for k in required}, "forward_index": counters["num_forwards"],
                        "phase": current_phase, "enabled": enabled, "num_rows": len(flat)}
            updated = hidden
            if enabled:
                result, metrics = apply_edit(flat, sae, feature_id, alpha, direction=direction)
                updated = result.reshape_as(hidden)
                counters["enabled_forwards"] += 1
                changed_rows = int((metrics["realized_norm"] > 0).sum())
                counters["edited_rows"] += changed_rows
                error = float(metrics["cast_norm_abs_error"].max())
                counters["max_cast_norm_abs_error"] = max(counters["max_cast_norm_abs_error"], error)
                evidence.update(edited_rows=changed_rows,
                    intended_norm_sum=float(metrics["intended_norm"].sum()),
                    realized_norm_sum=float(metrics["realized_norm"].sum()), max_cast_norm_abs_error=error,
                    norm_scope=metrics["norm_scope"])
            stream.write(json.dumps(evidence, allow_nan=False) + "\n")
            stream.flush()
            return (updated, *output[1:])
        except Exception as exc:
            if isinstance(exc, SAEHookError):
                raise
            raise SAEHookError(f"Follow-up intervention failed: {exc}") from exc

    try:
        handle = layers[layer_idx].register_forward_hook(hook)
    except Exception:
        stream.close()
        raise
    return FollowupHook(handle, stream, counters)
