"""CPU-testable intervention kernels, not a GPU/runtime parity certificate.

Norm controls match the anchor's *pre-cast* edit norm at the same hidden
state and row. BF16 rounding and diverging closed-loop states are explicitly
not claimed to preserve trajectory-wide equality.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def _fraction(alpha: float) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha is the retained fraction and must lie in [0, 1]")
    return float(alpha)


@torch.inference_mode()
def feature_delta(hidden, sae, feature_id: int, alpha: float, *, encoded=None):
    """Use the original residual-preserving arithmetic before the hidden cast."""
    alpha = _fraction(alpha)
    if hidden.ndim != 2 or not hidden.is_floating_point() or not torch.isfinite(hidden).all():
        raise ValueError("hidden must be finite [original forward rows, hidden dim]")
    flat = hidden.float()
    z = sae.encode(flat) if encoded is None else encoded
    if (z.ndim != 2 or len(z) != len(flat) or z.dtype != torch.float32
            or z.device != flat.device or not torch.isfinite(z).all() or (z < 0).any()):
        raise ValueError("encoded must be original-forward finite nonnegative FP32 latents")
    if isinstance(feature_id, bool) or not isinstance(feature_id, int) or not 0 <= feature_id < z.shape[1]:
        raise ValueError("invalid feature ID")
    if alpha == 1:
        return torch.zeros_like(flat), z
    changed = z.clone()
    changed[:, feature_id] *= alpha
    delta = sae.decode(changed) - sae.decode(z)
    if delta.shape != flat.shape or not torch.isfinite(delta).all():
        raise ValueError("invalid decoder delta")
    return delta, z


@torch.inference_mode()
def decoder_direction(sae, feature_id: int, dict_size: int, *, device="cpu"):
    """Remove bias when obtaining a decoder column; intended for linear SAE."""
    if isinstance(feature_id, bool) or not isinstance(feature_id, int) or not 0 <= feature_id < dict_size:
        raise ValueError("invalid control feature ID")
    zero = torch.zeros((1, dict_size), dtype=torch.float32, device=device)
    unit = zero.clone()
    unit[0, feature_id] = 1
    direction = (sae.decode(unit) - sae.decode(zero))[0]
    if not torch.isfinite(direction).all() or direction.norm() == 0:
        raise ValueError("control decoder direction is zero/nonfinite")
    return direction / direction.norm()


def random_direction(hidden_dim: int, seed: int):
    """One fixed Gaussian direction per condition, generated on CPU, not per step."""
    if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim < 1:
        raise ValueError("invalid hidden dimension")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("invalid random direction seed")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    vector = torch.randn(hidden_dim, generator=generator, dtype=torch.float32)
    return vector / vector.norm()


@torch.inference_mode()
def apply_edit(hidden, sae, feature_id: int, alpha: float, *, direction=None, encoded=None):
    """Suppress a feature, or replace its delta by a norm-matched direction.

For a decoder control pass its negative unit direction, matching removal's
sign. The control's own activation is NOT used; it borrows anchor magnitude
and active support. This is a direction control, not control-feature ablation.
"""
    delta, _ = feature_delta(hidden, sae, feature_id, alpha, encoded=encoded)
    intended = delta.norm(dim=-1)
    if direction is not None:
        vector = torch.as_tensor(direction, device=hidden.device, dtype=torch.float32)
        if vector.shape != (hidden.shape[-1],) or not torch.isfinite(vector).all() or vector.norm() == 0:
            raise ValueError("direction must be a finite nonzero hidden-width vector")
        delta = intended[:, None] * (vector / vector.norm())[None, :]
    updated = (hidden.float() + delta).to(hidden.dtype)
    actual = (updated.float() - hidden.float()).norm(dim=-1)
    return updated, {
        "intended_norm": intended, "realized_norm": actual,
        "cast_norm_abs_error": (actual - intended).abs(),
        "norm_scope": "per_row_same_hidden_pre_cast_only",
    }


@dataclass(frozen=True)
class PhaseRule:
    """Physical-observation phase rule; never guesses a phase from step thirds.

The runtime adapter must supply verified target-specific pre-action telemetry.
No such adapter is assumed in the existing native runner.
"""
    grasp_distance: float

    def classify(self, observation: dict) -> str:
        if not math.isfinite(self.grasp_distance) or self.grasp_distance <= 0:
            raise ValueError("grasp_distance must be calibrated on discovery data")
        for key in ("target_grasped", "target_placed"):
            if not isinstance(observation.get(key), bool):
                raise ValueError(f"verified pre-action {key} is required")
        distance = observation.get("eef_target_distance")
        if (isinstance(distance, bool) or not isinstance(distance, (int, float))
                or not math.isfinite(distance) or distance < 0):
            raise ValueError("verified pre-action eef_target_distance is required")
        if observation["target_placed"]:
            return "other"
        if observation["target_grasped"]:
            return "transport"
        return "grasp" if distance <= self.grasp_distance else "approach"


def phase_gate(target_phase: str, observed_phase: str) -> bool:
    phases = {"approach", "grasp", "transport", "other"}
    if target_phase not in phases - {"other"} or observed_phase not in phases:
        raise ValueError("missing/unknown phase; do not silently disable an intervention")
    return observed_phase == target_phase
