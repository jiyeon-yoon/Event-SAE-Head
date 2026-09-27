"""Dependency-free design and cost inventory; deliberately cannot run rollouts."""
from __future__ import annotations

import math
import random

from event_sae.research.output_head.provenance import fingerprint


def _ids(values, name, *, size=None):
    if (not isinstance(values, list) or not values or (size is not None and len(values) != size)
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values)
            or len(set(values)) != len(values)):
        raise ValueError(f"invalid {name}")
    return values


def select_decoder_controls(active_ids, anchor_ids, seed):
    """Sampling input is discovery activity only. Outcomes/ranks are not inputs."""
    anchors = _ids(list(anchor_ids), "anchors")
    pool = sorted(set(_ids(list(active_ids), "active IDs")) - set(anchors))
    if len(pool) < len(anchors):
        raise ValueError("not enough active non-anchor control directions")
    chosen = random.Random(seed).sample(pool, len(anchors))
    return dict(zip(anchors, chosen))


def design_summary(spec):
    if spec.get("schema_version") != "head_followup_all5_design_v1":
        raise ValueError("unknown follow-up design")
    for name in ("execute", "recollect_discovery", "retrain_sae", "automatic_download", "automatic_cloud_provisioning"):
        if spec.get(name) is not False:
            raise ValueError(f"{name} must remain false in this design-only CLI")
    tasks = _ids(spec.get("task_ids"), "tasks")
    anchors = _ids(spec.get("anchor_features"), "anchors", size=3)
    event = _ids(spec.get("event_top3"), "Event Top-3", size=3)
    seeds = _ids(spec.get("random_direction_seeds"), "direction seeds")
    trials = spec.get("evaluation_cases_per_task")
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("invalid cases per task")
    if spec.get("decoder_controls_per_anchor") != 1 or spec.get("top_k") != 3:
        raise ValueError("this version specifies one decoder control per anchor and Top-3")
    alphas = spec.get("retained_fractions")
    if (not isinstance(alphas, list) or not alphas or len(set(alphas)) != len(alphas)
            or any(isinstance(a, bool) or not isinstance(a, (int, float)) or not 0 < a < 1 for a in alphas)):
        raise ValueError("weak-suppression retained fractions must be unique and between zero and one")
    if spec.get("behavior_phases") != ["approach", "grasp", "transport"]:
        raise ValueError("explicit physical phase protocol required")
    conditions = [{"id": "raw", "kind": "raw"}, {"id": "identity", "kind": "identity", "alpha": 1}]
    full_features = set(anchors) | set(event)
    missing = []
    for name in ("action_top3", "drvla_top3"):
        if spec.get(name) is None:
            missing.append(name)
        else:
            full_features.update(_ids(spec[name], name, size=3))
    for fid in sorted(full_features):
        conditions.append({"id": f"feature-{fid}-alpha0", "kind": "feature", "feature_id": fid, "alpha": 0})
    for fid in anchors:
        conditions.append({"id": f"matched-decoder-{fid}", "kind": "matched_decoder_direction",
                           "anchor_feature_id": fid, "control_feature_id": None})
        for seed in seeds:
            conditions.append({"id": f"matched-random-{fid}-{seed}", "kind": "matched_random_direction",
                               "anchor_feature_id": fid, "direction_seed": seed})
        for alpha in alphas:
            conditions.append({"id": f"feature-{fid}-alpha{alpha}", "kind": "feature",
                               "feature_id": fid, "alpha": alpha})
        for phase in spec["behavior_phases"]:
            conditions.append({"id": f"phase-{phase}-{fid}", "kind": "phase_feature",
                               "feature_id": fid, "alpha": 0, "phase": phase})
    cases = len(tasks) * trials
    low, high = len(conditions) * cases, (len(conditions) + 3 * len(missing)) * cases
    old_hours, old_rollouts = spec.get("recorded_old_condition_hours"), spec.get("recorded_old_rollouts")
    if (isinstance(old_hours, bool) or not isinstance(old_hours, (int, float)) or not math.isfinite(old_hours)
            or old_hours <= 0 or isinstance(old_rollouts, bool) or not isinstance(old_rollouts, int) or old_rollouts < 1):
        raise ValueError("invalid measured cost reference")
    return {
        "status": "design_only_not_executable", "execute": False, "design_hash": fingerprint(spec),
        "task_count": len(tasks), "cases_per_condition": cases,
        "known_conditions": len(conditions), "pending_rankings": missing,
        "rollouts_without_automatic_old_result_reuse": {"min": low, "max": high},
        "reference_wall_hours": {"min": low * old_hours / old_rollouts, "max": high * old_hours / old_rollouts},
        "cost_caveat": "linear_extrapolation_only_not_a_runtime_bound; excludes_offline_scoring_setup_smoke_and_phase_calibration",
        "conditions": conditions, "remaining_gates": spec["run_gates"],
    }
