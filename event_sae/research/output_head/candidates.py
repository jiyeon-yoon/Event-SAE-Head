"""Strict baseline adapters and reproducible, non-executing feature plans.

This module does not import a model or simulator. Matrix files are loaded on CPU
only when requested. Missing optional matrices never masquerade as event scores.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _id(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("Boolean is not a feature/task/trial ID")
    result = int(value)
    if result < 0 or str(result) != str(value):
        raise ValueError(f"Expected nonnegative integer ID, got {value!r}")
    return result


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _vector(values: Mapping[Any, Any]) -> dict[int, float]:
    out: dict[int, float] = {}
    for key, value in values.items():
        feature = _id(key)
        if feature in out:
            raise ValueError(f"Duplicate feature ID {feature}")
        out[feature] = _finite(value, "feature score")
    if not out:
        raise ValueError("A score vector cannot be empty")
    return out


def validate_score_scope(scope: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Refuse direct comparisons across tasks, dictionaries or discovery sets."""
    for name in ("task_ids", "dictionary_size", "discovery_manifest_hash", "split_manifest_hash", "sae_sha256"):
        if name not in expected or expected[name] is None:
            continue
        if name not in scope or scope[name] is None:
            raise ValueError(f"Score scope missing required {name}")
        actual, target = scope[name], expected[name]
        if name == "task_ids":
            actual, target = sorted(_id(v) for v in actual), sorted(_id(v) for v in target)
        if actual != target:
            raise ValueError(f"Score scope mismatch for {name}: {actual!r} != {target!r}")


def load_ranking_scores(
    source: str | Path | Mapping[str, Any],
    methods: Sequence[str],
    *,
    min_coverage: float = 0.5,
    expected_scope: Mapping[str, Any] | None = None,
    legacy_event_verified: bool = False,
) -> dict[str, Any]:
    """Return full suite vectors using original Event-SAE aggregation rules.

An old ``matrix`` is accepted for event_aligned only with an explicit verified
provenance assertion. Task means require real timestep weights; unlike the old
reader, missing weights do not silently become one. No episode split can be
recovered from an already aggregated matrix: its scope must certify that split.
"""
    import torch

    if not 0 <= min_coverage <= 1:
        raise ValueError("min_coverage must be in [0, 1]")
    payload = dict(source) if isinstance(source, Mapping) else torch.load(Path(source), map_location="cpu", weights_only=False)
    requested = list(methods)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Select one or more distinct ranking methods")
    keys = {"event_aligned": "matrix_raw", "window_mean": "matrix_window_mean", "task_mean": "matrix_task_mean"}
    if set(requested) - set(keys):
        raise ValueError(f"Unsupported matrix ranking: {set(requested) - set(keys)}")
    rows = payload.get("row_keys")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Score artifact requires nonempty row_keys")
    for row in rows:
        if "episode_coverage" not in row or "task_id" not in row:
            raise ValueError("Every matrix row requires task_id and episode_coverage")
        coverage = _finite(row["episode_coverage"], "episode_coverage")
        if not 0 <= coverage <= 1:
            raise ValueError("episode_coverage must be in [0, 1]")
        _id(row["task_id"])
    keep = [i for i, row in enumerate(rows) if float(row["episode_coverage"]) >= min_coverage]
    if not keep:
        raise ValueError("No canonical rows pass min_coverage")
    scope = dict(payload.get("scope", {}))
    row_task_ids = sorted({_id(row["task_id"]) for row in rows})
    if "task_ids" in scope and sorted(scope["task_ids"]) != row_task_ids:
        raise ValueError("Declared matrix task scope disagrees with row metadata")
    scope["task_ids"] = row_task_ids
    output: dict[str, dict[int, float]] = {}
    dimension: int | None = None
    for method in requested:
        key = keys[method]
        if key not in payload:
            if method == "event_aligned" and "matrix" in payload and legacy_event_verified:
                key = "matrix"
            else:
                raise ValueError(f"Selected {method} requires {key}; legacy fallback is forbidden")
        matrix = torch.as_tensor(payload[key], dtype=torch.float32, device="cpu")
        if matrix.ndim != 2 or matrix.shape[0] != len(rows) or not matrix.shape[1] or not torch.isfinite(matrix).all():
            raise ValueError(f"Invalid {key} shape/values")
        if dimension is not None and dimension != matrix.shape[1]:
            raise ValueError("Selected matrices have different feature dimensions")
        dimension = int(matrix.shape[1])
        if method == "event_aligned":
            vector = matrix[keep].mean(dim=0)
        elif method == "window_mean":
            if any("num_events" not in rows[i] for i in keep):
                raise ValueError("window_mean requires num_events")
            weights = torch.tensor([_id(rows[i]["num_events"]) for i in keep], dtype=torch.float32)
            if weights.sum() <= 0:
                raise ValueError("window_mean event weights sum to zero")
            vector = (matrix[keep] * weights[:, None]).sum(dim=0) / weights.sum()
        else:
            counts = payload.get("selection_counts", {}).get("task_timestep_counts", {})
            first: dict[int, int] = {}
            for i in keep:
                task = _id(rows[i]["task_id"])
                if task in first and not torch.equal(matrix[i], matrix[first[task]]):
                    raise ValueError("Repeated task_mean rows disagree")
                first.setdefault(task, i)
            weights_list = []
            for task in first:
                value = counts.get(task, counts.get(str(task)))
                if value is None or _id(value) == 0:
                    raise ValueError(f"task_mean requires positive timestep count for task {task}")
                weights_list.append(float(value))
            weights = torch.tensor(weights_list, dtype=torch.float32)
            vector = (matrix[list(first.values())] * weights[:, None]).sum(dim=0) / weights.sum()
        output[method] = {i: float(value) for i, value in enumerate(vector.tolist())}
    if "dictionary_size" in scope and scope["dictionary_size"] != dimension:
        raise ValueError("Declared dictionary_size disagrees with matrix")
    scope["dictionary_size"] = dimension
    if expected_scope:
        validate_score_scope(scope, expected_scope)
    return {"scores": output, "scope": scope, "synthetic": bool(payload.get("synthetic", False)),
            "provenance": {"legacy_event_verified": legacy_event_verified, "min_coverage": min_coverage,
                           "population_scope": "discovery" if scope.get("discovery_manifest_hash") else "legacy_reference"}}


def score_vectors(scores: Mapping[str, Any], methods: Sequence[str]) -> dict[str, dict[int, float]]:
    """Read metric columns or a method-to-vector dictionary without imputation."""
    if "feature_scores" in scores:
        records = scores["feature_scores"]
        if not isinstance(records, list) or not records:
            raise ValueError("feature_scores must contain records")
        ids = [_id(row["feature_id"]) for row in records]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate feature score records")
        output = {}
        for method in methods:
            if any(method not in row or row[method] is None for row in records):
                raise ValueError(f"Score records missing selected predictor {method}")
            output[method] = _vector({row["feature_id"]: row[method] for row in records})
        return output
    source = scores.get("scores", scores)
    return {method: _vector(source[method]) for method in methods if method in source}


def _eval_cases(manifest: Mapping[str, Any], *, synthetic: bool = False) -> list[dict[str, Any]]:
    cases = manifest.get("cases", manifest.get("episodes"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("eval_manifest requires nonempty cases/episodes")
    out, seen, hashes = [], set(), set()
    for row in cases:
        task, trial = _id(row["task_id"]), _id(row["task_episode_idx"])
        suite = row.get("suite", manifest.get("suite"))
        state_hash = row.get("initial_state_sha256")
        if not isinstance(suite, str) or not suite or not isinstance(state_hash, str) or not state_hash:
            raise ValueError("Evaluation cases require suite and initial_state_sha256")
        if not synthetic and not re.fullmatch(r"[0-9a-fA-F]{64}", state_hash):
            raise ValueError("Real evaluation initial_state_sha256 must be a SHA-256 digest")
        state_hash = state_hash.lower()
        key = (suite, task, trial)
        hash_key = (suite, task, state_hash)
        if key in seen or hash_key in hashes:
            raise ValueError("Duplicate evaluation initial state/trial")
        seen.add(key)
        hashes.add(hash_key)
        out.append({**row, "suite": suite, "task_id": task, "task_episode_idx": trial,
                    "initial_state_sha256": state_hash})
    return sorted(out, key=lambda row: (row["suite"], row["task_id"], row["task_episode_idx"]))


def build_evaluation_plan(
    scores: Mapping[str, Any],
    comparison_artifacts: Mapping[str, Any] | None,
    eval_manifest: Mapping[str, Any],
    budget: Mapping[str, Any],
    *,
    selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze a complete deduplicated panel; over-budget plans remain inspectable.

``budget`` accepts max_total_rollouts, max_unique_features, and optional
identity_features/identity_cases. It never starts execution or drops candidates.
"""
    settings = dict(selection or budget.get("selection", {}))
    methods = list(settings.get("methods", ["event_aligned", "head_full_vocab_kl_fixed_prefix", "mean_readout_activation", "readout_activation_frequency"]))
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("selection.methods must be nonempty and distinct")
    for name, expected in {"topk_tie_break": "feature_id_ascending", "audit_population": "discovery_readout_alive",
                           "audit_sampling": "uniform_without_replacement", "audit_overlap_policy": "keep_memberships_no_redraw"}.items():
        if settings.get(name, expected) != expected:
            raise ValueError(f"Unsupported {name}")
    top_k = _id(settings.get("top_k", 3))
    if not top_k:
        raise ValueError("top_k must be positive")
    comparison = dict(comparison_artifacts or {})
    local_methods = [method for method in methods if method in scores.get("scores", scores) or
                     (scores.get("feature_scores") and method in scores["feature_scores"][0])]
    vectors = score_vectors(scores, local_methods)
    missing = [method for method in methods if method not in vectors]
    if missing:
        vectors.update(score_vectors(comparison, missing))
        if scores.get("scope"):
            validate_score_scope(comparison.get("scope", {}), scores["scope"])
    if set(methods) - set(vectors):
        raise ValueError(f"Missing selected predictors: {set(methods) - set(vectors)}")
    universes = [set(vectors[method]) for method in methods]
    if any(ids != universes[0] for ids in universes[1:]):
        raise ValueError("Predictors must score the same feature universe; missing scores are not zeros")
    universe = universes[0]
    if len(universe) < top_k:
        raise ValueError("Feature universe is smaller than top_k")
    top_ids = {method: sorted(vectors[method], key=lambda i: (-vectors[method][i], i))[:top_k] for method in methods}
    if "alive_feature_ids" in scores:
        alive = sorted({_id(i) for i in scores["alive_feature_ids"]})
    elif scores.get("feature_scores"):
        alive = sorted(_id(row["feature_id"]) for row in scores["feature_scores"]
                       if float(row.get("num_active_readouts", row.get("readout_activation_frequency", 0))) > 0)
    elif "readout_activation_frequency" in vectors:
        alive = sorted(i for i, value in vectors["readout_activation_frequency"].items() if value > 0)
    else:
        raise ValueError("Audit population requires actual discovery readout alive metadata")
    if not set(alive) <= universe:
        raise ValueError("Alive population contains unscored feature IDs")
    audit_count = _id(settings.get("random_audit_features", 6))
    if len(alive) < audit_count:
        raise ValueError("Audit population smaller than requested sample; refusing a smaller draw")
    seed = _id(settings.get("random_seed", 2026))
    audit_ids = random.Random(seed).sample(alive, audit_count)
    memberships: dict[int, list[str]] = {}
    for method, ids in top_ids.items():
        for feature in ids:
            memberships.setdefault(feature, []).append(f"topk:{method}")
    for feature in audit_ids:
        memberships.setdefault(feature, []).append("audit")
    for record in settings.get("additional_candidates", []):
        feature = _id(record["feature_id"])
        if feature not in universe or not record.get("source"):
            raise ValueError("Additional candidate requires a scored feature and source")
        memberships.setdefault(feature, []).append(f"additional:{record['source']}")
    cases = _eval_cases(eval_manifest, synthetic=bool(scores.get("synthetic", False)))
    if scores.get("scope", {}).get("task_ids") is not None:
        validate_score_scope({"task_ids": sorted({row["task_id"] for row in cases})}, {"task_ids": scores["scope"]["task_ids"]})
    mode = eval_manifest.get("mode", "pilot")
    overlap = eval_manifest.get("discovery_eval_overlap", eval_manifest.get("selection_eval_overlap"))
    if overlap not in (None, True, False) or (overlap is not None and not isinstance(overlap, bool)):
        raise ValueError("discovery_eval_overlap must be boolean or explicitly unknown")
    if "selection_eval_overlap" in eval_manifest and "discovery_eval_overlap" in eval_manifest and eval_manifest["selection_eval_overlap"] != overlap:
        raise ValueError("Contradictory discovery/evaluation overlap declarations")
    if mode == "confirmatory":
        if eval_manifest.get("evaluation_labels_previously_used") is not False or eval_manifest.get("frozen_before_pilot") is not True:
            raise ValueError("Confirmatory evaluation requires a pre-pilot frozen, unseen-label split")
        if not eval_manifest.get("split_manifest_hash") or overlap is not False:
            raise ValueError("Confirmatory evaluation requires a disjoint declared split")
        validate_score_scope(scores.get("scope", {}), {"split_manifest_hash": eval_manifest["split_manifest_hash"]})
    elif mode != "pilot":
        raise ValueError("Unknown evaluation mode")
    unique_ids = sorted(memberships)
    candidates = [{"feature_id": feature, "memberships": memberships[feature],
                   "scores": {method: vectors[method][feature] for method in methods}} for feature in unique_ids]
    conditions = [{"condition_id": "raw", "mode": "raw", "feature_id": None, "alpha": None, "num_cases": len(cases)}]
    conditions.extend({"condition_id": f"feature-{feature}", "mode": "intervention", "feature_id": feature,
                       "alpha": 0.0, "num_cases": len(cases)} for feature in unique_ids)
    identity_features = [_id(i) for i in budget.get("identity_features", unique_ids[:1])]
    if len(identity_features) != len(set(identity_features)) or not set(identity_features) <= universe:
        raise ValueError("identity_features must be distinct, valid feature IDs")
    identity_cases = _id(budget.get("identity_cases", len(cases)))
    if identity_features and (identity_cases == 0 or identity_cases > len(cases)):
        raise ValueError("identity_cases must select a nonempty subset of eval_cases")
    conditions.extend({"condition_id": f"identity-{feature}", "mode": "identity", "feature_id": feature,
                       "alpha": 1.0, "num_cases": identity_cases} for feature in identity_features)
    total = sum(row["num_cases"] for row in conditions)
    cap = _id(settings.get("max_unique_features", budget.get("max_unique_features", 24)))
    rollout_cap = _id(budget.get("max_total_rollouts", 100))
    reasons = []
    if len(unique_ids) > cap:
        reasons.append(f"unique_features {len(unique_ids)} exceeds max_unique_features {cap}")
    if total > rollout_cap:
        reasons.append(f"total_rollouts {total} exceeds max_total_rollouts {rollout_cap}")
    plan = {"schema_version": "output_head_rollout_plan_v1", "synthetic": bool(scores.get("synthetic", False)),
            "status": "blocked_budget" if reasons else "planned", "blocked_reasons": reasons,
            "methods": methods, "top_k": top_k, "method_topk": top_ids, "audit_feature_ids": audit_ids,
            "feature_ids": unique_ids, "candidates": candidates, "score_vectors": vectors,
            "feature_universe": sorted(universe), "scope": dict(scores.get("scope", {})),
            "mode": mode, "selection_eval_overlap": overlap,
            "evaluation_labels_previously_used": eval_manifest.get("evaluation_labels_previously_used"),
            "audit": {"population": "discovery_readout_alive", "population_size": len(alive), "population_hash": _hash(alive),
                      "algorithm": "python_random_sample_v1", "seed": seed, "draw_order": audit_ids,
                      "overlap_policy": "keep_memberships_no_redraw", "topk_tie_break": "feature_id_ascending"},
            "conditions": conditions, "eval_cases": cases, "eval_manifest": dict(eval_manifest),
            "num_unique_features": len(unique_ids), "num_eval_cases": len(cases), "total_rollouts": total,
            "budget": {"max_unique_features": cap, "max_total_rollouts": rollout_cap},
            "execute": False, "execution_performed": False}
    plan["plan_hash"] = _hash(plan)
    return plan
