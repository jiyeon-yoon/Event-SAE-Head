"""Paired outcome assembly and fixed-panel, shared-state bootstrap analysis.

All statistics use episode outcomes, never token counts. The same resampled
initial states are used for every feature and predictor in each bootstrap draw.
No model, simulator, SciPy or network imports are needed.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .candidates import _id, _vector, score_vectors


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty population")
    return sum(values) / len(values)


def _condition_payload(result: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if isinstance(result, Mapping):
        metadata = dict(result)
        rows = metadata.get("episodes", metadata.get("episode_results"))
        if rows is None:
            path = metadata.get("episode_results_path")
            if not path:
                raise ValueError("Result requires episodes or episode_results_path")
            with Path(path).open(encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
    else:
        metadata, rows = {}, result
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("Condition has no episode outcomes")
    if metadata.get("status") not in (None, "completed", "complete"):
        raise ValueError("Condition is not completed")
    if metadata.get("completed_rollouts", len(rows)) != len(rows):
        raise ValueError("Completed rollout count disagrees with episode records")
    return metadata, [dict(row) for row in rows]


def _shared_protocol(metadata: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only non-policy output settings from existing CLI run metadata.

The old protocol_fingerprint includes the feature ID and is intentionally not
used as the cross-condition key. Its shared fields are checked independently.
"""
    shared = {}
    for name in ("model_checkpoint", "model_revision", "model_code_revision"):
        value = metadata.get(name, protocol.get(name))
        if name in metadata and name in protocol and metadata[name] != protocol[name]:
            raise ValueError(f"Result does not match expected {name}")
        if value is not None:
            shared[name] = value
    code = metadata.get("code", {})
    revision = code.get("commit", metadata.get("code_revision", protocol.get("code_revision")))
    if revision is not None and "code_revision" in protocol and revision != protocol["code_revision"]:
        raise ValueError("Result does not match expected code_revision")
    if revision is not None:
        shared["code_revision"] = revision
    if code.get("dirty") is True:
        raise ValueError("Refusing result from dirty code")
    if metadata.get("run_config") is not None:
        config = dict(metadata["run_config"])
        config.pop("logging", None)
        config.pop("sae_collect", None)
        shared["run_config"] = config
    return shared


def _normalize_condition(
    metadata: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> tuple[dict[tuple[Any, ...], dict[str, Any]], str, dict[str, Any]]:
    shared = _shared_protocol(metadata, protocol)
    if not protocol.get("synthetic", False) and any(name not in shared for name in ("model_checkpoint", "model_revision", "model_code_revision", "code_revision")):
        raise ValueError("Real outcomes require model and code provenance")
    declared_protocol = metadata.get("protocol_id", protocol.get("protocol_id"))
    if not declared_protocol and not shared:
        raise ValueError("Missing condition-independent rollout protocol")
    protocol_id = str(declared_protocol or _fingerprint(shared))
    output = {}
    trial_keys = set()
    state_keys = set()
    for record in rows:
        if record.get("caught_exception") not in (None, "") or record.get("error"):
            raise ValueError("Runtime exception is not a task failure")
        if "caught_exception" not in record:
            raise ValueError("Missing caught_exception field; completion is not verified")
        if not isinstance(record.get("success"), bool):
            raise ValueError("Episode success must be a boolean")
        if record.get("status") not in (None, "completed", "complete"):
            raise ValueError("Incomplete episode outcome")
        suite = record.get("suite", metadata.get("suite", metadata.get("task_suite", protocol.get("suite"))))
        seed = record.get("episode_seed", record.get("seed", metadata.get("seed", protocol.get("seed"))))
        if suite is None or seed is None:
            raise ValueError("Pairing requires explicit suite and seed")
        if "suite" in protocol and suite != protocol["suite"]:
            raise ValueError("Result suite differs from expected protocol")
        state_hash = record.get("initial_state_sha256")
        if not isinstance(state_hash, str) or not state_hash:
            raise ValueError("Pairing requires initial_state_sha256")
        if not protocol.get("synthetic", False) and not re.fullmatch(r"[0-9a-fA-F]{64}", state_hash):
            raise ValueError("Real initial_state_sha256 must be a SHA-256 digest")
        state_hash = state_hash.lower()
        task, trial = _id(record["task_id"]), _id(record["task_episode_idx"])
        row_protocol = record.get("protocol_id", protocol_id)
        if row_protocol != protocol_id:
            raise ValueError("Row and condition protocol disagree")
        key = (str(suite), task, trial, state_hash, _id(seed), protocol_id)
        trial_key = (str(suite), task, trial)
        state_key = (str(suite), task, state_hash)
        if key in output or trial_key in trial_keys or state_key in state_keys:
            raise ValueError("Duplicate task/initial-state episode outcome")
        trial_keys.add(trial_key)
        state_keys.add(state_key)
        output[key] = {**record, "suite": str(suite), "task_id": task, "task_episode_idx": trial,
                       "seed": _id(seed), "protocol_id": protocol_id}
    return output, protocol_id, shared


def assemble_paired_effects(
    raw_result: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    feature_results: Mapping[int | str, Mapping[str, Any] | Sequence[Mapping[str, Any]]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Pair complete conditions and calculate signed, equally task-weighted labels.

For legacy CLI summaries, ``episode_results_path`` is read and condition-wide
suite/seed/provenance is applied to those episode records. Synthetic list inputs
must be explicitly marked in protocol. Missing pairs always raise in the MVP.
"""
    if protocol.get("allow_missing_pairs", False):
        raise ValueError("MVP requires complete shared pairs; exclusions need a new explicit protocol")
    if protocol.get("require_declared_selection_eval_overlap", False) and not isinstance(protocol.get("selection_eval_overlap"), bool):
        raise ValueError("Declare selection_eval_overlap before analyzing paired results")
    if protocol.get("mode") == "confirmatory":
        if protocol.get("evaluation_labels_previously_used") is not False or protocol.get("selection_eval_overlap") is not False:
            raise ValueError("Confirmatory outcomes require disjoint, unused evaluation labels")
    raw_meta, raw_rows = _condition_payload(raw_result)
    if raw_meta.get("mode") not in (None, "raw"):
        raise ValueError("Primary baseline must be raw, not reconstruction/identity")
    raw, protocol_id, shared = _normalize_condition(raw_meta, raw_rows, protocol)
    cases = sorted(raw)
    if protocol.get("eval_cases") is not None:
        expected_cases = []
        for case in protocol["eval_cases"]:
            suite = case.get("suite", protocol.get("suite"))
            if suite is None or not case.get("initial_state_sha256"):
                raise ValueError("Expected eval_cases require suite and initial-state hashes")
            expected_cases.append((str(suite), _id(case["task_id"]), _id(case["task_episode_idx"]), case["initial_state_sha256"].lower()))
        if len(set(expected_cases)) != len(expected_cases) or set(expected_cases) != {key[:4] for key in cases}:
            raise ValueError("Result cases differ from the frozen evaluation manifest")
    features = sorted(_id(i) for i in feature_results)
    if len(set(features)) != len(features) or not features:
        raise ValueError("Feature results must contain distinct feature IDs")
    if protocol.get("expected_feature_ids") is not None and features != sorted(_id(i) for i in protocol["expected_feature_ids"]):
        raise ValueError("Feature result set differs from the approved panel")
    normalized_inputs = {_id(i): value for i, value in feature_results.items()}
    effects, paired_rows = [], []
    feature_condition_signature: dict[str, Any] | None = None
    for feature in features:
        metadata, rows = _condition_payload(normalized_inputs[feature])
        if metadata.get("feature_id", feature) != feature or metadata.get("mode") not in (None, "intervention"):
            raise ValueError("Wrong feature or mode in intervention result")
        if float(metadata.get("alpha", protocol.get("alpha", 0.0))) != 0.0:
            raise ValueError("Main effects require single-feature alpha=0")
        if not protocol.get("synthetic", False):
            signature = {}
            for name in ("sae_sha256", "layer_idx", "hook_start_step"):
                value = metadata.get(name, protocol.get(name))
                if value is None:
                    raise ValueError(f"Intervention missing {name}")
                signature[name] = value
                if name in protocol and value != protocol[name]:
                    raise ValueError(f"Intervention differs from expected {name}")
            if feature_condition_signature is not None and signature != feature_condition_signature:
                raise ValueError("Interventions have different SAE/layer/hook protocols")
            feature_condition_signature = signature
            metrics = metadata.get("hook_metrics", {})
            if metrics.get("num_forwards", 0) <= 0:
                raise ValueError("Intervention hook did not run")
            if metrics.get("post_intervention_active_feature_values") != 0 or metrics.get("post_intervention_max_feature_activation") != 0:
                raise ValueError("Target latent suppression was not verified")
        edited, edit_protocol_id, edit_shared = _normalize_condition(metadata, rows, protocol)
        if edit_protocol_id != protocol_id or edit_shared != shared:
            raise ValueError("Raw/edit model or rollout protocol mismatch")
        if set(edited) != set(raw):
            raise ValueError(f"Raw/edit pairing mismatch for feature {feature}: missing={len(set(raw)-set(edited))}, extra={len(set(edited)-set(raw))}")
        by_task: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
        counts = {"n_success_to_failure": 0, "n_failure_to_success": 0,
                  "n_success_to_success": 0, "n_failure_to_failure": 0}
        for key in cases:
            a, b = int(raw[key]["success"]), int(edited[key]["success"])
            by_task[(key[0], key[1])].append((a, b))
            counts[{(1, 0): "n_success_to_failure", (0, 1): "n_failure_to_success",
                    (1, 1): "n_success_to_success", (0, 0): "n_failure_to_failure"}[(a, b)]] += 1
            paired_rows.append({"feature_id": feature, "suite": key[0], "task_id": key[1],
                                "task_episode_idx": key[2], "initial_state_sha256": key[3], "seed": key[4],
                                "protocol_id": protocol_id, "raw_success": bool(a), "edit_success": bool(b), "drop": a-b})
        task_effects = [{"suite": task[0], "task_id": task[1], "num_valid_pairs": len(pairs),
                         "sr_raw": _mean([a for a, _ in pairs]), "sr_edit": _mean([b for _, b in pairs]),
                         "drop": _mean([a-b for a, b in pairs]),
                         "outcome_switch_rate": _mean([float(a != b) for a, b in pairs])}
                        for task, pairs in sorted(by_task.items())]
        drop = _mean([row["drop"] for row in task_effects])
        effects.append({"feature_id": feature, "alpha": 0.0, "num_tasks": len(task_effects), "num_valid_pairs": len(cases),
                        "sr_raw": _mean([row["sr_raw"] for row in task_effects]), "sr_edit": _mean([row["sr_edit"] for row in task_effects]),
                        "drop": drop, "delta_sr": -drop, **counts,
                        "outcome_switch_rate": _mean([row["outcome_switch_rate"] for row in task_effects]),
                        "selection_eval_overlap": protocol.get("selection_eval_overlap"),
                        "evaluation_labels_previously_used": protocol.get("evaluation_labels_previously_used"),
                        "protocol_id": protocol_id, "task_effects": task_effects})
    return {"schema_version": "output_head_paired_effects_v1", "synthetic": bool(protocol.get("synthetic", False)),
            "protocol_id": protocol_id, "feature_effects": effects, "paired_rows": paired_rows,
            "feature_ids": features, "num_shared_cases": len(cases), "population_scope": "fixed_evaluated_panel",
            "protocol": dict(protocol)}


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + end - 1) / 2 + 1
        for index in order[start:end]:
            ranks[index] = average
        start = end
    return ranks


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Average-tie-rank Spearman; constants and fewer than two rows are undefined."""
    if len(x) != len(y):
        raise ValueError("Correlation vectors have different lengths")
    if any(not math.isfinite(float(v)) for v in (*x, *y)):
        raise ValueError("Correlation requires finite values")
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = _mean(rx), _mean(ry)
    numerator = sum((a-mx)*(b-my) for a, b in zip(rx, ry))
    denominator = math.sqrt(sum((a-mx)**2 for a in rx) * sum((b-my)**2 for b in ry))
    return max(-1.0, min(1.0, numerator / denominator))


def _interval(values: Sequence[float | None], level: float) -> dict[str, Any]:
    valid = sorted(float(v) for v in values if v is not None)
    if not valid:
        return {"low": None, "high": None, "status": "undefined", "valid_replicates": 0, "undefined_replicates": len(values)}
    def quantile(q: float) -> float:
        index = (len(valid)-1)*q
        lower = int(index)
        upper = min(lower+1, len(valid)-1)
        return valid[lower] + (valid[upper]-valid[lower])*(index-lower)
    low, high = quantile((1-level)/2), quantile(1-(1-level)/2)
    return {"low": low, "high": high, "status": "degenerate_bootstrap" if low == high else "ok",
            "valid_replicates": len(valid), "undefined_replicates": len(values)-len(valid)}


def _bootstrap_drops(paired: Mapping[str, Any], replicates: int, seed: int) -> dict[int, list[float]]:
    """Use one within-task state resample across all features in each replicate."""
    rows_by_feature: dict[int, dict[tuple[Any, ...], float]] = defaultdict(dict)
    for row in paired["paired_rows"]:
        key = (row["suite"], row["task_id"], row["task_episode_idx"], row["initial_state_sha256"], row["seed"], row["protocol_id"])
        feature = _id(row["feature_id"])
        if key in rows_by_feature[feature]:
            raise ValueError("Duplicate pair in bootstrap input")
        rows_by_feature[feature][key] = float(row["drop"])
    if not rows_by_feature:
        raise ValueError("No paired rows to bootstrap")
    first_keys = set(next(iter(rows_by_feature.values())))
    if any(set(rows) != first_keys for rows in rows_by_feature.values()):
        raise ValueError("Every feature must share exactly the same paired case universe")
    by_task: dict[tuple[str, int], list[tuple[Any, ...]]] = defaultdict(list)
    for key in sorted(first_keys):
        by_task[(key[0], key[1])].append(key)
    rng = random.Random(seed)
    draws = {feature: [] for feature in rows_by_feature}
    for _ in range(replicates):
        selected = [[keys[rng.randrange(len(keys))] for _ in keys] for keys in by_task.values()]
        for feature, values in rows_by_feature.items():
            draws[feature].append(_mean([_mean([values[key] for key in keys]) for keys in selected]))
    return draws


def analyze_prediction(
    scores: Mapping[str, Any], paired_effects: Mapping[str, Any], analysis_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare frozen rankings and audit correlations with shared paired CIs.

Pass the frozen plan as analysis_config['plan'] (or scores itself may be a plan).
No top-K is recomputed after dropping unevaluated features: incomplete coverage
produces a null top-K effect with a list of missing feature IDs.
"""
    config = dict(analysis_config)
    if config.get("primary_label", "success_rate_drop") != "success_rate_drop":
        raise ValueError("MVP primary label is fixed to success_rate_drop")
    replicates = _id(config.get("bootstrap_replicates", 2000))
    level = float(config.get("confidence_level", 0.95))
    if replicates == 0 or not 0 < level < 1:
        raise ValueError("Positive bootstrap count and confidence_level in (0,1) required")
    seed = _id(config.get("bootstrap_seed", 2026))
    plan = config.get("plan", scores if "method_topk" in scores else {})
    methods = list(plan.get("methods", config.get("methods", scores.get("scores", {}))))
    if "score_vectors" in plan:
        vectors = {method: _vector(plan["score_vectors"][method]) for method in methods}
    else:
        vectors = score_vectors(scores, methods)
    if not methods or set(methods) != set(vectors):
        raise ValueError("Analysis requires all selected predictor vectors")
    effects = {_id(row["feature_id"]): dict(row) for row in paired_effects["feature_effects"]}
    if not effects:
        raise ValueError("No evaluated features")
    bootstrap = _bootstrap_drops(paired_effects, replicates, seed)
    if set(effects) != set(bootstrap):
        raise ValueError("Effect summaries and paired rows refer to different features")
    for feature, effect in effects.items():
        interval = _interval(bootstrap[feature], level)
        effect.update({"drop_ci_low": interval["low"], "drop_ci_high": interval["high"], "bootstrap_status": interval["status"]})
    top_k = _id(plan.get("top_k", config.get("top_k", 3)))
    if not top_k:
        raise ValueError("top_k must be positive")
    top_sets = {method: [_id(i) for i in plan.get("method_topk", {}).get(method, sorted(vectors[method], key=lambda i: (-vectors[method][i], i))[:top_k])]
                for method in methods}
    for method, ids in top_sets.items():
        if len(ids) != len(set(ids)) or not set(ids) <= set(vectors[method]):
            raise ValueError("Frozen top-K must have distinct scored feature IDs")
    audit = [_id(i) for i in plan.get("audit_feature_ids", config.get("audit_feature_ids", []))]
    groups = {"topk_union": sorted({i for ids in top_sets.values() for i in ids}), "audit_sample": sorted(set(audit))}
    comparisons, top_draws = [], {}
    for method in methods:
        missing_top = sorted(set(top_sets[method])-set(effects))
        full_top = len(top_sets[method]) == top_k and not missing_top
        top_value = _mean([effects[i]["drop"] for i in top_sets[method]]) if full_top else None
        top_ci = {"low": None, "high": None, "status": "incomplete_coverage"}
        if full_top:
            top_draws[method] = [_mean([bootstrap[i][r] for i in top_sets[method]]) for r in range(replicates)]
            top_ci = _interval(top_draws[method], level)
        for group, ids in groups.items():
            evaluated = [i for i in ids if i in effects and i in vectors[method]]
            x, y = [vectors[method][i] for i in evaluated], [effects[i]["drop"] for i in evaluated]
            correlation = spearman_correlation(x, y)
            correlations = [spearman_correlation(x, [bootstrap[i][r] for i in evaluated]) for r in range(replicates)]
            corr_ci = _interval(correlations, level)
            comparisons.append({"predictor": method, "population_scope": group,
                                "evaluated_feature_count": len(evaluated), "requested_feature_count": len(ids),
                                "missing_feature_ids": sorted(set(ids)-set(evaluated)),
                                "spearman": correlation, "correlation_status": "undefined" if correlation is None else "ok",
                                "correlation_ci_low": corr_ci["low"], "correlation_ci_high": corr_ci["high"],
                                "correlation_bootstrap": corr_ci,
                                "score_tie_fraction": 1-len(set(x))/len(x) if x else None,
                                "label_tie_fraction": 1-len(set(y))/len(y) if y else None,
                                "top_k": top_k, "topk_coverage": (len(top_sets[method])-len(missing_top))/top_k,
                                "topk_feature_ids": top_sets[method], "missing_topk_feature_ids": missing_top,
                                "topk_mean_drop": top_value, "topk_mean_drop_ci": [top_ci["low"], top_ci["high"]],
                                "topk_bootstrap_status": top_ci["status"],
                                "feature_universe": plan.get("feature_universe", sorted(vectors[method]))})
    contrasts = []
    for position, first in enumerate(methods):
        for second in methods[position+1:]:
            if first not in top_draws or second not in top_draws:
                contrasts.append({"first": first, "second": second, "difference": None, "status": "incomplete_coverage"})
                continue
            draws = [a-b for a, b in zip(top_draws[first], top_draws[second])]
            contrast = _interval(draws, level)
            difference = _mean([effects[i]["drop"] for i in top_sets[first]]) - _mean([effects[i]["drop"] for i in top_sets[second]])
            contrasts.append({"first": first, "second": second, "difference": difference, **contrast})
    return {"schema_version": "output_head_prediction_analysis_v1", "synthetic": bool(paired_effects.get("synthetic", False)),
            "primary_label": "success_rate_drop", "feature_effects": [effects[i] for i in sorted(effects)],
            "predictor_comparison": comparisons, "topk_contrasts": contrasts,
            "uncertainty": {"bootstrap_replicates": replicates, "bootstrap_seed": seed, "confidence_level": level,
                            "resampling": "shared_initial_states_within_task", "task_scope": "fixed_tasks",
                            "feature_scope": "fixed_evaluated_panel", "score_uncertainty_included": False},
            "protocol_id": paired_effects["protocol_id"], "protocol": paired_effects.get("protocol", {})}
