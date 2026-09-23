"""Portable, stdlib-only verification of completed output-head result artifacts.

Historical JSON is read verbatim, never rewritten. Absolute action paths are
resolved relative to the archived ``runs`` tree so an archive can move between
machines. This checks saved evidence; it does not rerun a model or simulator.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(ok, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _pairs(items):
    out = {}
    for key, value in items:
        _require(key not in out, f"Duplicate JSON key: {key}")
        out[key] = value
    return out


def _constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"),
                          object_pairs_hook=_pairs, parse_constant=_constant)
    except OSError as exc:
        raise ValueError(f"Missing or unreadable artifact: {path}") from exc


def _path(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative)
    _require(not parts.is_absolute() and all(p not in {"..", "."} for p in parts.parts)
             and "\\" not in relative, f"Unsafe relative artifact path: {relative}")
    target = root.joinpath(*parts.parts)
    _require(target.resolve().is_relative_to(root.resolve()), f"Artifact escapes root: {relative}")
    _require(target.is_file(), f"Missing artifact: {relative}")
    return target


def _actions_path(root: Path, recorded: str, condition: str) -> Path:
    _require(isinstance(recorded, str) and "\\" not in recorded,
             f"Invalid actions_path: {condition}")
    parts = PurePosixPath(recorded).parts
    _require(".." not in parts and "." not in parts and parts.count("runs") == 1,
             f"Unsafe actions_path: {condition}")
    suffix = parts[parts.index("runs"):]
    _require(len(suffix) >= 4 and suffix[1] == condition and suffix[-1] == "actions.json",
             f"Action path differs from condition directory: {condition}")
    return _path(root, "/".join(suffix))


def _uint(value, label: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
             f"Invalid integer {label}")
    return value


def _case(row: dict, suite=None) -> tuple:
    suite = row.get("suite", suite)
    _require(isinstance(suite, str) and bool(suite), "Missing episode suite")
    state = row.get("initial_state_sha256")
    _require(isinstance(state, str) and re.fullmatch(r"[0-9a-fA-F]{64}", state),
             "Missing/invalid initial_state_sha256")
    return (suite, _uint(row.get("task_id"), "task_id"),
            _uint(row.get("task_episode_idx"), "task_episode_idx"), state.lower())


def _case_map(rows: list, suite=None, outcomes=False) -> dict:
    _require(isinstance(rows, list) and bool(rows), "Empty episode cases")
    out, trials, states = {}, set(), set()
    for row in rows:
        key = _case(row, suite)
        _require(key not in out and key[:3] not in trials and (key[0], key[1], key[3]) not in states,
                 "Duplicate episode or initial state")
        if outcomes:
            _require(isinstance(row.get("success"), bool), "Missing boolean success")
            _require("caught_exception" in row and row["caught_exception"] in (None, "")
                     and not row.get("error"), "Exceptional episode is not a completed outcome")
            _require(row.get("status") in (None, "completed", "complete"), "Incomplete episode")
        out[key] = row
        trials.add(key[:3])
        states.add((key[0], key[1], key[3]))
    return out


def _same_number(actual, expected, label):
    _require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
             and math.isfinite(actual) and math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12),
             f"Inconsistent {label}")


def _mean(values):
    return sum(values) / len(values)


def _verify_sample(root, plan, scores):
    sample = read_json(_path(root, "sample_manifest.json"))
    _require(sample.get("synthetic") is False, "Synthetic sample cannot certify real experiment")
    mapping = fingerprint({"index": sample["source_index_hash"], "generation": sample["generation_spec"],
                           "mapping_version": sample["mapping_version"]})
    _require(mapping == sample.get("mapping_hash"), "Sample mapping hash mismatch")
    _require(bool(sample.get("readouts")), "Empty selected readouts")
    sample_hash = fingerprint({"mapping_hash": mapping, "sample": sample["sample_spec"],
                               "readouts": sample["readouts"]})
    _require(sample_hash == sample.get("sample_hash"), "Sample hash mismatch")
    population = {}
    for row in sample["discovery_episodes"]:
        identity = {key: row.get(key) for key in
                    ("suite", "task_id", "task_episode_idx", "initial_state_sha256")}
        population[fingerprint(identity)] = identity
    discovery_hash = fingerprint([population[key] for key in sorted(population)])
    _require(bool(population) and discovery_hash == sample.get("discovery_manifest_hash"),
             "Discovery population hash mismatch")
    expected = {"sample_manifest_hash": sample_hash, "discovery_manifest_hash": discovery_hash,
                "split_manifest_hash": sample.get("split_manifest_hash")}
    for key, value in expected.items():
        _require(scores.get("scope", {}).get(key) == value, f"Score/sample {key} mismatch")
    _require(plan.get("scope") == scores.get("scope"), "Frozen plan/score scopes differ")
    discovery_ids = {(r["suite"], r["task_id"], r["task_episode_idx"]) for r in sample["discovery_episodes"]}
    evaluation_ids = {key[:3] for key in _case_map(plan["eval_cases"])}
    _require(bool(discovery_ids & evaluation_ids) == plan.get("selection_eval_overlap"),
             "Declared discovery/evaluation overlap differs from case IDs")
    return sample


def _verify_readout_blobs(root, scores, sample):
    manifest_path = _path(root, "readouts/manifest.json")
    _require(sha256_file(manifest_path) == scores["identity"].get("cache_manifest_hash"),
             "Readout cache manifest hash mismatch")
    manifest = read_json(manifest_path)
    _require(manifest.get("synthetic") is False
             and manifest.get("sample_manifest_hash") == sample["sample_hash"],
             "Readout cache sample provenance mismatch")
    _require(manifest.get("num_readouts") == len(sample["readouts"]), "Readout cache count mismatch")
    shards = manifest.get("shards", [])
    _require(isinstance(shards, list) and bool(shards), "Missing readout cache shards")
    seen, num_rows = set(), 0
    for shard in shards:
        name = shard.get("path")
        _require(isinstance(name, str) and re.fullmatch(r"shard_[0-9]+\.pt", name)
                 and name not in seen, "Invalid or duplicate readout shard path")
        seen.add(name)
        path = _path(root, f"readouts/{name}")
        _require(sha256_file(path) == shard.get("sha256"), "Readout shard hash mismatch")
        count = _uint(shard.get("num_rows"), "readout num_rows")
        _require(count > 0, "Empty readout shard")
        num_rows += count
    _require(num_rows == manifest["num_readouts"], "Readout shard row totals differ from manifest")


def _verify_analysis(root, plan, results, cases, protocol_id, revision):
    paired = read_json(_path(root, "analysis/paired_effects.json"))
    analysis = read_json(_path(root, "analysis/analysis.json"))
    _path(root, "analysis/report.md")
    features = plan["feature_ids"]
    raw = results["raw"]["case_map"]
    protocol = paired.get("protocol", {})
    _require(protocol.get("protocol_id") == protocol_id and protocol.get("code_revision") == revision,
             "Analysis protocol provenance mismatch")
    _require(set(_case_map(protocol.get("eval_cases", []))) == set(cases)
             and sorted(protocol.get("expected_feature_ids", [])) == sorted(features),
             "Analysis protocol cases/features differ from plan")
    for obj in (paired, analysis):
        _require(obj.get("synthetic") is False and obj.get("protocol_id") == protocol_id
                 and obj.get("protocol") == protocol, "Paired/analysis protocol mismatch")
        effect_ids = [row["feature_id"] for row in obj.get("feature_effects", [])]
        _require(sorted(effect_ids) == sorted(features), "Analysis feature panel mismatch")
    _require(sorted(paired.get("feature_ids", [])) == sorted(features)
             and paired.get("num_shared_cases") == len(cases), "Paired cases/features mismatch")
    pair_index = {}
    for row in paired.get("paired_rows", []):
        feature = row.get("feature_id")
        key = _case(row)
        _require(feature in features and key in cases and (feature, key) not in pair_index,
                 "Unexpected or duplicate paired row")
        edited = results[f"feature-{feature}"]["case_map"][key]
        _require(row.get("raw_success") is raw[key]["success"]
                 and row.get("edit_success") is edited["success"]
                 and row.get("protocol_id") == protocol_id, "Paired outcomes differ from saved rollouts")
        seed = raw[key].get("episode_seed", raw[key].get("seed", results["raw"]["payload"].get("seed")))
        _require(row.get("seed") == seed, "Paired episode seed mismatch")
        _same_number(row.get("drop"), int(raw[key]["success"]) - int(edited["success"]), "paired drop")
        pair_index[feature, key] = row
    _require(len(pair_index) == len(features) * len(cases), "Missing paired outcomes")
    drops = {}
    for feature in features:
        tasks = defaultdict(list)
        for key in cases:
            tasks[key[:2]].append(pair_index[feature, key])
        drops[feature] = _mean([_mean([r["drop"] for r in rows]) for rows in tasks.values()])
        for obj in (paired, analysis):
            effect = next(row for row in obj["feature_effects"] if row["feature_id"] == feature)
            _require(effect.get("num_valid_pairs") == len(cases)
                     and effect.get("num_tasks") == len(tasks), "Effect pair/task count mismatch")
            _same_number(effect.get("drop"), drops[feature], "feature mean drop")
            _same_number(effect.get("sr_raw"), _mean([_mean([int(r["raw_success"]) for r in rows])
                                                       for rows in tasks.values()]), "feature raw success")
            _same_number(effect.get("sr_edit"), _mean([_mean([int(r["edit_success"]) for r in rows])
                                                        for rows in tasks.values()]), "feature edited success")
    summaries = {}
    for row in analysis.get("predictor_comparison", []):
        method = row["predictor"]
        _require(method in plan["method_topk"] and row.get("topk_feature_ids") == plan["method_topk"][method],
                 "Analysis top-K differs from frozen selection")
        expected_drop = _mean([drops[f] for f in plan["method_topk"][method]])
        _same_number(row.get("topk_mean_drop"), expected_drop, "method top-K mean drop")
        if row.get("population_scope") == "topk_union":
            _require(method not in summaries, "Duplicate method summary")
            summaries[method] = {"feature_ids": row["topk_feature_ids"], "mean_success_rate_drop": expected_drop,
                                 "mean_success_rate_drop_ci": row.get("topk_mean_drop_ci"),
                                 "spearman_topk_union": row.get("spearman")}
    _require(set(summaries) == set(plan["methods"]), "Missing method analysis")
    head, event = "head_full_vocab_kl_fixed_prefix", "event_aligned"
    contrast = None
    for row in analysis.get("topk_contrasts", []):
        first, second = row["first"], row["second"]
        _require(first in summaries and second in summaries, "Unknown contrast method")
        difference = summaries[first]["mean_success_rate_drop"] - summaries[second]["mean_success_rate_drop"]
        _same_number(row.get("difference"), difference, "method contrast")
        if {first, second} == {head, event}:
            _require(contrast is None, "Duplicate Head-KL/Event contrast")
            _require(row.get("status") in {"ok", "degenerate_bootstrap"}
                     and isinstance(row.get("low"), (int, float))
                     and isinstance(row.get("high"), (int, float))
                     and math.isfinite(row["low"]) and math.isfinite(row["high"])
                     and row["low"] <= row["high"], "Invalid stored contrast interval")
            reverse = first != head
            contrast = {"first": head, "second": event, "difference": -difference if reverse else difference,
                        "low": -row["high"] if reverse else row["low"],
                        "high": -row["low"] if reverse else row["high"], "status": row.get("status"),
                        "valid_replicates": row.get("valid_replicates"),
                        "confidence_level": analysis.get("uncertainty", {}).get("confidence_level")}
    if head in summaries and event in summaries:
        _require(contrast is not None, "Missing Head-KL/Event contrast")
    groups_path = root / "analysis/task_groups_v1.json"
    if groups_path.exists():
        groups = read_json(groups_path)
        _path(root, "analysis/task_groups_v1.md")
        _require(groups.get("plan_hash") == plan["plan_hash"] and groups.get("protocol_id") == protocol_id
                 and groups.get("method_topk") == plan["method_topk"]
                 and sorted(groups.get("feature_ids", [])) == sorted(features), "Task group provenance mismatch")
        for relative, expected in groups.get("source_artifact_sha256", {}).items():
            _require(sha256_file(_path(root, relative)) == expected, "Task group source hash mismatch")
    return {"methods": summaries, "head_minus_event": contrast,
            "uncertainty": analysis.get("uncertainty"), "protocol_scope": protocol}


def audit_results(result_root: Path) -> dict:
    """Verify a complete result tree on any machine without GPU dependencies."""
    root = Path(result_root).resolve()
    plan = read_json(_path(root, "plan/rollout_plan.json"))
    _require(plan.get("plan_hash") == fingerprint({k: v for k, v in plan.items() if k != "plan_hash"}),
             "Frozen plan hash mismatch")
    _require(plan.get("status") == "planned" and plan.get("synthetic") is False, "Not a real approved plan")
    evaluation_path = _path(root, "plan/eval_manifest.json")
    _require(read_json(evaluation_path) == plan.get("eval_manifest")
             and sha256_file(evaluation_path) == plan.get("eval_manifest_hash"), "Evaluation manifest mismatch")
    candidates_path = _path(root, "plan/candidates.jsonl")
    candidates = [json.loads(line, object_pairs_hook=_pairs, parse_constant=_constant)
                  for line in candidates_path.read_text().splitlines() if line.strip()]
    _require(candidates == plan.get("candidates"), "Candidate artifact differs from frozen plan")
    features = plan["feature_ids"]
    _require(features and len(features) == len(set(features))
             and sorted(row["feature_id"] for row in candidates) == sorted(features), "Candidate feature IDs mismatch")
    for value in features:
        _uint(value, "feature_id")
    scores_path = _path(root, "scores/scores.json")
    _require(sha256_file(scores_path) == plan.get("score_artifact_hash"), "Score artifact hash mismatch")
    scores = read_json(scores_path)
    score_manifest = read_json(_path(root, "scores/score_manifest.json"))
    _require(scores.get("synthetic") is False and score_manifest.get("synthetic") is False,
             "Synthetic score artifact")
    _require(scores.get("identity_hash") == fingerprint(scores["identity"]), "Score numerical identity hash mismatch")
    for key in ("identity", "identity_hash", "scope"):
        _require(score_manifest.get(key) == scores.get(key), f"Score manifest {key} mismatch")
    sample = _verify_sample(root, plan, scores)
    _verify_readout_blobs(root, scores, sample)
    cases = _case_map(plan["eval_cases"])
    _require(plan.get("num_eval_cases") == len(cases) and plan.get("num_unique_features") == len(features),
             "Plan case/feature count mismatch")
    conditions = plan["conditions"]
    names = [row["condition_id"] for row in conditions]
    _require(names and len(names) == len(set(names)) and names.count("raw") == 1, "Duplicate/missing conditions")
    _require(sum(row["num_cases"] for row in conditions) == plan["total_rollouts"], "Plan rollout total mismatch")
    _require({p.parent.name for p in (root / "runs").glob("*/result.json")} == set(names),
             "Missing or unexpected completed condition")
    raw = read_json(_path(root, "runs/raw/result.json"))
    identity, revision, protocol_id = raw["head_sae_identity"], raw["code"]["commit"], raw["protocol_id"]
    _require(isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision), "Invalid experiment code revision")
    _require(identity and all(scores["identity"].get(k) == v for k, v in identity.items()),
             "Scoring and rollout numerical identities differ")
    for name, key in (("output_head_manifest.json", "head_manifest_hash"),
                      ("output_head.safetensors", "head_weights_hash")):
        _require(sha256_file(_path(root, f"head/{name}")) == identity.get(key),
                 f"Preserved output head artifact hash mismatch: {name}")
    expected_protocol = fingerprint({"plan_hash": plan["plan_hash"], "identity": identity,
                                     "code_revision": revision, "effective_policy": raw["run_config"],
                                     "per_episode_seed": "sha256_seed_task_trial_v1"})
    _require(protocol_id == expected_protocol, "Rollout protocol hash mismatch")
    for relative in ("validation/runtime_parity.json", "validation/edit_parity.json"):
        parity = read_json(_path(root, relative))
        _require(parity.get("status") == "passed" and parity.get("synthetic") is False
                 and parity.get("identity") == identity
                 and parity.get("identity_fingerprint") == fingerprint(identity), "Runtime parity provenance mismatch")
        _require(parity.get("checks") and all(row.get("status") == "passed" for row in parity["checks"].values()),
                 "Failed or empty runtime parity checks")
    results, timings = {}, []
    for condition in conditions:
        name = condition["condition_id"]
        _require(re.fullmatch(r"(?:raw|feature-[0-9]+|identity-[0-9]+)", name), "Unsafe condition ID")
        payload = read_json(_path(root, f"runs/{name}/result.json"))
        _require(payload.get("result_hash") == fingerprint({k: v for k, v in payload.items() if k != "result_hash"}),
                 f"Result hash mismatch: {name}")
        expected = {"status": "completed", "synthetic": False, "completed_rollouts": len(cases),
                    "mode": condition["mode"], "feature_id": condition["feature_id"], "alpha": condition["alpha"],
                    "protocol_id": protocol_id, "head_sae_identity": identity,
                    "code": {"commit": revision, "dirty": False}, "run_config": raw["run_config"]}
        _require(condition["num_cases"] == len(cases)
                 and all(payload.get(k) == v for k, v in expected.items()), f"Result condition/provenance mismatch: {name}")
        for key in ("suite", "seed", "model_checkpoint", "model_revision", "model_code_revision", "sae_sha256", "layer_idx", "hook_start_step"):
            _require(key in raw and payload.get(key) == raw[key], f"Cross-condition {key} mismatch")
        rows = _case_map(payload.get("episodes", []), payload.get("suite"), outcomes=True)
        _require(set(rows) == set(cases), f"Result cases differ from approved states: {name}")
        for row in rows.values():
            _require(row.get("protocol_id", protocol_id) == protocol_id, "Episode protocol mismatch")
        action_path = _actions_path(root, payload.get("actions_path"), name)
        _require(sha256_file(action_path) == payload.get("actions_sha256"), f"Action artifact hash mismatch: {name}")
        if condition["mode"] != "raw":
            metrics = payload.get("hook_metrics") or {}
            _require(metrics.get("num_forwards", 0) > 0, "Missing executed feature hook evidence")
            if condition["alpha"] == 0:
                _require(metrics.get("post_intervention_active_feature_values") == 0
                         and metrics.get("post_intervention_max_feature_activation") == 0, "Feature suppression not verified")
        elapsed = payload.get("elapsed_seconds")
        _require(isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool)
                 and math.isfinite(elapsed) and elapsed >= 0, "Missing measured condition time")
        timings.append({"condition_id": name, "elapsed_seconds": elapsed, "num_rollouts": len(rows)})
        results[name] = {"payload": payload, "case_map": rows, "actions_path": action_path}
    _require(sorted(r["feature_id"] for r in conditions if r["mode"] == "intervention") == sorted(features),
             "Intervention conditions differ from feature panel")
    identities = [r for r in conditions if r["mode"] == "identity"]
    _require(bool(identities), "Missing identity condition")
    for condition in identities:
        item = results[condition["condition_id"]]
        _require(item["payload"].get("identity_check") == "passed", "Identity check not passed")
        _require(read_json(item["actions_path"]) == read_json(results["raw"]["actions_path"]),
                 "Raw/identity action sequences differ")
        _require(all(item["case_map"][k]["success"] == results["raw"]["case_map"][k]["success"] for k in cases),
                 "Raw/identity outcomes differ")
    for name, item in results.items():
        for key in cases:
            raw_row = results["raw"]["case_map"][key]
            edit_row = item["case_map"][key]
            raw_seed = raw_row.get("episode_seed", raw_row.get("seed", raw["seed"]))
            edit_seed = edit_row.get("episode_seed", edit_row.get("seed", item["payload"]["seed"]))
            _require(raw_seed == edit_seed, f"Paired episode RNG seed mismatch: {name}")
    summary = _verify_analysis(root, plan, results, cases, protocol_id, revision)
    baseline_tasks = defaultdict(list)
    for key, row in results["raw"]["case_map"].items():
        baseline_tasks[key[:2]].append(int(row["success"]))
    return {"status": "verified", "experiment_code_revision": revision, "plan_hash": plan["plan_hash"],
            "protocol_id": protocol_id, "total_rollouts": plan["total_rollouts"], "num_conditions": len(conditions),
            "num_eval_cases": len(cases), "num_features": len(features),
            "condition_elapsed_seconds_sum": sum(r["elapsed_seconds"] for r in timings),
            "timing_scope": "sum_of_measured_condition_elapsed_seconds_not_overall_wall_time",
            "conditions": timings, "comparison": summary,
            "baseline_success_rate": _mean([_mean(rows) for rows in baseline_tasks.values()]),
            "bootstrap_reexecuted": False,
            "verification_scope": "saved_artifact_integrity_and_consistency_not_model_reexecution"}
