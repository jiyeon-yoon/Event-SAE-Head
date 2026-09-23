"""A relocated real-schema fixture is auditable without model dependencies."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from event_sae.research.output_head.results import assemble_paired_effects, analyze_prediction


VERIFY_PATH = Path(__file__).resolve().parents[1] / "scripts/openvla/headbundle_verify.py"
spec = importlib.util.spec_from_file_location("headbundle_verify_test", VERIFY_PATH)
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)
HEAD = "head_full_vocab_kl_fixed_prefix"
EVENT = "event_aligned"


def _write(root, relative, payload):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path


def _resign(path):
    payload = verify.read_json(path)
    payload.pop("result_hash", None)
    payload["result_hash"] = verify.fingerprint(payload)
    _write(path.parent, path.name, payload)


def make_result_fixture(root, revision="b" * 40, plan_config_hash=None):
    """Use actual outcome/analysis functions, unequal task sizes and old absolute paths."""
    root.mkdir(parents=True)
    cases = [{"suite": "libero_spatial", "task_id": t, "task_episode_idx": trial,
              "initial_state_sha256": verify.fingerprint([t, trial])} for t, trial in [(0, 0), (0, 1), (1, 0)]]
    discovery = [{"suite": "libero_spatial", "task_id": 0, "task_episode_idx": 4,
                  "initial_state_sha256": None}]
    sample = {"synthetic": False, "source_index_hash": "source-index", "generation_spec": {"use_cache": True},
              "mapping_version": "openvla_complete_forward_readout_v1", "sample_spec": {"mode": "pilot"},
              "readouts": [{"source_shard": "/old/dense/one.pt", "row": 3}], "discovery_episodes": discovery,
              "split_manifest_hash": None, "discovery_manifest_hash": verify.fingerprint(discovery)}
    sample["mapping_hash"] = verify.fingerprint({"index": sample["source_index_hash"], "generation": sample["generation_spec"],
                                                  "mapping_version": sample["mapping_version"]})
    sample["sample_hash"] = verify.fingerprint({"mapping_hash": sample["mapping_hash"], "sample": sample["sample_spec"],
                                                 "readouts": sample["readouts"]})
    _write(root, "sample_manifest.json", sample)
    readout = root / "readouts/shard_000000.pt"
    readout.parent.mkdir()
    readout.write_bytes(b"Fixture bytes, never deserialize")
    cache_manifest = _write(root, "readouts/manifest.json", {
        "synthetic": False, "sample_manifest_hash": sample["sample_hash"], "num_readouts": 1,
        "shards": [{"path": readout.name, "num_rows": 1, "sha256": verify.sha256_file(readout)}]})
    head_manifest = _write(root, "head/output_head_manifest.json", {"schema_version": "fixture"})
    head_weights = root / "head/output_head.safetensors"
    head_weights.write_bytes(b"Fixture weights, never deserialize")
    identity = {"sae_checkpoint_hash": "a" * 64, "layer_idx": 31, "processor_identity": "processor",
                "generation_manifest_hash": hashlib.sha256(b"{}\n").hexdigest(),
                "head_manifest_hash": verify.sha256_file(head_manifest),
                "head_weights_hash": verify.sha256_file(head_weights)}
    score_identity = {**identity, "sample_hash": sample["sample_hash"],
                      "cache_manifest_hash": verify.sha256_file(cache_manifest)}
    scope = {"sample_manifest_hash": sample["sample_hash"], "discovery_manifest_hash": sample["discovery_manifest_hash"],
             "split_manifest_hash": None}
    scores = {"synthetic": False, "identity": score_identity, "identity_hash": verify.fingerprint(score_identity), "scope": scope}
    scores_path = _write(root, "scores/scores.json", scores)
    _write(root, "scores/score_manifest.json", scores)
    evaluation = {"cases": cases, "discovery_eval_overlap": False}
    evaluation_path = _write(root, "plan/eval_manifest.json", evaluation)
    candidates = [{"feature_id": 2}, {"feature_id": 10}]
    conditions = [{"condition_id": "raw", "mode": "raw", "feature_id": None, "alpha": None, "num_cases": len(cases)},
                  {"condition_id": "identity-2", "mode": "identity", "feature_id": 2, "alpha": 1.0, "num_cases": len(cases)}]
    conditions += [{"condition_id": f"feature-{f}", "mode": "intervention", "feature_id": f, "alpha": 0.0,
                    "num_cases": len(cases)} for f in [2, 10]]
    plan = {"status": "planned", "synthetic": False, "conditions": conditions, "eval_cases": cases,
            "eval_manifest": evaluation, "eval_manifest_hash": verify.sha256_file(evaluation_path),
            "candidates": candidates, "feature_ids": [2, 10], "num_eval_cases": len(cases), "num_unique_features": 2,
            "total_rollouts": len(cases) * len(conditions), "score_artifact_hash": verify.sha256_file(scores_path),
            "scope": scope, "selection_eval_overlap": False, "top_k": 1,
            "methods": [EVENT, HEAD], "method_topk": {EVENT: [10], HEAD: [2]},
            "score_vectors": {EVENT: {"2": 0.1, "10": 0.9}, HEAD: {"2": 0.9, "10": 0.1}},
            "audit_feature_ids": [2, 10], "feature_universe": [2, 10]}
    if plan_config_hash is not None:
        plan["experiment_config_hash"] = plan_config_hash
    plan["plan_hash"] = verify.fingerprint(plan)
    _write(root, "plan/rollout_plan.json", plan)
    (root / "plan/candidates.jsonl").write_text("".join(json.dumps(row) + "\n" for row in candidates))
    policy = {"model": {"checkpoint": "/old/model"}, "env": {"seed": 2026}}
    protocol_id = verify.fingerprint({"plan_hash": plan["plan_hash"], "identity": identity, "code_revision": revision,
                                      "effective_policy": policy, "per_episode_seed": "sha256_seed_task_trial_v1"})
    for name in ("runtime_parity", "edit_parity"):
        _write(root, f"validation/{name}.json", {"status": "passed", "synthetic": False, "identity": identity,
                                               "identity_fingerprint": verify.fingerprint(identity),
                                               "sample_identity": hashlib.sha256(b"{}\n").hexdigest(),
                                               "checks": {"one": {"status": "passed"}}})
    all_results = {}
    for condition in conditions:
        name = condition["condition_id"]
        # Raw is 2/2 on task 0, 0/1 task 1. Feature 2 harms task 0 but helps task 1:
        # task-weighted drop = (0.5 + -1)/2 = -0.25, not the episode mean 0.
        outcomes = [False, True, True] if name == "feature-2" else [True, True, False]
        episodes = [{**case, "success": success, "caught_exception": None, "episode_seed": 100+i}
                    for i, (case, success) in enumerate(zip(cases, outcomes))]
        actions = [{"actions": [[0.1, 0.2]], "trial": i} for i in range(len(cases))]
        action_rel = f"runs/{name}/EVAL-native/actions.json"
        action_path = _write(root, action_rel, actions)
        payload = {"status": "completed", "synthetic": False, "mode": condition["mode"],
                   "feature_id": condition["feature_id"], "alpha": condition["alpha"], "completed_rollouts": len(cases),
                   "episodes": episodes, "suite": "libero_spatial", "seed": 2026, "protocol_id": protocol_id,
                   "head_sae_identity": identity, "code": {"commit": revision, "dirty": False}, "run_config": policy,
                   "model_checkpoint": "openvla/frozen", "model_revision": "model-rev", "model_code_revision": "code-rev",
                   "sae_sha256": identity["sae_checkpoint_hash"], "layer_idx": 31, "hook_start_step": 0,
                   "hook_metrics": {"num_forwards": 21, "post_intervention_active_feature_values": 0,
                                    "post_intervention_max_feature_activation": 0},
                   "actions_path": f"/old/pod/full10/{action_rel}", "actions_sha256": verify.sha256_file(action_path),
                   "elapsed_seconds": 15.5}
        if condition["mode"] == "identity":
            payload["identity_check"] = "passed"
        payload["result_hash"] = verify.fingerprint(payload)
        _write(root, f"runs/{name}/result.json", payload)
        all_results[name] = payload
    protocol = {"synthetic": False, "suite": "libero_spatial", "protocol_id": protocol_id,
                "code_revision": revision, "eval_cases": cases, "expected_feature_ids": [2, 10],
                "model_checkpoint": "openvla/frozen", "model_revision": "model-rev", "model_code_revision": "code-rev",
                "selection_eval_overlap": False, "evaluation_labels_previously_used": None}
    paired = assemble_paired_effects(all_results["raw"], {f: all_results[f"feature-{f}"] for f in [2, 10]}, protocol)
    analysis = analyze_prediction(plan, paired, {"bootstrap_replicates": 20, "plan": plan})
    _write(root, "analysis/paired_effects.json", paired)
    _write(root, "analysis/analysis.json", analysis)
    (root / "analysis/report.md").write_text("# Synthetic test fixture using real artifact schemas\n")
    return root


def test_relocated_results_verify_from_raw_outcomes(tmp_path):
    original = make_result_fixture(tmp_path / "original")
    moved = tmp_path / "downloaded"
    shutil.move(str(original), moved)
    result = verify.audit_results(moved)
    assert result["total_rollouts"] == 12
    assert result["num_conditions"] == 4
    assert result["condition_elapsed_seconds_sum"] == 62
    assert result["baseline_success_rate"] == 0.5  # equal task weighting
    assert result["comparison"]["methods"][HEAD]["mean_success_rate_drop"] == -0.25
    assert result["comparison"]["head_minus_event"]["difference"] == -0.25
    assert result["comparison"]["head_minus_event"]["low"] <= result["comparison"]["head_minus_event"]["high"]
    assert result["bootstrap_reexecuted"] is False


@pytest.mark.parametrize("mutation,match", [
    ("plan", "plan hash"), ("result", "Result hash"), ("missing", "Missing or unexpected"),
    ("actions", "Action artifact hash"), ("identity", "action sequences differ"),
    ("exception", "Exceptional"), ("cases", "Duplicate"), ("analysis", "feature mean drop"),
    ("scope", "Score/sample"), ("escape", "Unsafe actions_path"),
    ("wrong_condition", "condition directory"), ("paired", "Paired outcomes"),
    ("seed", "RNG seed mismatch"),
    ("readout", "Readout shard hash"), ("head_weights", "output head artifact hash"),
    ("head_manifest", "output head artifact hash"), ("cache_manifest", "cache manifest hash"),
])
def test_incomplete_or_inconsistent_experiment_rejected(tmp_path, mutation, match):
    root = make_result_fixture(tmp_path / "results")
    target = root / "runs/feature-2/result.json"
    payload = verify.read_json(target)
    if mutation == "plan":
        path = root / "plan/rollout_plan.json"
        data = verify.read_json(path)
        data["total_rollouts"] = 9
        _write(root, str(path.relative_to(root)), data)
    elif mutation == "missing":
        target.unlink()
    elif mutation == "actions":
        (root / "runs/feature-2/EVAL-native/actions.json").write_text("[]")
    elif mutation in {"readout", "head_weights", "head_manifest", "cache_manifest"}:
        relative = {"readout": "readouts/shard_000000.pt", "head_weights": "head/output_head.safetensors",
                    "head_manifest": "head/output_head_manifest.json", "cache_manifest": "readouts/manifest.json"}[mutation]
        (root / relative).write_bytes(b"changed")
    elif mutation == "identity":
        action = root / "runs/identity-2/EVAL-native/actions.json"
        _write(root, str(action.relative_to(root)), [{"different": True}])
        target = root / "runs/identity-2/result.json"
        payload = verify.read_json(target)
        payload["actions_sha256"] = verify.sha256_file(action)
        _write(root, str(target.relative_to(root)), payload)
        _resign(target)
    elif mutation in {"analysis", "paired"}:
        path = root / "analysis" / ("analysis.json" if mutation == "analysis" else "paired_effects.json")
        data = verify.read_json(path)
        if mutation == "analysis":
            data["feature_effects"][0]["drop"] = 0.5
        else:
            data["paired_rows"][0]["raw_success"] = False
        _write(root, str(path.relative_to(root)), data)
    elif mutation == "scope":
        # Re-sign scores and the plan to prove scope joins, not just byte hashes, are checked.
        scores = verify.read_json(root / "scores/scores.json")
        scores["scope"]["discovery_manifest_hash"] = "changed"
        score_path = _write(root, "scores/scores.json", scores)
        _write(root, "scores/score_manifest.json", scores)
        plan = verify.read_json(root / "plan/rollout_plan.json")
        plan["score_artifact_hash"] = verify.sha256_file(score_path)
        plan.pop("plan_hash")
        plan["plan_hash"] = verify.fingerprint(plan)
        _write(root, "plan/rollout_plan.json", plan)
    else:
        if mutation == "result":
            payload["elapsed_seconds"] = 9
        elif mutation == "exception":
            payload["episodes"][0]["caught_exception"] = "RuntimeError"
        elif mutation == "cases":
            payload["episodes"][1] = copy.deepcopy(payload["episodes"][0])
        elif mutation == "escape":
            payload["actions_path"] = "/old/runs/feature-2/../../private/actions.json"
        elif mutation == "wrong_condition":
            payload["actions_path"] = "/old/runs/raw/EVAL-native/actions.json"
        elif mutation == "seed":
            payload["episodes"][0]["episode_seed"] = 999
        _write(root, str(target.relative_to(root)), payload)
        if mutation != "result":
            _resign(target)
    with pytest.raises(ValueError, match=match):
        verify.audit_results(root)
