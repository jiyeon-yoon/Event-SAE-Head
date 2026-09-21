"""Regression for multi-digit feature keys changing a frozen plan after JSON I/O."""

import copy

import pytest

from event_sae.research.output_head.candidates import build_evaluation_plan
from event_sae.research.output_head.provenance import atomic_write_json, fingerprint, read_json


def _scores():
    return {"synthetic": False, "scope": {"task_ids": [0]},
            "alive_feature_ids": [2, 10, 100],
            "scores": {"head_full_vocab_kl_fixed_prefix": {2: 0.25, 10: 0.5, 100: 0.1}}}


def _selection():
    return {"methods": ["head_full_vocab_kl_fixed_prefix"], "top_k": 1,
            "random_audit_features": 2, "max_unique_features": 3}


def _evaluation():
    return {"suite": "libero_spatial", "mode": "pilot", "discovery_eval_overlap": False,
            "evaluation_labels_previously_used": False,
            "cases": [{"task_id": 0, "task_episode_idx": 5, "initial_state_sha256": "a" * 64}]}


@pytest.mark.parametrize("string_ids", [False, True])
def test_plan_hash_and_contents_survive_json_round_trip(tmp_path, string_ids):
    scores = _scores()
    if string_ids:
        scores["scores"] = {method: {str(key): value for key, value in vector.items()}
                            for method, vector in scores["scores"].items()}
    original = copy.deepcopy(scores)
    plan = build_evaluation_plan(scores, None, _evaluation(), {"max_total_rollouts": 10},
                                 selection=_selection())
    assert scores == original
    assert plan["method_topk"]["head_full_vocab_kl_fixed_prefix"] == [10]
    assert set(plan["score_vectors"]["head_full_vocab_kl_fixed_prefix"]) == {"2", "10", "100"}
    path = tmp_path / "plan.json"
    atomic_write_json(path, plan)
    restored = read_json(path)
    assert restored == plan
    assert restored["plan_hash"] == fingerprint({k: v for k, v in restored.items() if k != "plan_hash"})


def test_workflow_plan_round_trip_resume_and_tamper_guard(tmp_path, monkeypatch):
    from event_sae.research.output_head import runtime, workflow
    from event_sae.research.output_head.config import validate_config

    root = tmp_path / "output"
    evaluation_path = tmp_path / "eval.json"
    base_path = tmp_path / "base.json"
    cfg = validate_config({"output": {"root_dir": str(root)}, "selection": _selection(),
                           "rollout": {"eval_manifest": str(evaluation_path),
                                       "base_eval_config": str(base_path), "max_total_rollouts": 10}})
    atomic_write_json(root / "scores/scores.json", _scores())
    atomic_write_json(root / "sample_manifest.json", {"discovery_episodes": []})
    atomic_write_json(evaluation_path, _evaluation())
    atomic_write_json(base_path, {})
    monkeypatch.setattr(workflow, "implementation_fingerprint", lambda: "test-implementation")
    monkeypatch.setattr(runtime, "numerical_implementation_fingerprint", lambda: "test-implementation")
    monkeypatch.setattr(runtime, "git_identity", lambda root: {"commit": "test-commit", "dirty": False})
    plan = workflow.plan_workflow(cfg)
    path = root / "plan/rollout_plan.json"
    original_bytes = path.read_bytes()
    assert read_json(path) == plan
    assert runtime.verify_approved_plan(cfg, plan["plan_hash"], "test-commit") == plan
    assert workflow.plan_workflow(cfg) == plan
    assert path.read_bytes() == original_bytes
    tampered = read_json(path)
    tampered["score_vectors"]["head_full_vocab_kl_fixed_prefix"]["10"] = 9.0
    atomic_write_json(path, tampered, overwrite=True)
    with pytest.raises(ValueError, match="tampered"):
        runtime.verify_approved_plan(cfg, plan["plan_hash"], "test-commit")


def test_historical_fingerprints_are_not_reinterpreted():
    # Preserve both old in-memory numeric-key digests and already corrected,
    # serialized pilot digests. Do not silently repair a frozen old artifact.
    assert fingerprint({"score_vectors": {"head": {2: 0.25, 10: 0.5}}}) == (
        "fbba69893dcdd336ff1d6bebccd6b2b19633510655a353b5feb17cb2abcc20fd")
    assert fingerprint({"score_vectors": {"head": {"2": 0.25, "10": 0.5}}}) == (
        "843c10ccdeac6f552a8a9938d5775395e758f6f71065bfbad3a29e24f2bcdf7c")
