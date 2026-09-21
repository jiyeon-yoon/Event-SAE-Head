"""Post-pilot followup gates protect new evaluation without inventing history."""

import copy
import hashlib
import json

import pytest

from event_sae.research.output_head.capture import _capture_case_specs
from event_sae.research.output_head.config import validate_config
from event_sae.research.output_head.readouts import build_readout_manifest
from event_sae.research.output_head.runtime import _validate_frozen_split
from event_sae.research.output_head.splits import (
    _digest, build_followup_split, validate_selection_split, validate_split_manifest,
)
from event_sae.research.output_head.workflow import split_workflow


def registry():
    rows = []
    for trial in range(6):
        digest = hashlib.sha256(f"state:{trial}".encode()).hexdigest()
        rows.append({"suite": "libero_spatial", "task_id": 0, "task_episode_idx": trial,
                     "episode_num": trial, "source_run_id": "run-a",
                     "initial_state_sha256": digest, "current_runtime_state_sha256": digest,
                     "historical_initial_state_sha256": None,
                     "initial_state_hash_provenance": "current_libero_state_registry_v1",
                     "evaluation_labels_previously_used": True})
    return {"episodes": rows, "historical_state_identity_verified": False}


def settings():
    return {"discovery_per_task": 3, "validation_per_task": 1, "evaluation_per_task": 2,
            "split_seed": 2026, "frozen_before_pilot": False}


def resign(split):
    split["manifest_hash"] = _digest({key: value for key, value in split.items() if key != "manifest_hash"})


def test_followup_preserves_unknown_historical_states_and_used_labels():
    split = build_followup_split(registry(), settings())
    assert validate_split_manifest(split, require_followup=True)["num_episodes"] == 6
    assert split["evaluation_labels_previously_used"] is True
    assert split["historical_state_identity_verified"] is False
    assert split["historical_source_state_status"] == "unavailable"
    assert split["frozen_before_pilot"] is False
    assert split["frozen_at_utc"]
    for rows in split["splits"].values():
        assert all(row["historical_initial_state_sha256"] is None for row in rows)
    with pytest.raises(ValueError, match="confirmatory"):
        validate_split_manifest(split, require_confirmatory=True)
    with pytest.raises(ValueError, match="backdated"):
        build_followup_split(registry(), {**settings(), "frozen_before_pilot": True})


@pytest.mark.parametrize("fault", ["current_hash", "invented_history", "backdate", "timestamp", "state_overlap", "misleading_scope"])
def test_resigned_followup_cannot_bypass_state_or_history_gates(fault):
    split = build_followup_split(registry(), settings())
    if fault == "current_hash":
        split["splits"]["discovery"][0]["current_runtime_state_sha256"] = "f" * 64
    elif fault == "invented_history":
        split["historical_state_identity_verified"] = True
    elif fault == "backdate":
        split["frozen_before_pilot"] = True
    elif fault == "timestamp":
        split["frozen_at_utc"] = "2999-01-01T00:00:00+00:00"
    elif fault == "misleading_scope":
        split["historical_source_state_status"] = "verified"
    else:
        split["splits"]["evaluation"][0] = copy.deepcopy(split["splits"]["discovery"][0])
    resign(split)
    with pytest.raises(ValueError):
        validate_split_manifest(split, require_followup=True)


def test_followup_enforces_stage_membership_without_weakening_pilot(tmp_path):
    split = build_followup_split(registry(), settings())
    for stage, allowed in (("score", "discovery"), ("pilot", "validation"), ("evaluation", "evaluation")):
        validate_selection_split(split, split["splits"][allowed], stage=stage, mode="followup")
    for stage in ("score", "pilot", "validation"):
        with pytest.raises(ValueError, match="forbidden split"):
            validate_selection_split(split, split["splits"]["evaluation"], stage=stage, mode="followup")
    with pytest.raises(ValueError, match="confirmatory"):
        validate_selection_split(split, split["splits"]["discovery"], stage="score", mode="pilot")
    path = tmp_path / "split.json"
    path.write_text(json.dumps(split))
    cfg = validate_config({"sampling": {"mode": "followup", "task_ids": [0], "split_manifest": str(path)}})
    specs = _capture_case_specs(cfg)
    assert specs[0]["task_episode_idx"] in {row["task_episode_idx"] for name in ("discovery", "validation") for row in split["splits"][name]}
    _validate_frozen_split(cfg, split["splits"]["evaluation"], stage="evaluation")
    with pytest.raises(ValueError, match="forbidden split"):
        _validate_frozen_split(cfg, split["splits"]["discovery"], stage="evaluation")


def test_followup_readouts_only_score_discovery_and_keep_provenance():
    source = registry()
    split = build_followup_split(source, settings())
    rows = [{"layer_idx": 31, "shard_path": "dense.pt", "row_start": i, "row_end": i + 1,
             "episode_num": i, "step_in_episode": 0, "global_forward_idx": i,
             "task_id": 0, "task_episode_idx": i} for i in range(6)]
    generation = {"layer_idx": 31, "action_dim": 1, "batch_size": 1, "padding": "none",
                  "use_cache": True, "source_run_id": "run-a", "suite": "libero_spatial", "evidence": "fixture"}
    sample = build_readout_manifest(rows, {"mode": "followup", "split_manifest": split},
                                    generation, source_episodes=source)
    assert len(sample["readouts"]) == 3
    assert {row["task_episode_idx"] for row in sample["readouts"]} == {row["task_episode_idx"] for row in split["splits"]["discovery"]}
    assert sample["state_identity_scope"]["historical_state_identity_verified"] is False
    assert all(row["historical_initial_state_sha256"] is None for row in sample["discovery_episodes"])
    source["episodes"][split["splits"]["discovery"][0]["episode_num"]]["current_runtime_state_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="provenance"):
        build_readout_manifest(rows, {"mode": "followup", "split_manifest": split}, generation, source_episodes=source)


def test_followup_freeze_resume_keeps_original_timestamp_and_rejects_reselection(tmp_path):
    source = tmp_path / "registry.json"
    source.write_text(json.dumps(registry()))
    cfg = validate_config({"inputs": {"source_episode_manifest": str(source)},
                           "sampling": {"mode": "followup"}, "splits": settings(),
                           "output": {"root_dir": str(tmp_path / "results")}})
    first = split_workflow(cfg)
    assert split_workflow(cfg) == first
    cfg["splits"]["split_seed"] += 1
    with pytest.raises(ValueError, match="differs"):
        split_workflow(cfg)


def test_followup_plan_binds_all_scores_samples_and_eval_to_frozen_split(tmp_path, monkeypatch):
    from event_sae.research.output_head import runtime, workflow
    from event_sae.research.output_head.provenance import atomic_write_json
    from event_sae.research.output_head.readouts import discovery_population_hash

    split = build_followup_split(registry(), settings())
    root = tmp_path / "results"
    split_path, eval_path, base_path = (tmp_path / name for name in ("split.json", "eval.json", "base.json"))
    evaluation = {key: split[key] for key in ("mode", "frozen_before_pilot", "frozen_at_utc",
                  "frozen_before_followup_evaluation", "historical_state_identity_verified",
                  "evaluation_labels_previously_used")}
    evaluation.update(split_manifest_hash=split["manifest_hash"], discovery_eval_overlap=False,
                      cases=split["splits"]["evaluation"])
    sample = {"discovery_episodes": split["splits"]["discovery"],
              "discovery_manifest_hash": discovery_population_hash(split["splits"]["discovery"]),
              "split_manifest_hash": split["manifest_hash"]}
    scores = {"synthetic": False, "scope": {"task_ids": [0],
               "split_manifest_hash": split["manifest_hash"],
               "discovery_manifest_hash": sample["discovery_manifest_hash"]},
              "alive_feature_ids": [2, 10], "scores": {"head_full_vocab_kl_fixed_prefix": {2: 0.1, 10: 0.3}}}
    cfg = validate_config({"sampling": {"mode": "followup", "task_ids": [0], "split_manifest": str(split_path)},
            "selection": {"methods": ["head_full_vocab_kl_fixed_prefix"], "top_k": 1, "random_audit_features": 1},
            "rollout": {"eval_manifest": str(eval_path), "base_eval_config": str(base_path)},
            "output": {"root_dir": str(root)}})
    for path, payload in ((split_path, split), (eval_path, evaluation), (base_path, {}),
                          (root / "sample_manifest.json", sample), (root / "scores/scores.json", scores)):
        atomic_write_json(path, payload)
    monkeypatch.setattr(workflow, "implementation_fingerprint", lambda: "fixture-code")
    monkeypatch.setattr(runtime, "numerical_implementation_fingerprint", lambda: "fixture-code")
    monkeypatch.setattr(runtime, "git_identity", lambda root: {"commit": "fixture-commit", "dirty": False})
    plan = workflow.plan_workflow(cfg)
    assert plan["mode"] == "followup" and plan["selection_eval_overlap"] is False
    assert plan["evaluation_labels_previously_used"] is True
    assert runtime.verify_approved_plan(cfg, plan["plan_hash"], "fixture-commit") == plan
    scores["scope"]["split_manifest_hash"] = "another-split"
    atomic_write_json(root / "scores/scores.json", scores, overwrite=True)
    with pytest.raises(ValueError, match="Score/sample"):
        workflow.plan_workflow(cfg)
    scores["scope"]["split_manifest_hash"] = split["manifest_hash"]
    atomic_write_json(root / "scores/scores.json", scores, overwrite=True)
    split["split_seed"] += 1
    resign(split)
    atomic_write_json(split_path, split, overwrite=True)
    with pytest.raises(ValueError, match="split changed"):
        runtime.verify_approved_plan(cfg, plan["plan_hash"], "fixture-commit")
