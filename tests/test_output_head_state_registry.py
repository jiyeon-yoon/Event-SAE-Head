import ast
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from event_sae.research.output_head.readouts import build_readout_manifest
from event_sae.research.output_head.splits import build_followup_split
from event_sae.research.output_head.state_registry import (
    build_generation_manifest,
    build_state_registry,
    initial_state_sha256,
)


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def corpus(tmp_path, tasks=(0, 2), trials=(0, 1, 2)):
    prompts, index, states = [], [], {}
    episode, forward = 0, 0
    dense = tmp_path / "sae_activations/post_mlp_residual"
    dense.mkdir(parents=True)
    for run, task in enumerate(tasks):
        offset = 0
        shard = f"run{run}.pt"
        (dense / shard).write_bytes(b"metadata-only fixture: not a torch tensor")
        for trial in trials:
            episode += 1
            prompt = {"source_run_idx": run, "episode_num": episode,
                      "source_episode_num": trial + 1, "task_id": task,
                      "task_episode_idx": trial}
            prompts.append(prompt)
            states[("libero_spatial", task, trial)] = np.asarray([task, trial, 3.25], dtype=np.float64)
            for step in (0, 1):
                for dim in range(7):
                    forward += 1
                    count = 9 if dim == 0 else 1
                    index.append({**prompt, "layer_idx": 31, "shard_path": shard,
                                  "step_in_episode": step, "global_forward_idx": forward,
                                  "row_start": offset, "row_end": offset + count})
                    offset += count
    _write(tmp_path / "prompt_records.jsonl", prompts)
    _write(dense / "activation_index.jsonl", index)
    return prompts, index, states


def build(root, states, **kwargs):
    return build_state_registry(root, runtime_states=states,
                                expected_tasks=(0, 2), expected_trials=3, **kwargs)


def test_all_rows_join_and_current_hash_does_not_fabricate_history(tmp_path):
    prompts, index, states = corpus(tmp_path)
    result = build(tmp_path, states, expected_shards=2)
    assert len(result["episodes"]) == 6
    assert result["historical_state_identity_verified"] is False
    assert result["structural_audit"]["num_forwards"] == 84
    assert result["structural_audit"]["num_action_queries"] == 12
    for row in result["episodes"]:
        assert row["current_runtime_state_sha256"] == row["initial_state_sha256"]
        assert row["historical_initial_state_sha256"] is None
        assert row["historical_state_identity_verified"] is False
        assert row["evaluation_labels_previously_used"] is (True if row["task_id"] == 0 else None)
    generation = build_generation_manifest(result)
    assert generation["evidence"]["historical_runtime_flags_verified"] is False
    split = build_followup_split(result, {"discovery_per_task": 1, "validation_per_task": 1,
                                          "evaluation_per_task": 1, "split_seed": 2026})
    mapped = build_readout_manifest(index, {"mode": "followup", "split_manifest": split},
                                    generation, source_episodes=result)
    assert len(mapped["readouts"]) == 28
    assert all(row["initial_state_sha256"] for row in mapped["readouts"])
    with pytest.raises(ValueError, match="followup"):
        build_readout_manifest(index, {"mode": "pilot"}, generation, source_episodes=result)
    index_path = tmp_path / "sae_activations/post_mlp_residual/activation_index.jsonl"
    assert result["source_files"]["activation_index"]["sha256"] == hashlib.sha256(index_path.read_bytes()).hexdigest()


def test_state_hash_matches_native_runner_without_importing_simulator():
    runner = Path(__file__).resolve().parents[1] / "event_sae/openvla/eval/runner.py"
    parsed = ast.parse(runner.read_text())
    function = next(node for node in parsed.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_initial_state_sha256")
    namespace = {"np": np, "hashlib": hashlib, "json": json}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(runner), "exec"), namespace)
    native = namespace["_initial_state_sha256"]
    for values in [np.asarray([1, 2]), np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]]:
        assert initial_state_sha256(values) == native(values)
    assert initial_state_sha256(np.array([1], dtype=np.float32)) != initial_state_sha256(np.array([1], dtype=np.float64))


@pytest.mark.parametrize("change,match", [
    ("prompt_duplicate", "Duplicate/ambiguous"),
    ("source_duplicate", "Duplicate/ambiguous"),
    ("task_trial_duplicate", "Duplicate/ambiguous"),
    ("index_task", "Activation/prompt mismatch"),
    ("index_original_episode", "Activation/prompt mismatch"),
    ("index_unknown_episode", "unknown prompt"),
    ("missing_episode", "coverage differs"),
    ("missing_forward", "Expected 7"),
    ("duplicate_forward", "Duplicate activation forward"),
    ("wrong_layer", "suite/layer"),
    ("cached_two_rows", "Cached decode row"),
    ("row_overlap", "gap/overlap"),
    ("wrong_dim", "dimension order"),
    ("batch_two", "contradicts generation"),
])
def test_corrupt_or_ambiguous_source_records_fail(tmp_path, change, match):
    prompts, index, states = corpus(tmp_path)
    if change == "prompt_duplicate":
        prompts.append(prompts[0])
    elif change == "source_duplicate":
        prompts[1]["source_episode_num"] = prompts[0]["source_episode_num"]
    elif change == "task_trial_duplicate":
        prompts[1]["task_episode_idx"] = prompts[0]["task_episode_idx"]
    elif change == "index_task":
        index[0]["task_id"] = 10
    elif change == "index_original_episode":
        index[0]["source_episode_num"] = 99
    elif change == "index_unknown_episode":
        index[0]["episode_num"] = 99
    elif change == "missing_episode":
        index = [row for row in index if row["episode_num"] != 6]
    elif change == "missing_forward":
        # Truncate the last query so unrelated range checks still pass.
        index.pop()
    elif change == "duplicate_forward":
        index[1]["global_forward_idx"] = index[0]["global_forward_idx"]
    elif change == "wrong_layer":
        index[0]["layer_idx"] = 30
    elif change == "cached_two_rows":
        index[-1]["row_end"] += 1
    elif change == "row_overlap":
        index[1]["row_start"] -= 1
    elif change == "wrong_dim":
        index[0]["action_dim_index"] = 4
    elif change == "batch_two":
        index[0]["batch_size"] = 2
    _write(tmp_path / "prompt_records.jsonl", prompts)
    _write(tmp_path / "sae_activations/post_mlp_residual/activation_index.jsonl", index)
    with pytest.raises(ValueError, match=match):
        build(tmp_path, states)


def test_original_historical_hash_checked_against_runtime(tmp_path):
    prompts, index, states = corpus(tmp_path)
    for row in prompts:
        row["initial_state_sha256"] = initial_state_sha256(states[("libero_spatial", row["task_id"], row["task_episode_idx"])])
    _write(tmp_path / "prompt_records.jsonl", prompts)
    assert build(tmp_path, states)["historical_state_identity_verified"] is True
    states[("libero_spatial", 2, 2)] += 0.5
    with pytest.raises(ValueError, match="Historical/current"):
        build(tmp_path, states)


def test_conflicting_historical_hash_in_dense_index_is_rejected(tmp_path):
    prompts, index, states = corpus(tmp_path)
    prompts[0]["initial_state_sha256"] = "a" * 64
    index[0]["initial_state_sha256"] = "b" * 64
    _write(tmp_path / "prompt_records.jsonl", prompts)
    _write(tmp_path / "sae_activations/post_mlp_residual/activation_index.jsonl", index)
    with pytest.raises(ValueError, match="Conflicting historical"):
        build(tmp_path, states)


def test_duplicate_runtime_state_and_missing_state_are_rejected(tmp_path):
    _, _, states = corpus(tmp_path)
    original = states[("libero_spatial", 0, 1)]
    states[("libero_spatial", 0, 1)] = states[("libero_spatial", 0, 0)]
    with pytest.raises(ValueError, match="Duplicate current initial"):
        build(tmp_path, states)
    states[("libero_spatial", 0, 1)] = original
    del states[("libero_spatial", 2, 2)]
    with pytest.raises(ValueError, match="Missing current runtime"):
        build(tmp_path, states)


def test_registry_supports_lazy_native_loader_and_callable_states(tmp_path):
    _, _, states = corpus(tmp_path)
    calls = []
    def loader(suite, tasks):
        calls.append((suite, tasks))
        return states, {"source": "test-loader", "git_revision": "test"}
    result = build_state_registry(tmp_path, native_state_loader=loader,
                                 expected_tasks=(0, 2), expected_trials=3)
    assert calls == [("libero_spatial", [0, 2])]
    assert result["runtime_evidence"]["source"] == "test-loader"
    assert build(tmp_path, lambda *key: states[key])["episodes"] == result["episodes"]


def test_invalid_source_never_loads_native_runtime(tmp_path):
    prompts, _, _ = corpus(tmp_path)
    prompts.append(prompts[0])
    _write(tmp_path / "prompt_records.jsonl", prompts)
    def not_called(*_):
        raise AssertionError("Runtime must not load on invalid metadata")
    with pytest.raises(ValueError, match="Duplicate/ambiguous"):
        build_state_registry(tmp_path, native_state_loader=not_called)


def test_count_and_manifest_tampering_are_rejected(tmp_path):
    _, _, states = corpus(tmp_path)
    with pytest.raises(ValueError, match="task coverage"):
        build_state_registry(tmp_path, runtime_states=states, expected_tasks=10, expected_trials=3)
    with pytest.raises(ValueError, match="trial coverage"):
        build_state_registry(tmp_path, runtime_states=states, expected_tasks=(0, 2), expected_trials=50)
    with pytest.raises(ValueError, match="shard count"):
        build(tmp_path, states, expected_shards=369)
    result = copy.deepcopy(build(tmp_path, states))
    result["historical_state_identity_verified"] = True
    with pytest.raises(ValueError, match="manifest hash"):
        build_generation_manifest(result)
