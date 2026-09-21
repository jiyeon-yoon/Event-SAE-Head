"""Discovery-only event ranking must never read evaluation activation values."""

import json
import runpy
import sys
from pathlib import Path

import pytest
import torch

from event_sae.scoring.score_matrix import score_cluster_features


def _jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _fixture(tmp_path, *, shard_tasks=True, prompts=True, held_out_scale=1000.0):
    episodes = [1] * 3 + [2] * 3 + [3] * 3
    shard = {
        "episode_num": torch.tensor(episodes),
        "step_in_episode": torch.tensor([0, 1, 2] * 3),
        "token_idx": torch.zeros(9, dtype=torch.int64),
        "chunk_start_step": torch.tensor([0, 1, 2] * 3),
        "executed_chunk_len": torch.ones(9, dtype=torch.int64),
        "top_feature_ids": torch.tensor([[0]] * 3 + [[1]] * 3 + [[0]] * 3),
        "top_feature_vals": torch.tensor(
            [[0.0], [1.0], [2.0], [0.0], [held_out_scale], [0.0], [6.0], [6.0], [6.0]]
        ),
    }
    if shard_tasks:
        shard["task_id"] = torch.zeros(9, dtype=torch.int64)
    torch.save(shard, tmp_path / "s0.pt")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "format": "token_topk_sparse_v1", "dict_size": 2, "topk": 1,
        "shards": [{"path": "s0.pt"}],
    }), encoding="utf-8")
    events = [{
        "sample_id": f"s{episode}", "task_description": "task", "task_id": 0,
        "task_episode_idx": episode - 1, "episode_num": episode, "waypoint_rank": 0,
        "waypoint_step": 1, "progress_percent": 0.5, "num_steps": 3,
    } for episode in (1, 2)]
    _jsonl(tmp_path / "events.jsonl", events)
    _jsonl(tmp_path / "assignments.jsonl", [{
        "sample_id": f"s{episode}", "task_description": "task", "cluster_id": "c0",
    } for episode in (1, 2)])
    _jsonl(tmp_path / "clusters.jsonl", [{
        "cluster_id": "c0", "task_description": "task", "episode_coverage": 0.99,
    }])
    _jsonl(tmp_path / "prompts.jsonl", [{
        "episode_num": episode, "task_id": 0, "task_episode_idx": episode - 1,
        "task_description": "task",
    } for episode in (1, 2, 3)])
    return {
        "topk_run_dir": tmp_path, "event_features_path": tmp_path / "events.jsonl",
        "cluster_assignments_path": tmp_path / "assignments.jsonl",
        "clusters_path": tmp_path / "clusters.jsonl", "cluster_annotations_path": None,
        "prompt_records_path": tmp_path / "prompts.jsonl" if prompts else None,
        "output_path": tmp_path / "scores.pt", "window_size": 1,
    }


def _score(options, **kwargs):
    score_cluster_features(**options, **kwargs)
    return torch.load(options["output_path"], map_location="cpu", weights_only=False)


@pytest.mark.parametrize("step_mapping", ["inference_step", "action_executed", "chunk_executed"])
@pytest.mark.parametrize("shard_tasks,prompts", [(True, True), (False, True), (True, False)])
def test_discovery_filter_covers_events_windows_task_mean_and_coverage(
    tmp_path, step_mapping, shard_tasks, prompts,
):
    options = _fixture(tmp_path, shard_tasks=shard_tasks, prompts=prompts)
    result = _score(options, allowed_episode_nums={1, 3}, step_mapping=step_mapping)
    assert [event["episode_num"] for event in result["selected_events"]] == [1]
    assert result["matrix_raw"][0, 0] > 0
    assert result["matrix_raw"][0, 1] == 0
    torch.testing.assert_close(result["matrix_window_mean"], torch.tensor([[1.0, 0.0]]))
    # Episode 3 has no event, but its three timesteps still belong in task_mean.
    torch.testing.assert_close(result["matrix_task_mean"], torch.tensor([[3.5, 0.0]]))
    assert result["row_keys"][0]["episode_coverage"] == 0.5
    assert result["row_results"][0]["episode_coverage"] == 0.5
    assert result["row_keys"][0]["total_task_episodes"] == 2
    assert result["source"]["allowed_episode_nums"] == [1, 3]
    assert result["source"]["episode_filter"] == "allowed_episode_nums_v1"
    counts = result["selection_counts"]
    assert counts["rows_used"] == 6
    assert counts["rows_skipped_episode_filter"] == 3
    assert counts["event_features_skipped_episode_filter"] == 1
    assert counts["discovery_task_episode_counts"] == {0: 2}
    assert counts["task_timestep_counts"] == {0: 6}


def test_held_out_activations_do_not_change_any_matrix(tmp_path):
    options = _fixture(tmp_path, held_out_scale=1000.0)
    before = _score(options, allowed_episode_nums={1, 3})
    options = _fixture(tmp_path, held_out_scale=-99999.0)
    after = _score(options, allowed_episode_nums={1, 3})
    for key in ("matrix_raw", "matrix_window_mean", "matrix_task_mean"):
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)


def test_omitted_filter_preserves_whole_run_formula_and_metadata(tmp_path):
    options = _fixture(tmp_path)
    result = _score(options)
    assert len(result["selected_events"]) == 2
    torch.testing.assert_close(
        result["matrix_task_mean"], torch.tensor([[21 / 9, 1000 / 9]])
    )
    assert result["row_keys"][0]["episode_coverage"] == 0.99
    assert result["source"]["allowed_episode_nums"] is None


@pytest.mark.parametrize("source", ["events", "prompts", "shard"])
def test_conflicting_discovery_episode_task_mapping_fails(tmp_path, source):
    options = _fixture(tmp_path)
    if source == "shard":
        path = tmp_path / "s0.pt"
        data = torch.load(path, weights_only=False)
        data["task_id"][0] = 8
        torch.save(data, path)
    else:
        path = tmp_path / f"{source}.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records.append({**records[0], "task_id": 8, "sample_id": "another-event"})
        _jsonl(path, records)
    with pytest.raises(ValueError, match="Conflicting task_id"):
        _score(options, allowed_episode_nums={1, 3})


def test_missing_task_mapping_for_event_free_discovery_episode_fails(tmp_path):
    options = _fixture(tmp_path, shard_tasks=False, prompts=False)
    with pytest.raises(ValueError, match="Missing task mapping.*3"):
        _score(options, allowed_episode_nums={1, 3})


@pytest.mark.parametrize("step_mapping", ["inference_step", "action_executed", "chunk_executed"])
@pytest.mark.parametrize("missing_kind", ["absent", "no_valid_steps"])
def test_discovery_episode_without_contributing_timesteps_fails(tmp_path, step_mapping, missing_kind):
    options = _fixture(tmp_path)
    path = tmp_path / "s0.pt"
    data = torch.load(path, weights_only=False)
    if missing_kind == "absent":
        data = {key: value[:6] for key, value in data.items()}
    else:
        data["step_in_episode"][6:] = -1
        data["executed_chunk_len"][6:] = 0
    torch.save(data, path)
    with pytest.raises(ValueError, match="No valid activation timesteps.*3"):
        _score(options, allowed_episode_nums={1, 3}, step_mapping=step_mapping)


def test_task_with_no_discovery_events_is_explicitly_rejected(tmp_path):
    options = _fixture(tmp_path)
    path = tmp_path / "s0.pt"
    data = torch.load(path, weights_only=False)
    data["task_id"][6:] = 5
    torch.save(data, path)
    _jsonl(tmp_path / "prompts.jsonl", [{"episode_num": 3, "task_id": 5}])
    with pytest.raises(ValueError, match="No usable clustered events.*5"):
        _score(options, allowed_episode_nums={1, 3})


def test_excluded_prompt_and_shard_tasks_cannot_override_discovery_scope(tmp_path):
    options = _fixture(tmp_path)
    path = tmp_path / "s0.pt"
    data = torch.load(path, weights_only=False)
    data["task_id"][3:6] = 999
    torch.save(data, path)
    result = _score(options, allowed_episode_nums={1, 3})
    torch.testing.assert_close(result["matrix_task_mean"], torch.tensor([[3.5, 0.0]]))


def test_duplicate_task_trial_mapped_to_different_episode_fails(tmp_path):
    options = _fixture(tmp_path)
    _jsonl(tmp_path / "prompts.jsonl", [
        {"episode_num": 3, "task_id": 0, "task_episode_idx": 0}
    ])
    with pytest.raises(ValueError, match="Duplicate discovery task/episode identity"):
        _score(options, allowed_episode_nums={1, 3})


def test_sample_id_collision_cannot_silently_remove_discovery_event(tmp_path):
    options = _fixture(tmp_path)
    path = tmp_path / "events.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[1]["sample_id"] = records[0]["sample_id"]
    _jsonl(path, records)
    with pytest.raises(ValueError, match="Duplicate sample_id spans"):
        _score(options, allowed_episode_nums={1, 3})


@pytest.mark.parametrize("allowed", [set(), [-1], [1, 1], [True], [1.1]])
def test_invalid_discovery_episode_ids_fail(tmp_path, allowed):
    options = _fixture(tmp_path)
    with pytest.raises(ValueError, match="episode"):
        _score(options, allowed_episode_nums=allowed)


def test_cli_passes_explicit_episode_filter(tmp_path, monkeypatch):
    import event_sae.scoring

    selected = tmp_path / "allowed.json"
    selected.write_text("[1, 3]", encoding="utf-8")
    captured = {}
    monkeypatch.setattr(event_sae.scoring, "score_cluster_features", lambda **kw: captured.update(kw) or {})
    monkeypatch.setattr(sys, "argv", [
        "score_cluster_features.py", "--topk-run-dir", str(tmp_path),
        "--event-features-path", "events.jsonl", "--cluster-assignments-path", "assignments.jsonl",
        "--clusters-path", "clusters.jsonl", "--output-path", "scores.pt",
        "--allowed-episodes-path", str(selected),
    ])
    script = Path(__file__).resolve().parents[1] / "scripts" / "score_cluster_features.py"
    runpy.run_path(str(script), run_name="__main__")
    assert captured["allowed_episode_nums"] == {1, 3}
