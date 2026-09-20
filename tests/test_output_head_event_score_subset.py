"""Exact-population derivation tests for legacy Event-SAE score matrices."""

from __future__ import annotations

import json

import pytest

from event_sae.research.output_head.candidates import load_ranking_scores
from event_sae.research.output_head.event_score_subset import build_exact_task_subset
from event_sae.research.output_head.provenance import sha256_file
from event_sae.research.output_head.readouts import discovery_population_hash


def _inputs(tmp_path, *, trials=(0, 1), state_hash=None):
    torch = pytest.importorskip("torch")
    source = tmp_path / "source.pt"
    rows = [
        {"task_id": 0, "episode_coverage": 1.0, "num_events": 2},
        {"task_id": 2, "episode_coverage": 1.0, "num_events": 3},
        {"task_id": 1, "episode_coverage": 1.0, "num_events": 4},
    ]
    matrices = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    selected = [
        {"task_id": task, "task_episode_idx": trial, "sample_id": f"{task}-{trial}"}
        for task in (0, 1, 2) for trial in (0, 1)
    ]
    torch.save({
        "row_keys": rows,
        "row_results": [{"row": index} for index in range(3)],
        "matrix_raw": matrices,
        "matrix_window_mean": matrices + 1,
        "matrix_task_mean": matrices + 2,
        "matrix": matrices,
        "selected_events": selected,
        "selection_counts": {"task_timestep_counts": {"0": 10, "1": 20, "2": 30},
                             "selected_events_after_activation_filter": 6},
    }, source)
    discovery = [
        {"suite": "libero_spatial", "task_id": task, "task_episode_idx": trial,
         "initial_state_sha256": state_hash, "episode_uid": f"uid-{task}-{trial}"}
        for task in (0, 1) for trial in trials
    ]
    discovery_hash = discovery_population_hash(discovery)
    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps({"sample_hash": "sample-a", "discovery_episodes": discovery,
                                  "discovery_manifest_hash": discovery_hash}))
    scores = tmp_path / "scores.json"
    scores.write_text(json.dumps({"scope": {
        "task_ids": [0, 1], "dictionary_size": 4,
        "discovery_manifest_hash": discovery_hash, "sample_manifest_hash": "sample-a",
        "split_manifest_hash": None, "sae_sha256": "sae-a",
    }}))
    return source, sample, scores


def test_exact_subset_filters_rows_and_carries_verified_scope(tmp_path):
    torch = pytest.importorskip("torch")
    source, sample, scores = _inputs(tmp_path)
    output = tmp_path / "subset.pt"
    report = build_exact_task_subset(
        source, sample, scores, output, task_ids=[0, 1], suite="libero_spatial",
        expected_source_sha256=sha256_file(source), expected_sae_sha256="sae-a",
        expected_episodes_per_task=2,
    )
    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert report["status"] == "created"
    assert [row["task_id"] for row in payload["row_keys"]] == [0, 1]
    assert payload["matrix_raw"].tolist() == [[0., 1., 2., 3.], [8., 9., 10., 11.]]
    assert payload["selection_counts"] == {"task_timestep_counts": {"0": 10, "1": 20}}
    assert payload["scope"]["population_binding"]["relation"] == "exact_equal"
    assert payload["scope"]["population_binding"]["num_episodes"] == 4
    assert payload["scope"]["population_binding"]["episodes_per_task"] == {0: 2, 1: 2}
    assert payload["scope"]["discovery_manifest_hash"] == payload["scope"]["population_binding"]["population_hash"]
    assert "sample_manifest_hash" not in payload["scope"]
    assert payload["scope"]["timestep_scope"] == "legacy_event_all_retained_timesteps"
    assert payload["scope"]["sae_sha256"] == "sae-a"
    assert payload["derivation"]["operation"] == "row_filter_only_no_episode_reaggregation"
    assert payload["derivation"]["source_artifact_sha256"] == sha256_file(source)
    assert payload["derivation"]["head_sample_manifest_hash"] == "sample-a"
    loaded = load_ranking_scores(payload, ["event_aligned"], expected_scope=payload["scope"])
    assert loaded["scope"]["task_ids"] == [0, 1]


def test_partial_population_cannot_be_stamped_with_discovery_hash(tmp_path):
    source, sample, scores = _inputs(tmp_path, trials=(0,))
    output = tmp_path / "subset.pt"
    with pytest.raises(ValueError, match="not exactly equal"):
        build_exact_task_subset(
            source, sample, scores, output, task_ids=[0, 1], suite="libero_spatial",
            expected_source_sha256=sha256_file(source), expected_sae_sha256="sae-a",
            expected_episodes_per_task=1,
        )
    assert not output.exists()


def test_unknown_source_state_hash_cannot_bind_hashed_sample(tmp_path):
    source, sample, scores = _inputs(tmp_path, state_hash="a" * 64)
    with pytest.raises(ValueError, match="lack initial-state hashes"):
        build_exact_task_subset(
            source, sample, scores, tmp_path / "subset.pt", task_ids=[0, 1],
            suite="libero_spatial", expected_source_sha256=sha256_file(source),
            expected_sae_sha256="sae-a",
            expected_episodes_per_task=2,
        )


def test_source_and_sae_hashes_are_mandatory_evidence(tmp_path):
    source, sample, scores = _inputs(tmp_path)
    with pytest.raises(ValueError, match="Source SHA-256 mismatch"):
        build_exact_task_subset(
            source, sample, scores, tmp_path / "bad-source.pt", task_ids=[0, 1],
            suite="libero_spatial", expected_source_sha256="0" * 64,
            expected_sae_sha256="sae-a",
            expected_episodes_per_task=2,
        )
    with pytest.raises(ValueError, match="SAE SHA-256"):
        build_exact_task_subset(
            source, sample, scores, tmp_path / "bad-sae.pt", task_ids=[0, 1],
            suite="libero_spatial", expected_source_sha256=sha256_file(source),
            expected_sae_sha256="wrong",
            expected_episodes_per_task=2,
        )
