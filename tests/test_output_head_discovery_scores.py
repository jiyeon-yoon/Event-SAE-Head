"""Exact discovery population/provenance binding, with a real tiny scorer."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from event_sae.research.output_head.config import load_config
from event_sae.research.output_head.discovery_scores import build_discovery_event_scores
from event_sae.research.output_head.provenance import fingerprint, sha256_file
from event_sae.research.output_head.readouts import build_readout_manifest
from event_sae.research.output_head.splits import build_followup_split
from event_sae.research.output_head.state_registry import build_generation_manifest, build_state_registry


def _json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _fixture(tmp_path):
    torch = pytest.importorskip("torch")
    dense = tmp_path / "merged/sae_activations/post_mlp_residual"
    dense.mkdir(parents=True)
    (dense / "dense.pt").write_bytes(b"dense is unused by the Event scorer")
    weights = tmp_path / "inputs/ae.pt"
    weights.parent.mkdir()
    weights.write_bytes(b"frozen-sae-fixture")
    episodes = [{"suite": "libero_spatial", "task_id": task, "task_episode_idx": trial,
                 "episode_num": task * 3 + trial + 1, "source_run_idx": 0,
                 "source_episode_num": task * 3 + trial + 1,
                 "task_description": f"task {task}"}
                for task in (0, 1) for trial in range(3)]
    registry = tmp_path / "inputs/source_episodes.json"
    _jsonl(tmp_path / "merged/prompt_records.jsonl", episodes)
    split_path = tmp_path / "inputs/split.json"
    rows = []
    for ep in episodes:
        for step in range(11):
            index = len(rows)
            rows.append({key: ep[key] for key in ("episode_num", "source_run_idx", "source_episode_num", "task_id", "task_episode_idx")})
            rows[-1].update(layer_idx=31, global_forward_idx=index, row_start=index,
                            row_end=index + 1, step_in_episode=step, shard_path="dense.pt")
    index_path = dense / "activation_index.jsonl"
    _jsonl(index_path, rows)
    source = build_state_registry(
        tmp_path / "merged", expected_tasks=2, expected_trials=3, expected_shards=1,
        action_dim=1, runtime_states={
            (ep["suite"], ep["task_id"], ep["task_episode_idx"]): [ep["task_id"], ep["task_episode_idx"]]
            for ep in episodes})
    _json(registry, source)
    split = build_followup_split(source, {
        "discovery_per_task": 1, "validation_per_task": 1, "evaluation_per_task": 1,
        "split_seed": 2026, "frozen_before_pilot": False,
        "evaluation_labels_previously_used": None})
    _json(split_path, split)
    root = tmp_path / "results"
    root.mkdir()
    cfg_path = tmp_path / "head.yaml"
    cfg = {"inputs": {"dense_dir": str(dense), "source_episode_manifest": str(registry),
                       "sae_checkpoint": str(weights)}, "output": {"root_dir": str(root)},
           "sampling": {"mode": "followup", "task_ids": [0, 1], "split_manifest": str(split_path),
                        "max_episodes_per_task": 1, "max_steps_per_episode": 3}}
    cfg_path.write_text(yaml.safe_dump(cfg))
    full = load_config(cfg_path)
    generation = build_generation_manifest(source)
    sample = build_readout_manifest(index_path, full["sampling"], generation,
                                    dense_dir=dense, source_episodes=registry)
    _json(root / "sample_manifest.json", sample)
    _json(root / "scores/scores.json", {"scope": {
        "task_ids": [0, 1], "dictionary_size": 2, "sae_sha256": sha256_file(weights),
        "discovery_manifest_hash": sample["discovery_manifest_hash"],
        "sample_manifest_hash": sample["sample_hash"], "split_manifest_hash": split["manifest_hash"]}})
    event_root = tmp_path / "events"
    topk_dir = event_root / "topk"
    topk_dir.mkdir(parents=True)
    torch.save({"episode_num": torch.tensor([row["episode_num"] for row in rows]),
                "task_id": torch.tensor([row["task_id"] for row in rows]),
                "step_in_episode": torch.tensor([row["step_in_episode"] for row in rows]),
                "top_feature_ids": torch.tensor([[row["task_id"]] for row in rows]),
                "top_feature_vals": torch.tensor([[float(6 - abs(row["step_in_episode"] - 5))] for row in rows])},
               topk_dir / "shard.pt")
    _json(topk_dir / "manifest.json", {
        "format": "token_topk_sparse_v1", "capture_target": "post_mlp_residual", "layer": 31,
        "dict_size": 2, "topk": 1, "sae_sha256": sha256_file(weights),
        "activation_index_sha256": sha256_file(index_path), "num_shards": 1,
        "shards": [{"path": "shard.pt"}]})
    events = [{"sample_id": f"s{ep['episode_num']}", "task_description": ep["task_description"],
               "task_id": ep["task_id"], "task_episode_idx": ep["task_episode_idx"],
               "episode_num": ep["episode_num"], "waypoint_rank": 0,
               "waypoint_step": 5, "progress_percent": 0.5, "num_steps": 11}
              for ep in episodes]
    _jsonl(event_root / "events/event_features.jsonl", events)
    _jsonl(event_root / "clusters/cluster_assignments.jsonl", [
        {"sample_id": row["sample_id"], "task_description": row["task_description"],
         "cluster_id": f"c{row['task_id']}"} for row in events])
    _jsonl(event_root / "clusters/clusters.jsonl", [
        {"cluster_id": f"c{task}", "task_description": f"task {task}", "episode_coverage": 1.0}
        for task in (0, 1)])
    return cfg_path, event_root, root, split


def test_recompute_exact_discovery_and_reuse(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    config, event_root, root, split = _fixture(tmp_path)
    report = build_discovery_event_scores(config, event_root)
    assert report["status"] == "created"
    assert report["discovery_episodes"] == 2
    assert report["coverage"]["task_timestep_counts"] == {"0": 11, "1": 11}
    result = torch.load(report["output"], map_location="cpu", weights_only=False)
    assert set(result["source"]["allowed_episode_nums"]) == {row["episode_num"] for row in split["splits"]["discovery"]}
    assert result["scope"]["split_manifest_hash"] == split["manifest_hash"]
    assert "sample_manifest_hash" not in result["scope"]  # Different timestep sample, same episodes.
    assert result["derivation"]["clustering_refit"] is False
    assert result["derivation"]["historical_state_identity_verified"] is False
    assert load_config(config)["inputs"]["event_scores_path"] == report["output"]
    import event_sae.scoring.score_matrix as scoring
    monkeypatch.setattr(scoring, "score_cluster_features", lambda **kwargs: pytest.fail("unexpected recomputation"))
    assert build_discovery_event_scores(config, event_root)["status"] == "reused"


def test_public_legacy_null_capture_target_uses_pinned_source_path_evidence(tmp_path):
    config, event_root, _, _ = _fixture(tmp_path)
    path = event_root / "topk/manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(capture_target=None,
                    dense_dir="/workspace/event-sae-spatial-repro/merged/sae_activations/post_mlp_residual")
    _json(path, manifest)
    result = build_discovery_event_scores(config, event_root)
    assert result["scope"]["capture_target_binding"] == "legacy_source_dense_path_plus_exact_index_and_sae_hashes"


def test_unknown_legacy_capture_target_blocks(tmp_path):
    config, event_root, _, _ = _fixture(tmp_path)
    path = event_root / "topk/manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(capture_target=None, dense_dir="/unknown")
    _json(path, manifest)
    with pytest.raises(ValueError, match="dense-path evidence"):
        build_discovery_event_scores(config, event_root)


@pytest.mark.parametrize("field,value,match", [
    ("activation_index_sha256", "0" * 64, "activation index SHA-256"),
    ("sae_sha256", "0" * 64, "SAE SHA-256"),
    ("dict_size", 3, "dictionary size"),
    ("layer", 30, "Top-K layer"),
])
def test_topk_provenance_mismatch_blocks(tmp_path, field, value, match):
    config, event_root, _, _ = _fixture(tmp_path)
    path = event_root / "topk/manifest.json"
    data = json.loads(path.read_text())
    data[field] = value
    _json(path, data)
    with pytest.raises(ValueError, match=match):
        build_discovery_event_scores(config, event_root)
    assert load_config(config)["inputs"]["event_scores_path"] is None


def test_head_discovery_binding_mismatch_blocks(tmp_path):
    config, event_root, root, _ = _fixture(tmp_path)
    path = root / "scores/scores.json"
    data = json.loads(path.read_text())
    data["scope"]["discovery_manifest_hash"] = "0" * 64
    _json(path, data)
    with pytest.raises(ValueError, match="Head scores do not match"):
        build_discovery_event_scores(config, event_root)


def test_partial_discovery_population_cannot_rebind(tmp_path):
    config, event_root, root, _ = _fixture(tmp_path)
    path = root / "sample_manifest.json"
    data = json.loads(path.read_text())
    data["discovery_episodes"] = data["discovery_episodes"][:1]
    _json(path, data)
    with pytest.raises(ValueError, match="complete discovery split"):
        build_discovery_event_scores(config, event_root)


def test_rehashed_registry_cannot_replace_frozen_split_source(tmp_path):
    config, event_root, _, _ = _fixture(tmp_path)
    path = tmp_path / "inputs/source_episodes.json"
    data = json.loads(path.read_text())
    data["new_provenance"] = "modified after freeze"
    data.pop("manifest_hash")
    data["manifest_hash"] = fingerprint(data)
    _json(path, data)
    with pytest.raises(ValueError, match="Frozen split source manifest"):
        build_discovery_event_scores(config, event_root)


def test_modified_registry_manifest_hash_blocks(tmp_path):
    config, event_root, _, _ = _fixture(tmp_path)
    path = tmp_path / "inputs/source_episodes.json"
    data = json.loads(path.read_text())
    data["manifest_hash"] = "0" * 64
    _json(path, data)
    with pytest.raises(ValueError, match="registry manifest hash"):
        build_discovery_event_scores(config, event_root)


def test_false_readout_origin_cannot_rebind_even_with_fresh_sample_hash(tmp_path):
    config, event_root, root, _ = _fixture(tmp_path)
    path = root / "sample_manifest.json"
    data = json.loads(path.read_text())
    data["readouts"][0]["source_run_idx"] = 1
    data["sample_hash"] = fingerprint({"mapping_hash": data["mapping_hash"],
                                       "sample": data["sample_spec"], "readouts": data["readouts"]})
    _json(path, data)
    head_path = root / "scores/scores.json"
    head = json.loads(head_path.read_text())
    head["scope"]["sample_manifest_hash"] = data["sample_hash"]
    _json(head_path, head)
    with pytest.raises(ValueError, match="readout/prompt: mismatched source_run_idx"):
        build_discovery_event_scores(config, event_root)


def test_changed_inputs_block_resume(tmp_path):
    config, event_root, _, _ = _fixture(tmp_path)
    build_discovery_event_scores(config, event_root)
    path = event_root / "clusters/clusters.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="stale or modified"):
        build_discovery_event_scores(config, event_root)


def test_modified_output_blocks_resume(tmp_path):
    config, event_root, _, _ = _fixture(tmp_path)
    result = build_discovery_event_scores(config, event_root)
    with open(result["output"], "ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="stale or modified"):
        build_discovery_event_scores(config, event_root)


def test_missing_discovery_timestep_blocks_publish(tmp_path):
    torch = pytest.importorskip("torch")
    config, event_root, _, split = _fixture(tmp_path)
    path = event_root / "topk/shard.pt"
    data = torch.load(path, map_location="cpu", weights_only=False)
    episode = split["splits"]["discovery"][0]["episode_num"]
    keep = ~((data["episode_num"] == episode) & (data["step_in_episode"] == 10))
    torch.save({key: value[keep] for key, value in data.items()}, path)
    with pytest.raises((ValueError, RuntimeError), match="timestep coverage|windows|events"):
        build_discovery_event_scores(config, event_root)
    assert load_config(config)["inputs"]["event_scores_path"] is None
    assert not (tmp_path / "inputs/discovery_event_scores.pt").exists()


def test_cli_help_is_torch_free():
    root = Path(__file__).resolve().parents[1]
    code = ("import runpy,sys; sys.argv=['headevents.py','--help']; "
            "\ntry: runpy.run_path('scripts/openvla/headevents.py',run_name='__main__')"
            "\nexcept SystemExit as e: assert e.code == 0"
            "\nassert 'torch' not in sys.modules; assert 'libero' not in sys.modules")
    result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
