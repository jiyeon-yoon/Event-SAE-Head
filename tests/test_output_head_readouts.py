import copy
import hashlib
import json

import pytest
import torch

from event_sae.research.output_head.readouts import (
    build_readout_manifest,
    discovery_population_hash,
    load_readout_cache,
    prepare_readout_cache,
)
from event_sae.research.output_head.splits import build_split_manifest


def generation(**overrides):
    return {"layer_idx": 31, "action_dim": 7, "batch_size": 1,
            "padding": "none", "use_cache": True,
            "source_run_id": "toy-run", "suite": "toy", "evidence": "synthetic fixture",
            **overrides}


def index_records(episodes=1, steps=1, prefill=5, action_dim=7):
    result, row_start, forward = [], 0, 10
    for episode in range(episodes):
        for step in range(steps):
            for dim in range(action_dim):
                count = prefill + episode if dim == 0 else 1
                result.append({"layer_idx": 31, "shard_path": "dense.pt",
                               "row_start": row_start, "row_end": row_start + count,
                               "episode_num": episode, "step_in_episode": step,
                               "task_id": 0, "task_episode_idx": episode,
                               "global_forward_idx": forward})
                row_start += count
                forward += 2  # Deliberate forward-ID gaps must not be guessed as missing rows.
    return result


def test_readouts_use_each_complete_forward_last_row_not_fixed_offsets():
    records = index_records(episodes=2)
    result = build_readout_manifest(records, {"synthetic": True}, generation())
    assert len(result["readouts"]) == 14
    assert result["readouts"][0]["source_row"] == 4
    assert result["readouts"][7]["source_row"] == 16
    assert [r["action_dim_index"] for r in result["readouts"]] == list(range(7)) * 2
    assert result["audit"]["forward_id_gaps"]["toy-run"] > 0


@pytest.mark.parametrize("change", ["unknown_batch", "padded", "no_evidence", "no_cache", "no_run"])
def test_generation_mapping_requires_explicit_evidence(change):
    spec = generation()
    for key, value in {"unknown_batch": ("batch_size", None), "padded": ("padding", "left"),
                       "no_evidence": ("evidence", None), "no_cache": ("use_cache", False),
                       "no_run": ("source_run_id", None)}.items():
        if key == change:
            spec[value[0]] = value[1]
    with pytest.raises(ValueError):
        build_readout_manifest(index_records(), {}, spec)


@pytest.mark.parametrize("dimensions", [6, 8])
def test_incomplete_or_extra_prediction_forwards_fail(dimensions):
    with pytest.raises(ValueError, match="prediction forwards"):
        build_readout_manifest(index_records(action_dim=dimensions), {}, generation())


@pytest.mark.parametrize("change", ["overlap", "gap", "duplicate_forward", "wrong_layer", "unknown_trial", "cached_rows"])
def test_corrupted_indexes_do_not_silently_repair(change):
    rows = index_records()
    if change == "overlap":
        rows[1]["row_start"] = rows[0]["row_start"]
    elif change == "gap":
        for row in rows:
            row["row_start"] += 1
            row["row_end"] += 1
    elif change == "duplicate_forward":
        rows[1]["global_forward_idx"] = rows[0]["global_forward_idx"]
    elif change == "wrong_layer":
        rows[0]["layer_idx"] = 30
    elif change == "unknown_trial":
        rows[0]["task_episode_idx"] = None
    else:
        rows[-1]["row_end"] += 1
    with pytest.raises(ValueError):
        build_readout_manifest(rows, {}, generation())


def test_exclusion_mode_reports_population_change():
    rows = index_records(episodes=2)
    rows.pop()
    result = build_readout_manifest(rows, {}, generation(), strict=False)
    assert result["audit"]["exclusion_fraction"] == 0.5
    assert len(result["readouts"]) == 7


def test_merged_sources_with_same_local_ids_remain_distinct():
    rows = index_records()
    second = copy.deepcopy(rows)
    for row in rows:
        row["source_run_idx"] = 0
    for row in second:
        row["source_run_idx"] = 1
        row["shard_path"] = "other.pt"
    with pytest.raises(ValueError, match="dataset_id"):
        build_readout_manifest(rows + second, {}, generation())
    result = build_readout_manifest(rows + second, {}, generation(dataset_id="toy-merged"))
    assert len({row["episode_uid"] for row in result["readouts"]}) == 2


def test_episode_and_even_step_sampling_is_seeded_and_query_complete():
    rows = index_records(episodes=5, steps=9)
    sampling = {"sample_seed": 22, "max_episodes_per_task": 2, "max_steps_per_episode": 3}
    first = build_readout_manifest(rows, sampling, generation())
    second = build_readout_manifest(list(reversed(rows)), sampling, generation())
    assert first["readouts"] == second["readouts"]
    assert len(first["readouts"]) == 42
    assert {row["step_in_episode"] for row in first["readouts"]} == {0, 4, 8}


def test_discovery_scope_hash_does_not_depend_on_readout_step_sampling():
    rows = index_records(episodes=5, steps=9)
    common = {"sample_seed": 22, "max_episodes_per_task": 2}
    small = build_readout_manifest(rows, {**common, "max_steps_per_episode": 2}, generation())
    larger = build_readout_manifest(rows, {**common, "max_steps_per_episode": 7}, generation())
    assert small["discovery_manifest_hash"] == larger["discovery_manifest_hash"]
    assert small["sample_hash"] != larger["sample_hash"]
    assert small["discovery_episodes"] == larger["discovery_episodes"]
    renamed = [{**row, "source_run_id": "other", "episode_uid": "renumbered"}
               for row in small["discovery_episodes"]]
    assert discovery_population_hash(renamed) == small["discovery_manifest_hash"]


def test_missing_shard_is_caught_without_loading_tensors(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing dense shard"):
        build_readout_manifest(index_records(), {}, generation(), dense_dir=tmp_path)


def test_confirmatory_sampling_filters_before_episode_sampling():
    rows = index_records(episodes=6)
    episodes = [{"suite": "toy", "source_run_id": "toy-run", "episode_num": episode,
                 "task_id": 0, "task_episode_idx": episode,
                 "initial_state_sha256": hashlib.sha256(str(episode).encode()).hexdigest()}
                for episode in range(6)]
    split = build_split_manifest({"episodes": episodes, "initial_state_hash_provenance": "synthetic"},
                                 {"discovery_per_task": 3, "validation_per_task": 1,
                                  "frozen_before_pilot": True,
                                  "evaluation_per_task": 2, "evaluation_labels_previously_used": False})
    result = build_readout_manifest(rows, {"mode": "confirmatory", "split_manifest": split},
                                    generation(), source_episodes=episodes)
    expected = {row["task_episode_idx"] for row in split["splits"]["discovery"]}
    assert {row["task_episode_idx"] for row in result["readouts"]} == expected
    assert result["sample_spec"]["split_manifest_hash"] == split["manifest_hash"]


class IndependentSAE(torch.nn.Module):
    def encode(self, values):
        return torch.cat([values.abs(), values.abs() + 1, values.abs() + 2], dim=-1)


class BatchDependentSAE(torch.nn.Module):
    def encode(self, values):
        return values.abs() + values.abs().mean(dim=0, keepdim=True)


def tensor_fixture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    rows = index_records(episodes=2)
    tensor = torch.arange(rows[-1]["row_end"] * 2, dtype=torch.float32).reshape(-1, 2)
    torch.save(tensor, source / "dense.pt")
    result = build_readout_manifest(rows, {"synthetic": True}, generation(), dense_dir=source)
    return result, tensor


def test_cache_preserves_every_nonzero_and_roundtrips(tmp_path):
    manifest, dense = tensor_fixture(tmp_path)
    sae = IndependentSAE().eval()
    cache = prepare_readout_cache(manifest, sae, tmp_path / "cache", sae_id="toy-a",
                                   hidden_dtype="float32", rows_per_shard=5)
    assert cache["grouping_check"]["readout_only_exact"] is True
    assert cache["dict_size"] == 6
    assert len(cache["feature_ids"][0]) == 6  # No legacy top-K truncation.
    for i, row in enumerate(manifest["readouts"]):
        expected = dense[row["source_row"]]
        assert torch.equal(cache["hidden"][i], expected)
        assert torch.equal(cache["feature_values"][i], sae.encode(expected[None])[0])
    loaded = load_readout_cache(tmp_path / "cache")
    assert torch.equal(loaded["hidden"], cache["hidden"])
    assert loaded["records"] == cache["records"]
    resumed = prepare_readout_cache(manifest, sae, tmp_path / "cache", sae_id="toy-a", hidden_dtype="float32")
    assert torch.equal(resumed["hidden"], cache["hidden"])


def test_batch_dependent_encoder_keeps_original_forward_latents(tmp_path):
    manifest, dense = tensor_fixture(tmp_path)
    sae = BatchDependentSAE().eval()
    cache = prepare_readout_cache(manifest, sae, sae_id="toy-b", hidden_dtype="float32")
    assert cache["grouping_check"]["readout_only_exact"] is False
    row = manifest["readouts"][0]
    expected = sae.encode(dense[row["row_start"]:row["row_end"]])[-1]
    assert torch.equal(cache["feature_values"][0], expected)
    with pytest.raises(ValueError, match="differs from original"):
        prepare_readout_cache(manifest, sae, sae_id="toy-b", hidden_dtype="float32", encoder_mode="readout_only")


def test_cache_refuses_provenance_mismatch_and_source_mutation(tmp_path):
    manifest, _ = tensor_fixture(tmp_path)
    sae = IndependentSAE().eval()
    prepare_readout_cache(manifest, sae, tmp_path / "cache", sae_id="toy-a", hidden_dtype="float32")
    with pytest.raises(ValueError, match="provenance mismatch"):
        prepare_readout_cache(manifest, sae, tmp_path / "cache", sae_id="toy-other", hidden_dtype="float32")
    torch.save(torch.ones((2, 2)), tmp_path / "source" / "dense.pt")
    with pytest.raises(ValueError, match="changed since mapping"):
        prepare_readout_cache(manifest, sae, tmp_path / "cache", sae_id="toy-a", hidden_dtype="float32")


def test_cache_output_cannot_modify_source_tree(tmp_path):
    manifest, _ = tensor_fixture(tmp_path)
    with pytest.raises(ValueError, match="outside existing input"):
        prepare_readout_cache(manifest, IndependentSAE().eval(), tmp_path / "source" / "cache",
                              sae_id="toy", hidden_dtype="float32")


def test_cache_load_detects_corruption(tmp_path):
    manifest, _ = tensor_fixture(tmp_path)
    prepare_readout_cache(manifest, IndependentSAE().eval(), tmp_path / "cache",
                          sae_id="toy", hidden_dtype="float32")
    (tmp_path / "cache" / "shard_000000.pt").write_bytes(b"broken")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_readout_cache(tmp_path / "cache")


def test_actual_tensor_shape_is_checked_before_encoding(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    torch.save(torch.zeros((2, 2)), source / "dense.pt")
    manifest = build_readout_manifest(index_records(), {}, generation(), dense_dir=source)
    with pytest.raises(ValueError, match="range exceeds"):
        prepare_readout_cache(manifest, IndependentSAE().eval(), sae_id="toy", hidden_dtype="float32")


def test_cache_manifest_tampering_is_rejected(tmp_path):
    manifest, _ = tensor_fixture(tmp_path)
    prepare_readout_cache(manifest, IndependentSAE().eval(), tmp_path / "cache",
                          sae_id="toy", hidden_dtype="float32")
    path = tmp_path / "cache" / "manifest.json"
    saved = json.loads(path.read_text())
    saved["hidden_dtype"] = "float16"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="disagrees with signed provenance"):
        load_readout_cache(tmp_path / "cache")


def test_prepare_detects_changed_index_file(tmp_path):
    manifest, _ = tensor_fixture(tmp_path)
    index_path = tmp_path / "source" / "activation_index.jsonl"
    rows = index_records(episodes=2)
    index_path.write_text("\n".join(json.dumps(row) for row in rows))
    manifest = build_readout_manifest(index_path, {}, generation())
    rows[0]["extra_metadata"] = "changed"
    index_path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="index changed"):
        prepare_readout_cache(manifest, IndependentSAE().eval(), sae_id="toy", hidden_dtype="float32")
