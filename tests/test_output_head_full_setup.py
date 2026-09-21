"""CPU-only full10 orchestration tests; all model/source pins are fixture pins."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from event_sae.research.output_head.provenance import sha256_file
from event_sae.research.output_head import state_registry


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/openvla/headfull.py"


def load_script():
    spec = importlib.util.spec_from_file_location("headfull_test_entrypoint", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def full10(tmp_path, monkeypatch):
    script = load_script()
    p = script.paths(tmp_path)
    dense = p["merged"] / "sae_activations/post_mlp_residual"
    dense.mkdir(parents=True)
    prompts, index, states, offsets = [], [], {}, {}
    for task in range(10):
        for trial in range(50):
            episode = task * 50 + trial + 1
            prompt = {"source_run_idx": task // 2, "source_episode_num": (task % 2) * 50 + trial + 1,
                      "episode_num": episode, "task_id": task, "task_episode_idx": trial}
            prompts.append(prompt)
            states[("libero_spatial", task, trial)] = np.asarray([task, trial, 0.25], dtype=np.float64)
            for dimension in range(7):
                forward = (episode - 1) * 7 + dimension
                shard = f"fixture_{forward * 369 // 3500:04d}.pt"
                start = offsets.get(shard, 0)
                end = start + (9 if dimension == 0 else 1)
                offsets[shard] = end
                index.append({**prompt, "layer_idx": 31, "step_in_episode": 0,
                              "global_forward_idx": forward + 1, "shard_path": shard,
                              "row_start": start, "row_end": end})
    for name in offsets:
        (dense / name).write_bytes(b"fixture, never loaded as tensor")
    prompt_path = p["merged"] / "prompt_records.jsonl"
    index_path = dense / "activation_index.jsonl"
    prompt_path.write_text("".join(json.dumps(row) + "\n" for row in prompts))
    index_path.write_text("".join(json.dumps(row) + "\n" for row in index))
    p["sae"].parent.mkdir(parents=True)
    p["sae"].write_bytes(b"fixture, never loaded as SAE")
    monkeypatch.setattr(script, "INDEX_SHA", sha256_file(index_path))
    monkeypatch.setattr(script, "SAE_SHA", sha256_file(p["sae"]))
    monkeypatch.setattr(script, "check_merged", lambda paths: {"num_episodes": 500, "num_activation_shards": 369})
    public_path = p["events"] / "topk/manifest.json"
    public_path.parent.mkdir(parents=True)
    public_path.write_text(json.dumps({"activation_index_sha256": script.INDEX_SHA,
                                       "sae_sha256": script.SAE_SHA}))
    monkeypatch.setattr(state_registry, "_native_state_loader",
                        lambda suite, tasks: (states, {"source": "synthetic_test_arrays", "git_revision": None}))
    return script, tmp_path, tmp_path / "local/head-full10.yaml", p, states


def artifact_snapshot(directory):
    return {str(path.relative_to(directory)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in directory.rglob("*") if path.is_file()}


def test_full10_setup_freezes_250_50_200_and_repeat_is_readonly(full10):
    script, workspace, config, p, states = full10
    output = script.setup(workspace, config, allow_libero=True)
    assert output["status"] == "configured"
    assert output["discovery_episodes"] == 250
    assert output["validation_episodes"] == 50
    assert output["evaluation_cases"] == 200
    assert output["max_rollouts"] == 4000
    assert output["historical_state_identity_verified"] is False
    assert output["historical_generation_flags_verified"] is False
    split_path = p["metadata"] / "split.json"
    split = json.loads(split_path.read_text())
    assert split["schema_version"] == "output_head_followup_split_v1"
    assert split["evaluation_labels_previously_used"] is True
    keys = [set((row["task_id"], row["task_episode_idx"]) for row in split["splits"][name])
            for name in ("discovery", "validation", "evaluation")]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
    before = artifact_snapshot(p["metadata"])
    config_before = config.read_bytes(), config.stat().st_mtime_ns
    again = script.setup(workspace, config, allow_libero=True)
    assert again == output
    assert artifact_snapshot(p["metadata"]) == before
    assert (config.read_bytes(), config.stat().st_mtime_ns) == config_before


def test_setup_without_libero_gate_has_no_side_effect(tmp_path):
    script = load_script()
    with pytest.raises(ValueError, match="allow-libero"):
        script.setup(tmp_path, tmp_path / "head.yaml", allow_libero=False)
    assert list(tmp_path.iterdir()) == []


def test_new_split_over_existing_results_is_rejected(full10):
    script, workspace, config, p, _ = full10
    p["results"].mkdir(parents=True)
    (p["results"] / "scores.json").write_text("{}")
    with pytest.raises(ValueError, match="existing experiment"):
        script.setup(workspace, config, allow_libero=True)
    assert not p["metadata"].exists()
    assert not config.exists()


def test_changed_runtime_state_cannot_replace_frozen_split(full10):
    script, workspace, config, p, states = full10
    script.setup(workspace, config, allow_libero=True)
    before = artifact_snapshot(p["metadata"])
    states[("libero_spatial", 8, 14)] += 0.1
    with pytest.raises(ValueError, match="frozen split differs"):
        script.setup(workspace, config, allow_libero=True)
    assert artifact_snapshot(p["metadata"]) == before


def test_changed_local_settings_cannot_be_silently_overwritten(full10):
    import yaml
    script, workspace, config, p, _ = full10
    script.setup(workspace, config, allow_libero=True)
    cfg = yaml.safe_load(config.read_text())
    cfg["selection"]["top_k"] = 2
    config.write_text(yaml.safe_dump(cfg))
    before = config.read_bytes()
    with pytest.raises(ValueError, match="Existing local config differs"):
        script.setup(workspace, config, allow_libero=True)
    assert config.read_bytes() == before


def test_subsequent_pipeline_paths_are_preserved(full10):
    import yaml
    script, workspace, config, p, _ = full10
    script.setup(workspace, config, allow_libero=True)
    cfg = yaml.safe_load(config.read_text())
    for field in ("event_scores_path", "output_head_bundle", "runtime_inputs"):
        cfg["inputs"][field] = str(workspace / "later" / field)
    config.write_text(yaml.safe_dump(cfg))
    before = config.read_bytes(), config.stat().st_mtime_ns
    script.setup(workspace, config, allow_libero=True)
    assert (config.read_bytes(), config.stat().st_mtime_ns) == before


def test_download_preview_imports_no_hub_model_or_simulator_and_writes_nothing(tmp_path):
    program = """
import importlib.util, json, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location('headfull_preview', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result = module.download(Path(sys.argv[2]), execute=False)
for prefix in ('torch', 'tensorflow', 'huggingface_hub', 'libero', 'transformers', 'event_sae'):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules), prefix
print(json.dumps(result))
"""
    result = subprocess.run([sys.executable, "-I", "-c", program, str(SCRIPT), str(tmp_path)],
                            text=True, capture_output=True, check=True)
    preview = json.loads(result.stdout)
    assert preview["status"] == "download_plan"
    assert preview["execute"] is False
    assert preview["gpu_execution"] is False
    assert list(tmp_path.iterdir()) == []


def test_code_cache_alias_is_pinned_idempotent_and_never_overwrites_conflict(tmp_path):
    script = load_script()
    snapshot = tmp_path / "models--openvla--openvla-7b/snapshots" / script.CODE_REVISION
    snapshot.mkdir(parents=True)
    script.pin_nested_code_reference(snapshot)
    alias = snapshot.parent.parent / "refs/main"
    assert alias.read_text().strip() == script.CODE_REVISION
    before = alias.read_bytes(), alias.stat().st_mtime_ns
    script.pin_nested_code_reference(snapshot)
    assert (alias.read_bytes(), alias.stat().st_mtime_ns) == before
    alias.write_text("another-existing-revision")
    with pytest.raises(ValueError, match="Conflicting custom-code"):
        script.pin_nested_code_reference(snapshot)
    assert alias.read_text() == "another-existing-revision"
    with pytest.raises(ValueError, match="Unexpected pinned"):
        script.pin_nested_code_reference(snapshot.with_name("unapproved-revision"))
