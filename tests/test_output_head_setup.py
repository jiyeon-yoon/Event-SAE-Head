"""Local setup is deterministic, model-free, and preserves user config."""

import json
import importlib.util
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import save_file

from event_sae.research.output_head.config import load_config
from event_sae.research.output_head.workflow import (
    PROCESSOR_IDENTITY_FILES, setup_head_workflow,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/openvla/headsetup.py"


def _inputs(tmp_path: Path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps({
        "text_config": {"num_hidden_layers": 2, "rms_norm_eps": 1e-5,
                        "torch_dtype": "bfloat16", "hidden_size": 3, "vocab_size": 5},
    }))
    (snapshot / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "language_model.model.norm.weight": "model-01.safetensors",
        "language_model.lm_head.weight": "model-01.safetensors",
    }}))
    for index, name in enumerate(PROCESSOR_IDENTITY_FILES):
        (snapshot / name).write_bytes(f"processor-{index}".encode())
    config = tmp_path / "head.yaml"
    config.write_text(yaml.safe_dump({
        "scope": {"layer_idx": 1},
        "sampling": {"task_ids": [0, 1]},
        "selection": {"top_k": 7},
        "output": {"root_dir": str(tmp_path / "results")},
    }, sort_keys=False))
    return snapshot, config


def test_setup_head_writes_stable_metadata_and_preserves_local_options(tmp_path):
    snapshot, config = _inputs(tmp_path)
    metadata = tmp_path / "inputs" / "head.json"
    first = setup_head_workflow(load_config(config), config, snapshot, metadata)
    second = setup_head_workflow(load_config(config), config, snapshot, metadata)
    assert first == second
    assert first["status"] == "configured"
    assert first["model_execution"] is False
    payload = json.loads(metadata.read_text())
    assert payload["target_layer"] == 1
    assert payload["norm_spec"]["eps"] == 1e-5
    assert payload["processor_identity"].startswith("sha256:")
    updated = yaml.safe_load(config.read_text())
    assert updated["selection"]["top_k"] == 7
    assert updated["sampling"]["task_ids"] == [0, 1]
    assert updated["inputs"]["local_model_snapshot"] == str(snapshot.resolve())
    assert updated["inputs"]["head_export_metadata"] == str(metadata.resolve())
    assert updated["scoring"]["device"] == "cuda:0"


def test_setup_head_refuses_changed_or_incomplete_processor_identity(tmp_path):
    snapshot, config = _inputs(tmp_path)
    metadata = tmp_path / "head.json"
    setup_head_workflow(load_config(config), config, snapshot, metadata)
    before_config = config.read_bytes()
    (snapshot / PROCESSOR_IDENTITY_FILES[0]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="incompatible existing artifact"):
        setup_head_workflow(load_config(config), config, snapshot, metadata)
    assert config.read_bytes() == before_config

    snapshot2, config2 = _inputs(tmp_path / "second")
    (snapshot2 / "preprocessor_config.json").unlink()
    with pytest.raises(FileNotFoundError, match="Required processor identity files missing"):
        setup_head_workflow(load_config(config2), config2, snapshot2)
    assert "local_model_snapshot" not in yaml.safe_load(config2.read_text()).get("inputs", {})


def test_setup_head_records_absent_optional_processor_files(tmp_path):
    snapshot, config = _inputs(tmp_path)
    (snapshot / "processor_config.json").unlink()
    (snapshot / "added_tokens.json").unlink()
    metadata = tmp_path / "head.json"
    setup_head_workflow(load_config(config), config, snapshot, metadata)
    files = json.loads(metadata.read_text())["processor_identity_evidence"]["files"]
    assert files["processor_config.json"] is None
    assert files["added_tokens.json"] is None


def test_short_setup_cli_can_export_toy_head(tmp_path, capsys):
    snapshot, config = _inputs(tmp_path)
    save_file({
        "language_model.model.norm.weight": torch.ones(3),
        "language_model.lm_head.weight": torch.ones(5, 3),
    }, snapshot / "model-01.safetensors")
    spec = importlib.util.spec_from_file_location("tested_headsetup", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.main(["--config", str(config), "--snapshot", str(snapshot), "--export"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["setup"]["status"] == "configured"
    assert payload["export"]["source_kind"] == "local_safetensors"
    assert (tmp_path / "results/head/output_head_manifest.json").is_file()
