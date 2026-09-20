import copy
import importlib.util
from dataclasses import asdict
from pathlib import Path

import pytest

from event_sae.openvla.eval.config import EnvConfig, RunConfig, resolve_trial_indices
from event_sae.research.output_head.config import DEFAULTS, validate_config
from event_sae.research.output_head.provenance import (
    assert_compatible, assert_safe_output, atomic_write_json, fingerprint, read_json,
)


def test_trial_indices_keep_original_ids_and_legacy_default():
    assert resolve_trial_indices(None, 3, 10) == [0, 1, 2]
    assert resolve_trial_indices([9, 2, 7], 3, 10) == [9, 2, 7]
    for values in ([], [1, 1], [-1, 2], [True, 2], [1.0, 2], [1]):
        with pytest.raises(ValueError):
            EnvConfig(num_trials_per_task=2, trial_indices=values)
    with pytest.raises(ValueError, match="exceeds"):
        resolve_trial_indices([10], 1, 10)


def test_legacy_fingerprint_omits_new_default_fields():
    # Both expensive execution and preflight scripts use the same protocol.
    root = Path(__file__).resolve().parents[1]
    for name in ("run_intervention_sweep", "run_intervention_development_validation"):
        spec = importlib.util.spec_from_file_location(name, root / f"scripts/openvla/{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg = RunConfig()
        expected = asdict(cfg)
        expected["model"].pop("attn_implementation")
        expected["model"].pop("local_files_only")
        expected["env"].pop("trial_indices")
        expected["env"].pop("per_episode_seed")
        expected["logging"].pop("root_dir")
        expected["env"]["resolved_task_ids"] = [0]
        assert module._protocol_config(cfg, [0]) == expected
        cfg.env.trial_indices = list(range(50))
        cfg.env.per_episode_seed = True
        assert module._protocol_config(cfg, [0])["env"]["per_episode_seed"] is True


@pytest.mark.parametrize("updates", [
    {"typo": 1}, {"model": {"allow_download": True}},
    {"scoring": {"pair_batch_size": 0}}, {"scoring": {"pair_batch_size": 2}},
    {"scoring": {"alpha": .5}},
    {"rollout": {"require_head_parity": False}}, {"scope": {"retrain_sae": True}},
    {"sampling": {"task_ids": [0, 0]}}, {"output": {"overwrite": True}},
    {"scoring": {"max_scored_pairs": True}},
])
def test_strict_config(updates):
    with pytest.raises(ValueError):
        validate_config(updates)


def test_defaults_and_path_protection(tmp_path):
    assert validate_config({}) == DEFAULTS
    source = tmp_path / "source"
    source.mkdir()
    link = tmp_path / "link"
    link.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="overlaps"):
        assert_safe_output(link / "new", [source])
    with pytest.raises(ValueError, match="overlaps"):
        assert_safe_output(tmp_path, [source / "ae.pt"])


def test_atomic_and_identity(tmp_path):
    path = tmp_path / "result.json"
    atomic_write_json(path, {"identity": "a"})
    with pytest.raises(FileExistsError):
        atomic_write_json(path, {"identity": "b"})
    assert read_json(path) == {"identity": "a"}
    assert fingerprint({"b": 1, "a": 2}) == fingerprint({"a": 2, "b": 1})
    for field in ("sae", "head", "mapping", "alpha", "dtype", "sample"):
        actual = {field: "old"}
        with pytest.raises(ValueError, match="Stale"):
            assert_compatible(actual, {field: "new"})
