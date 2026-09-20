"""CLI smoke tests run in fresh processes with heavyweight imports prohibited."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from event_sae.research.output_head.config import validate_config
from event_sae.research.output_head.provenance import atomic_write_json, fingerprint
from event_sae.research.output_head.runtime import verify_approved_plan


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/openvla/output_head_sensitivity.py"
GUARD = """
import importlib.abc, runpy, sys
class ForbidHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'tensorflow', 'libero', 'transformers', 'dictionary_learning'}:
            raise RuntimeError('FORBIDDEN HEAVY IMPORT: ' + fullname)
sys.meta_path.insert(0, ForbidHeavy())
sys.argv = [sys.argv[1], *sys.argv[2:]]
runpy.run_path(sys.argv[0], run_name='__main__')
"""


def run_cli(*args):
    return subprocess.run([sys.executable, "-c", GUARD, str(SCRIPT), *args], cwd=ROOT,
                          capture_output=True, text=True, timeout=30)


def test_help_never_imports_model_or_simulator():
    result = run_cli("--help")
    assert result.returncode == 0, result.stderr
    assert "export-head" in result.stdout


def test_audit_reports_missing_inputs_without_heavy_imports():
    result = run_cli("audit", "--config", "configs/research/openvla/output_head_sensitivity.yaml")
    assert result.returncode == 2
    assert json.loads(result.stdout)["runtime_validation"] == "not_run"
    assert "FORBIDDEN" not in result.stderr


def test_plan_empty_inputs_is_explicitly_blocked(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"output": {"root_dir": str(tmp_path / "output")}}))
    result = run_cli("plan", "--config", str(config))
    assert result.returncode == 2
    value = json.loads(result.stdout)
    assert value["total_rollouts"] is None
    assert value["execute"] is False


def test_split_cli_is_metadata_only(tmp_path):
    source = tmp_path / "episodes.json"
    atomic_write_json(source, {"synthetic": True, "initial_state_hash_verified": True,
                              "episodes": [{"suite": "libero_spatial", "task_id": 0,
                                            "task_episode_idx": i, "initial_state_sha256": f"{i:064x}"}
                                           for i in range(3)]})
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"inputs": {"source_episode_manifest": str(source)},
                                     "output": {"root_dir": str(tmp_path / "output")},
                                     "splits": {"discovery_per_task": 1, "validation_per_task": 1,
                                                "evaluation_per_task": 1}}))
    result = run_cli("split", "--config", str(config))
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "output/split_manifest.json").is_file()


@pytest.mark.parametrize("command", ["run", "validate"])
def test_explicit_execution_guard_precedes_runtime_import(command):
    result = run_cli(command, "--config", "configs/research/openvla/output_head_sensitivity.yaml")
    assert result.returncode == 2
    assert "FORBIDDEN" not in result.stderr
    assert "requires" in json.loads(result.stderr)["error"]


def test_eval_config_import_is_lazy_in_fresh_process():
    code = GUARD.split("sys.argv =")[0] + "\nfrom event_sae.openvla.eval.config import EnvConfig\nprint(EnvConfig().trial_indices)"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "None"


def test_approved_plan_rejects_tamper_without_model(tmp_path):
    cfg = validate_config({"output": {"root_dir": str(tmp_path / "output")}})
    plan = {"synthetic": False, "status": "planned", "total_rollouts": 3}
    plan["plan_hash"] = fingerprint(plan)
    atomic_write_json(tmp_path / "output/plan/rollout_plan.json", plan)
    with pytest.raises(ValueError, match="hash"):
        verify_approved_plan(cfg, "incorrect", "revision")
    plan["total_rollouts"] = 999
    (tmp_path / "output/plan/rollout_plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="tampered"):
        verify_approved_plan(cfg, plan["plan_hash"], "revision")
