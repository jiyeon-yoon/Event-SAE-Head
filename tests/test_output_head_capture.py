"""Runtime input capture uses a live-style env/processor boundary, never a model."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from event_sae.research.output_head.capture import capture_runtime_inputs
from event_sae.research.output_head.config import load_config
from event_sae.research.output_head.workflow import (
    PROCESSOR_IDENTITY_FILES, build_local_processor_identity,
)


class FakeEnv:
    def __init__(self, closed):
        self.closed = closed
        self.steps = 0

    def seed(self, _seed):
        pass

    def reset(self):
        pass

    def set_init_state(self, _state):
        return self._obs()

    def step(self, _action):
        self.steps += 1
        return self._obs(), 0.0, False, {}

    def close(self):
        self.closed.append(self.steps)

    @staticmethod
    def _obs():
        return {
            "robot0_eef_pos": np.zeros(3, dtype=np.float32),
            "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=np.float32),
            "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
        }


class FakeSuite:
    n_tasks = 10

    @staticmethod
    def get_task(task_id):
        return SimpleNamespace(task_id=task_id)

    @staticmethod
    def get_task_init_states(task_id):
        return [np.array([task_id, 0.25], dtype=np.float32)]


def _configured(tmp_path):
    snapshot = tmp_path / "inputs" / "snapshot"
    snapshot.mkdir(parents=True)
    for index, name in enumerate(PROCESSOR_IDENTITY_FILES):
        (snapshot / name).write_bytes(f"processor-{index}".encode())
    metadata = snapshot.parent / "head.json"
    metadata.write_text("{}")
    config = tmp_path / "head.yaml"
    config.write_text(yaml.safe_dump({
        "inputs": {"local_model_snapshot": str(snapshot),
                   "head_export_metadata": str(metadata)},
        "sampling": {"task_ids": [0, 1]},
        "rollout": {"env_seed": 0},
        "output": {"root_dir": str(tmp_path / "results")},
    }, sort_keys=False))
    cfg = load_config(config)
    identity = build_local_processor_identity(cfg, snapshot)
    head = tmp_path / "results" / "head"
    head.mkdir(parents=True)
    (head / "output_head_manifest.json").write_text(json.dumps({
        "processor_identity": identity["identity"],
        "model_revision": cfg["model"]["revision"],
        "code_revision": cfg["model"]["code_revision"],
    }))
    return cfg, config, snapshot


def _runtime(cfg, closed):
    model = SimpleNamespace(
        revision=cfg["model"]["revision"], code_revision=cfg["model"]["code_revision"],
        checkpoint="remote", local_files_only=False, attn_implementation=None,
        family="openvla", center_crop=True,
    )
    env = SimpleNamespace(task_suite_name="libero_spatial", seed=0,
                          per_episode_seed=False, num_steps_wait=3)

    def get_action(proxy, _processor, _base, _observation, _description, *, unnorm_key):
        proxy.predict_action(
            input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
            attention_mask=torch.ones((1, 3), dtype=torch.bool),
            pixel_values=torch.ones((1, 6, 4, 4), dtype=torch.bfloat16),
            unnorm_key=unnorm_key, do_sample=False,
        )

    def resolve_task_ids(task_ids, num_tasks):
        selected = list(task_ids)
        if any(task_id < 0 or task_id >= num_tasks for task_id in selected):
            raise ValueError("task id outside suite")
        return selected

    return SimpleNamespace(
        load_eval_config=lambda _path: SimpleNamespace(model=model, env=env),
        get_processor=lambda _base: object(), make_suite=lambda _name: FakeSuite(),
        get_env=lambda task, resolution: (FakeEnv(closed), f"task {task.task_id}"),
        get_dummy_action=lambda: [0, 0, 0, 0, 0, 0, -1],
        get_image=lambda _obs, _size: np.zeros((4, 4, 3), dtype=np.uint8),
        quat2axisangle=lambda _quat: np.zeros(3, dtype=np.float32),
        get_action=get_action,
        initial_state_sha256=lambda state: __import__("hashlib").sha256(state.tobytes()).hexdigest(),
        get_resize_size=lambda _family: 224, set_seed=lambda _seed: None,
        resolve_task_ids=resolve_task_ids,
    )


def test_capture_writes_two_tensor_calls_and_updates_only_local_input(tmp_path):
    cfg, config, _snapshot = _configured(tmp_path)
    closed = []
    result = capture_runtime_inputs(cfg, config, allow_simulator=True,
                                    runtime=_runtime(cfg, closed))
    assert result["status"] == "captured"
    assert result["num_calls"] == 2
    assert result["model_action_queries_budget"] == 8
    assert result["policy_model_loaded"] is False
    assert closed == [3, 3]
    fixture = torch.load(result["runtime_inputs"], map_location="cpu", weights_only=True)
    assert fixture["synthetic"] is True
    assert len(fixture["calls"]) == len(fixture["episodes"]) == 2
    assert fixture["calls"][0]["pixel_values"].dtype == torch.bfloat16
    assert fixture["capture"]["policy_action_executed"] is False
    updated = yaml.safe_load(config.read_text())
    assert updated["inputs"]["runtime_inputs"] == result["runtime_inputs"]
    assert Path(result["runtime_inputs"]).parent != Path(updated["output"]["root_dir"])
    assert result["synthetic"] is True

    from event_sae.research.output_head.runtime import validate_runtime
    with pytest.raises(ValueError, match="non-synthetic runtime"):
        validate_runtime(load_config(config))


def test_capture_guard_and_processor_identity_fail_before_simulator(tmp_path):
    with pytest.raises(ValueError, match="allow-simulator"):
        capture_runtime_inputs({}, tmp_path / "missing", allow_simulator=False)
    cfg, config, snapshot = _configured(tmp_path)
    (snapshot / "tokenizer.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="processor identity"):
        capture_runtime_inputs(cfg, config, allow_simulator=True,
                               runtime=SimpleNamespace())
    assert not (snapshot.parent / "runtime.pt").exists()
