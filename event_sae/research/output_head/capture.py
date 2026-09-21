"""Capture real, preprocessed OpenVLA calls without loading policy weights."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .config import validate_config
from .provenance import atomic_write_text, read_json
from .workflow import (_head_path, build_local_processor_identity, required_input)


SCHEMA = "output_head_runtime_inputs_v1"


class _InputsCaptured(Exception):
    """Internal control flow: stop before a policy action is produced."""


class _CaptureProxy:
    def __init__(self, expected_unnorm_key: str):
        self.expected_unnorm_key = expected_unnorm_key
        self.call: dict[str, Any] | None = None

    def predict_action(self, *, unnorm_key: str, do_sample: bool, **kwargs):
        import torch

        if self.call is not None:
            raise ValueError("Capture proxy received more than one policy query")
        if unnorm_key != self.expected_unnorm_key or do_sample is not False:
            raise ValueError("Capture query differs from the deterministic OpenVLA protocol")
        if not kwargs or any(not torch.is_tensor(value) for value in kwargs.values()):
            raise ValueError("Captured predict_action kwargs must be nonempty tensor-only inputs")
        self.call = {key: value.detach().cpu().contiguous().clone()
                     for key, value in kwargs.items()}
        raise _InputsCaptured


def _offline_environment() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def _native_runtime() -> SimpleNamespace:
    """Import simulator/processor dependencies only after explicit approval."""
    from libero.libero import benchmark
    from event_sae.openvla.eval.config import (
        load_config as load_eval_config, resolve_task_ids,
    )
    from event_sae.openvla.eval.libero_utils import (
        get_libero_dummy_action, get_libero_env, get_libero_image, quat2axisangle,
    )
    from event_sae.openvla.eval.model import get_action, get_processor
    from event_sae.openvla.eval.runner import _initial_state_sha256
    from event_sae.openvla.eval.utils import get_resize_size, set_seed_everywhere

    return SimpleNamespace(
        make_suite=lambda name: benchmark.get_benchmark_dict()[name](),
        load_eval_config=load_eval_config,
        get_env=get_libero_env,
        get_dummy_action=get_libero_dummy_action,
        get_image=get_libero_image,
        quat2axisangle=quat2axisangle,
        get_action=get_action,
        get_processor=get_processor,
        initial_state_sha256=_initial_state_sha256,
        get_resize_size=get_resize_size,
        set_seed=set_seed_everywhere,
        resolve_task_ids=resolve_task_ids,
    )


def _capture_case_specs(cfg: dict) -> list[dict]:
    task_ids = list(cfg["sampling"]["task_ids"])
    max_inputs = cfg["validation"]["max_model_calls"] // 4
    if max_inputs < 1:
        raise ValueError("Validation budget cannot accommodate one four-condition input")
    task_ids = task_ids[:max_inputs]
    split_path = cfg["sampling"]["split_manifest"]
    mode = cfg["sampling"]["mode"]
    if mode in {"confirmatory", "followup"} and not split_path:
        raise ValueError(f"{mode} input capture requires a frozen split manifest")
    if not split_path:
        return [{"task_id": task_id, "task_episode_idx": 0, "expected_state_hash": None}
                for task_id in task_ids]

    from .splits import validate_split_manifest
    split = read_json(split_path)
    validate_split_manifest(split, require_confirmatory=mode != "followup", require_followup=mode == "followup")
    allowed = split["splits"]["discovery"] + split["splits"]["validation"]
    specs = []
    for task_id in task_ids:
        candidates = sorted((row for row in allowed
                             if row["suite"] == cfg["scope"]["suite"] and row["task_id"] == task_id),
                            key=lambda row: row["task_episode_idx"])
        if not candidates:
            raise ValueError(f"No discovery/validation capture state for task {task_id}")
        row = candidates[0]
        specs.append({"task_id": task_id, "task_episode_idx": row["task_episode_idx"],
                      "expected_state_hash": row["initial_state_sha256"]})
    return specs


def _validate_call(call: dict[str, Any]) -> dict[str, dict[str, Any]]:
    import torch

    if not call or any(not torch.is_tensor(value) for value in call.values()):
        raise ValueError("Captured call must be a nonempty flat tensor mapping")
    input_ids = call.get("input_ids")
    if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Captured input_ids must have unpadded batch size one")
    mask = call.get("attention_mask")
    if mask is not None and (mask.shape != input_ids.shape or not bool(mask.all())):
        raise ValueError("Captured attention mask must match input_ids and contain no padding")
    if "pixel_values" not in call:
        raise ValueError("Captured call lacks pixel_values")
    for name, value in call.items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"Captured floating tensor is non-finite: {name}")
    return {name: {"shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch.")}
            for name, value in call.items()}


def _image_sha256(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(array.shape).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _save_torch_once(path: Path, payload: dict) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Runtime input fixture already exists: {path}")
    fd, temporary = tempfile.mkstemp(prefix=".runtime-inputs-", suffix=".pt", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def capture_runtime_inputs(cfg: dict, config_path: str | Path, *, allow_simulator: bool,
                           output_path: str | Path | None = None,
                           runtime: SimpleNamespace | None = None) -> dict:
    """Capture two fresh simulator observations at the predict_action boundary."""
    if not allow_simulator:
        raise ValueError("Runtime input capture requires --allow-simulator")
    _offline_environment()
    import numpy as np
    import yaml

    snapshot = required_input(cfg, "local_model_snapshot")
    head_dir = _head_path(cfg)
    if head_dir.is_file():
        head_dir = head_dir.parent
    head_manifest = read_json(head_dir / "output_head_manifest.json")
    processor = build_local_processor_identity(cfg, snapshot)
    if head_manifest.get("processor_identity") != processor["identity"]:
        raise ValueError("Current processor identity differs from the exported output head")
    if (head_manifest.get("model_revision") != cfg["model"]["revision"] or
            head_manifest.get("code_revision") != cfg["model"]["code_revision"]):
        raise ValueError("Exported head model/code identity differs from capture config")

    metadata_path = required_input(cfg, "head_export_metadata")
    output = (Path(output_path).expanduser().resolve() if output_path else
              metadata_path.parent / "runtime.pt")
    if output.exists():
        raise FileExistsError(f"Runtime input fixture already exists: {output}")
    config_file = Path(config_path).expanduser().resolve()
    document = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    document.setdefault("inputs", {})["runtime_inputs"] = str(output)
    validate_config(document)

    synthetic_runtime = runtime is not None
    native = runtime or _native_runtime()
    base = native.load_eval_config(cfg["rollout"]["base_eval_config"])
    if (base.model.revision != cfg["model"]["revision"] or
            base.model.code_revision != cfg["model"]["code_revision"]):
        raise ValueError("Base eval model/code revisions differ from capture config")
    base.model.checkpoint = str(snapshot)
    base.model.local_files_only = True
    base.model.attn_implementation = "sdpa"
    base.env.task_suite_name = cfg["scope"]["suite"]
    base.env.seed = cfg["rollout"]["env_seed"]
    base.env.per_episode_seed = True
    native.set_seed(base.env.seed)
    policy_processor = native.get_processor(base)
    suite = native.make_suite(cfg["scope"]["suite"])
    if not hasattr(suite, "n_tasks"):
        raise ValueError("Task suite does not expose n_tasks for range validation")
    native.resolve_task_ids(cfg["sampling"]["task_ids"], suite.n_tasks)
    specs = _capture_case_specs(cfg)
    calls, episodes, call_provenance = [], [], []
    resize_size = native.get_resize_size(base.model.family)

    for call_index, spec in enumerate(specs):
        task_id, trial = spec["task_id"], spec["task_episode_idx"]
        task = suite.get_task(task_id)
        states = suite.get_task_init_states(task_id)
        if trial < 0 or trial >= len(states):
            raise ValueError(f"Capture trial is outside available initial states: {(task_id, trial)}")
        initial_state = states[trial]
        state_hash = native.initial_state_sha256(initial_state)
        if spec["expected_state_hash"] is not None and state_hash != spec["expected_state_hash"]:
            raise ValueError(f"Frozen capture state hash mismatch: {(task_id, trial)}")
        env, task_description = native.get_env(task, resolution=256)
        seed_material = f"{base.env.seed}:{task_id}:{trial}".encode()
        episode_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], "big")
        try:
            native.set_seed(episode_seed)
            env.seed(episode_seed)
            env.reset()
            obs = env.set_init_state(initial_state)
            for _ in range(base.env.num_steps_wait):
                obs, _reward, _done, _info = env.step(native.get_dummy_action())
            image = native.get_image(obs, resize_size)
            observation = {
                "full_image": image,
                "state": np.concatenate((obs["robot0_eef_pos"],
                                         native.quat2axisangle(obs["robot0_eef_quat"]),
                                         obs["robot0_gripper_qpos"])),
            }
            proxy = _CaptureProxy(cfg["model"]["unnorm_key"])
            try:
                native.get_action(proxy, policy_processor, base, observation, task_description,
                                  unnorm_key=cfg["model"]["unnorm_key"])
            except _InputsCaptured:
                pass
            else:
                raise RuntimeError("Capture proxy unexpectedly allowed a policy action")
            if proxy.call is None:
                raise RuntimeError("No predict_action inputs were captured")
            tensor_spec = _validate_call(proxy.call)
            calls.append(proxy.call)
            episodes.append({
                "suite": cfg["scope"]["suite"], "episode_num": call_index + 1,
                "task_id": task_id, "task_episode_idx": trial,
                "initial_state_sha256": state_hash, "call_index": call_index,
                "step_in_episode": 0,
                "initial_state_hash_provenance": "live_libero_task_suite_state_bytes_v1",
            })
            call_provenance.append({
                "call_index": call_index, "task_description": task_description,
                "episode_seed": episode_seed, "image_sha256": _image_sha256(image),
                "tensor_spec": tensor_spec,
            })
        finally:
            env.close()

    if not calls or len(calls) != len(episodes) or len(calls) * 4 > cfg["validation"]["max_model_calls"]:
        raise ValueError("Captured calls do not satisfy the four-condition validation budget")
    if cfg["sampling"]["split_manifest"]:
        from .splits import validate_selection_split
        validate_selection_split(read_json(cfg["sampling"]["split_manifest"]), episodes, stage="pilot",
                                 mode=cfg["sampling"]["mode"])
    fixture = {
        "schema_version": SCHEMA, "synthetic": synthetic_runtime,
        "model_revision": cfg["model"]["revision"],
        "code_revision": cfg["model"]["code_revision"],
        "processor_identity": processor["identity"],
        "calls": calls, "episodes": episodes,
        "capture": {
            "source": "fresh_libero_observation_after_native_wait",
            "policy_model_loaded": False, "policy_action_executed": False,
            "recovered_from_saved_jpeg_or_video": False,
            "renderer_resolution": 256, "resize_size": resize_size,
            "num_steps_wait": base.env.num_steps_wait,
            "base_seed": base.env.seed, "per_episode_seed": True,
            "center_crop": base.model.center_crop,
            "calls": call_provenance,
        },
    }
    _save_torch_once(output, fixture)
    atomic_write_text(config_file, yaml.safe_dump(document, sort_keys=False), overwrite=True)
    return {
        "status": "captured", "runtime_inputs": str(output),
        "num_calls": len(calls), "model_action_queries_budget": len(calls) * 4,
        "tasks": [row["task_id"] for row in episodes],
        "trials": [row["task_episode_idx"] for row in episodes],
        "policy_model_loaded": False, "policy_action_executed": False,
        "synthetic": synthetic_runtime,
        "processor_identity": processor["identity"],
    }
