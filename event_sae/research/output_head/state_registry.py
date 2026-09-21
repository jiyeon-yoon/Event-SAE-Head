"""Join legacy collection identities to a separately identified current LIBERO.

No policy model is loaded. JSONL files are streamed; dense tensors are never read.
Current initial-state bytes cannot establish the state used in a historical run.
That distinction is retained in every episode and in the manifest scope.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .provenance import fingerprint, sha256_file


SCHEMA = "output_head_current_state_registry_v1"
DEFAULT_DATASET_ID = "jiyeony/libero-spatial-openvla-rollouts-500"
PROVENANCE = "current_libero_state_registry_v1"


def initial_state_sha256(value: Any) -> str:
    """Exactly the dtype/shape/contiguous-byte hash used by eval.runner."""
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(array.shape).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _sha(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError(f"{name} must be a SHA-256 digest or null")
    return value.lower()


def _records(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid JSON in {path.name}:{number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object in {path.name}:{number}")
            yield row


def _native_state_loader(suite: str, task_ids: list[int]) -> tuple[dict, dict]:
    """Explicit caller gate must precede this lazy LIBERO import."""
    from libero.libero import benchmark, get_libero_path

    task_suite = benchmark.get_benchmark_dict()[suite]()
    values, initial_files = {}, {}
    for task_id in task_ids:
        states = task_suite.get_task_init_states(task_id)
        for trial, state in enumerate(states):
            values[(suite, task_id, trial)] = state
        task = task_suite.get_task(task_id)
        path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        if not path.is_file():
            raise FileNotFoundError(f"Loaded LIBERO initial-state file is unavailable: {path}")
        initial_files[str(task_id)] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    module_path = Path(inspect.getfile(benchmark)).resolve()
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(module_path.parent), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(module_path.parent), "status", "--porcelain"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return values, {"source": "live_libero_initial_state_files", "git_revision": revision,
                    "git_dirty": dirty, "benchmark_module": str(module_path),
                    "benchmark_module_sha256": sha256_file(module_path),
                    "initial_state_files": initial_files}


def _expected(values: Iterable[int] | int | None, field: str) -> set[int] | None:
    if values is None:
        return None
    if isinstance(values, int) and not isinstance(values, bool):
        values = range(_int(values, field))
    result = list(values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{field} must specify unique, nonempty IDs")
    return {_int(value, field) for value in result}


def build_state_registry(
    merged_run_dir: str | Path, *, runtime_states: Mapping | Callable | None = None,
    native_state_loader: Callable | None = None,
    expected_tasks: Iterable[int] | int | None = range(10),
    expected_trials: Iterable[int] | int | None = range(50),
    expected_shards: int | None = None, dataset_id: str = DEFAULT_DATASET_ID,
    suite: str = "libero_spatial", layer_idx: int = 31, action_dim: int = 7,
) -> dict:
    """Check every legacy join/query, then hash current runtime state arrays.

``runtime_states`` accepts raw arrays keyed by (suite, task, trial), or a callable
with those three positional arguments. It is useful for CPU-only verification.
Otherwise ``native_state_loader(suite, task_ids)`` returns (states, evidence);
the default imports LIBERO lazily. The CLI must gate that call explicitly.
Expected task/trial arguments accept an ID iterable, count, or None.
"""
    root = Path(merged_run_dir).expanduser().resolve()
    prompt_path = root / "prompt_records.jsonl"
    index_path = root / "sae_activations/post_mlp_residual/activation_index.jsonl"
    if not isinstance(dataset_id, str) or not dataset_id or not isinstance(suite, str) or not suite:
        raise ValueError("dataset_id and suite must be nonempty")
    _int(layer_idx, "layer_idx")
    if _int(action_dim, "action_dim") < 1:
        raise ValueError("action_dim must be positive")
    wanted_tasks, wanted_trials = _expected(expected_tasks, "expected_tasks"), _expected(expected_trials, "expected_trials")
    prompts, originals, task_trials, global_ids = {}, set(), set(), set()
    for row in _records(prompt_path):
        run, episode = _int(row.get("source_run_idx"), "source_run_idx"), _int(row.get("episode_num"), "episode_num")
        original = _int(row.get("source_episode_num"), "source_episode_num")
        task, trial = _int(row.get("task_id"), "task_id"), _int(row.get("task_episode_idx"), "task_episode_idx")
        if row.get("suite", suite) != suite:
            raise ValueError("Prompt suite differs from requested suite")
        key = (run, episode)
        if key in prompts or episode in global_ids or (run, original) in originals or (task, trial) in task_trials:
            raise ValueError("Duplicate/ambiguous original prompt episode mapping")
        prompts[key] = {"source_run_idx": run, "source_run_id": f"{dataset_id}:run:{run}",
                        "episode_num": episode, "source_episode_num": original, "suite": suite,
                        "task_id": task, "task_episode_idx": trial,
                        "historical_initial_state_sha256": _sha(row.get("initial_state_sha256"), "historical initial state")}
        originals.add((run, original))
        task_trials.add((task, trial))
        global_ids.add(episode)
    if not prompts:
        raise ValueError("Prompt records are empty")
    tasks = {task for task, _ in task_trials}
    if wanted_tasks is not None and tasks != wanted_tasks:
        raise ValueError("Prompt task coverage does not match expected_tasks")
    for task in tasks:
        if wanted_trials is not None and {trial for candidate, trial in task_trials if candidate == task} != wanted_trials:
            raise ValueError(f"Prompt trial coverage differs for task {task}")

    # Only compact identity/range metadata is retained. No dense tensor is loaded.
    steps, seen_forwards, ranges = defaultdict(list), set(), defaultdict(list)
    episodes_seen, row_counts, total = set(), Counter(), 0
    for row in _records(index_path):
        key = (_int(row.get("source_run_idx"), "source_run_idx"), _int(row.get("episode_num"), "episode_num"))
        if key not in prompts:
            raise ValueError(f"Activation index references unknown prompt episode {key}")
        prompt = prompts[key]
        for field in ("task_id", "task_episode_idx", "source_episode_num"):
            if _int(row.get(field), field) != prompt[field]:
                raise ValueError(f"Activation/prompt mismatch: {field} for {key}")
        if row.get("suite", suite) != suite or _int(row.get("layer_idx"), "layer_idx") != layer_idx:
            raise ValueError("Activation suite/layer differs from requested scope")
        historical = _sha(row.get("initial_state_sha256"), "index initial state")
        previous = prompt["historical_initial_state_sha256"]
        if historical is not None:
            if previous is not None and previous != historical:
                raise ValueError("Conflicting historical initial-state hashes")
            prompt["historical_initial_state_sha256"] = historical
        for field, expected in (("batch_size", 1), ("padding", "none"), ("use_cache", True)):
            if field in row and (row[field] != expected or (field == "batch_size" and isinstance(row[field], bool))):
                raise ValueError(f"Activation metadata contradicts generation mapping: {field}")
        step, forward = _int(row.get("step_in_episode"), "step_in_episode"), _int(row.get("global_forward_idx"), "global_forward_idx")
        if (key[0], forward) in seen_forwards:
            raise ValueError("Duplicate activation forward identity")
        seen_forwards.add((key[0], forward))
        start, end = _int(row.get("row_start"), "row_start"), _int(row.get("row_end"), "row_end")
        if end <= start:
            raise ValueError("Invalid activation row range")
        shard = row.get("shard_path")
        if not isinstance(shard, str) or not shard or Path(shard).is_absolute() or ".." in Path(shard).parts:
            raise ValueError("Activation shard_path must be a relative contained path")
        shard_path = index_path.parent / shard
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing dense shard: {shard_path}")
        ranges[shard].append((start, end))
        steps[(*key, step)].append((forward, end - start, row.get("action_dim_index")))
        episodes_seen.add(key)
        row_counts[end - start] += 1
        total += 1
    if episodes_seen != set(prompts):
        raise ValueError("Prompt and activation episode coverage differs")
    if expected_shards is not None and len(ranges) != _int(expected_shards, "expected_shards"):
        raise ValueError("Dense shard count differs from expected_shards")
    for shard, intervals in ranges.items():
        cursor = 0
        for start, end in sorted(intervals):
            if start != cursor:
                raise ValueError(f"Activation row gap/overlap in {shard}")
            cursor = end
    for key, forwards in steps.items():
        ordered = sorted(forwards)
        if len(ordered) != action_dim:
            raise ValueError(f"Expected {action_dim} generation forwards for query {key}; got {len(ordered)}")
        if any(count != 1 for _, count, _ in ordered[1:]):
            raise ValueError(f"Cached decode row count is not one for query {key}")
        if any(dim is not None and (isinstance(dim, bool) or dim != index) for index, (_, _, dim) in enumerate(ordered)):
            raise ValueError("Action dimension order contradicts forward order")

    if runtime_states is not None and native_state_loader is not None:
        raise ValueError("Provide runtime_states or native_state_loader, not both")
    if runtime_states is None:
        runtime_states, runtime_evidence = (native_state_loader or _native_state_loader)(suite, sorted(tasks))
    else:
        runtime_evidence = {"source": "caller_supplied_state_arrays", "git_revision": None}
    states_seen, episodes = set(), []
    for prompt in sorted(prompts.values(), key=lambda item: item["episode_num"]):
        key = (suite, prompt["task_id"], prompt["task_episode_idx"])
        try:
            state = runtime_states(*key) if callable(runtime_states) else runtime_states[key]
        except (IndexError, KeyError) as exc:
            raise ValueError(f"Missing current runtime initial state for {key}") from exc
        current_hash = initial_state_sha256(state)
        state_key = (suite, prompt["task_id"], current_hash)
        if state_key in states_seen:
            raise ValueError("Duplicate current initial-state bytes within one task")
        states_seen.add(state_key)
        historical_hash = prompt["historical_initial_state_sha256"]
        if historical_hash is not None and historical_hash != current_hash:
            raise ValueError(f"Historical/current initial-state mismatch for {key}")
        episodes.append({**prompt, "initial_state_sha256": current_hash,
                         "current_runtime_state_sha256": current_hash,
                         "historical_state_identity_verified": historical_hash is not None,
                         "initial_state_hash_provenance": PROVENANCE,
                         "evaluation_labels_previously_used": True if prompt["task_id"] in (0, 1) else None})
    structural_audit = {"num_episodes": len(episodes), "num_tasks": len(tasks),
                        "num_shards": len(ranges), "num_forwards": total, "num_action_queries": len(steps),
                        "action_dim": action_dim, "layer_idx": layer_idx,
                        "all_action_queries_complete": True,
                        "forward_row_count_histogram": {str(key): value for key, value in sorted(row_counts.items())},
                        "dense_tensor_contents_verified": False,
                        "historical_runtime_flags_verified": False}
    result = {"schema_version": SCHEMA, "suite": suite, "dataset_id": dataset_id,
              "episodes": episodes, "synthetic": False,
              "initial_state_hash_provenance": PROVENANCE,
              "state_identity_scope": "current_runtime_registry_joined_by_original_task_trial",
              "historical_state_identity_verified": all(row["historical_state_identity_verified"] for row in episodes),
              "source_files": {"prompt_records": {"path": str(prompt_path), "sha256": sha256_file(prompt_path)},
                               "activation_index": {"path": str(index_path), "sha256": sha256_file(index_path)}},
              "runtime_evidence": runtime_evidence, "structural_audit": structural_audit}
    result["manifest_hash"] = fingerprint(result)
    return result


def build_generation_manifest(source_manifest: Mapping[str, Any]) -> dict:
    """Describe the audited readout interpretation without inventing old logs.

Flags are interpretation assumptions supported by observed seven-forward shape
and the current collector. They do not establish historical runtime settings.
"""
    source = dict(source_manifest)
    stored_hash = source.pop("manifest_hash", None)
    if source.get("schema_version") != SCHEMA or fingerprint(source) != stored_hash:
        raise ValueError("Invalid current state registry manifest hash")
    audit = source["structural_audit"]
    if audit.get("all_action_queries_complete") is not True:
        raise ValueError("Generation manifest requires a complete structural audit")
    package = Path(__file__).resolve().parents[2]
    code_files = {str(path.relative_to(package)): sha256_file(path) for path in (
        package / "openvla/activations.py", package / "openvla/eval/model.py",
        package / "openvla/eval/runner.py") if path.is_file()}
    return {"schema_version": "output_head_generation_interpretation_v1",
            "dataset_id": source["dataset_id"], "suite": source["suite"],
            "layer_idx": audit["layer_idx"], "action_dim": audit["action_dim"],
            "batch_size": 1, "padding": "none", "use_cache": True, "index_scope": "complete",
            "evidence": {"source_registry_hash": stored_hash,
                         "source_index_sha256": source["source_files"]["activation_index"]["sha256"],
                         "structural_audit": audit, "current_collector_source_sha256": code_files,
                         "historical_runtime_flags_verified": False,
                         "mapping_assumptions": {"batch_size": 1, "padding": "none", "use_cache": True},
                         "scope": "observed_index_structure_and_current_code_not_historical_runtime_attestation"}}
