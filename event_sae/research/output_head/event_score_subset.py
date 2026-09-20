"""Create an exact-population task subset of a legacy Event-SAE score artifact.

The public Spatial-500 score matrices are already aggregated.  Row filtering can
change their task scope, but it cannot change their episode population.  This
adapter therefore binds a derived artifact to an output-head discovery sample
only when the source ``selected_events`` episode population is exactly equal to
the sample population.  A subset/superset relation is rejected rather than
being mislabeled with the sample's discovery hash.
"""

from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .provenance import fingerprint, read_json, sha256_file
from .readouts import discovery_population_hash


SCHEMA_VERSION = "event_score_exact_task_subset_v1"


def _task_id(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"task_id must be a nonnegative integer, got {value!r}")
    return value


def _task_ids(values: Sequence[int]) -> list[int]:
    result = sorted(_task_id(value) for value in values)
    if not result or len(result) != len(set(result)):
        raise ValueError("task_ids must be nonempty and unique")
    return result


def _episode_key(row: Mapping[str, Any], suite: str) -> tuple[str, int, int]:
    row_suite = row.get("suite", suite)
    task = _task_id(row.get("task_id"))
    trial = row.get("task_episode_idx")
    if isinstance(trial, bool) or not isinstance(trial, int) or trial < 0:
        raise ValueError("episode population requires nonnegative task_episode_idx")
    if not isinstance(row_suite, str) or not row_suite:
        raise ValueError("episode population requires a suite")
    return row_suite, task, trial


def _exact_population(
    selected_events: Any,
    discovery_episodes: Any,
    *,
    task_ids: Sequence[int],
    suite: str,
    expected_episodes_per_task: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(selected_events, list) or not selected_events:
        raise ValueError("Source score artifact requires nonempty selected_events provenance")
    if not isinstance(discovery_episodes, list) or not discovery_episodes:
        raise ValueError("Sample manifest requires nonempty discovery_episodes")
    wanted = set(task_ids)
    if (isinstance(expected_episodes_per_task, bool)
            or not isinstance(expected_episodes_per_task, int)
            or expected_episodes_per_task < 1):
        raise ValueError("expected_episodes_per_task must be a positive integer")
    source_keys = {_episode_key(row, suite) for row in selected_events
                   if _task_id(row.get("task_id")) in wanted}
    all_sample_tasks = {_task_id(row.get("task_id")) for row in discovery_episodes}
    if all_sample_tasks != wanted:
        raise ValueError("Sample discovery task scope differs from the requested task subset")
    sample_rows = [dict(row) for row in discovery_episodes
                   if _task_id(row.get("task_id")) in wanted]
    if {row.get("task_id") for row in sample_rows} != wanted:
        raise ValueError("Sample discovery population does not contain every requested task")
    if any(row.get("initial_state_sha256") is not None for row in sample_rows):
        raise ValueError(
            "Source selected_events lack initial-state hashes; cannot prove exact identity "
            "against a sample containing state hashes"
        )
    sample_keys = {_episode_key(row, suite) for row in sample_rows}
    if len(sample_keys) != len(sample_rows):
        raise ValueError("Sample discovery population contains duplicate task/trial identities")
    if source_keys != sample_keys:
        missing = sorted(source_keys - sample_keys)
        extra = sorted(sample_keys - source_keys)
        raise ValueError(
            "Event-score/source and output-head discovery populations are not exactly equal: "
            f"source_only={len(missing)} sample_only={len(extra)}; "
            "aggregated matrices cannot be rebound to a partial population"
        )
    per_task_counts = {
        task: sum(key[1] == task for key in source_keys)
        for task in sorted(wanted)
    }
    if any(count != expected_episodes_per_task for count in per_task_counts.values()):
        raise ValueError(
            "Exact population has the wrong episode count: "
            f"expected {expected_episodes_per_task} per task, got {per_task_counts}"
        )
    population = [
        {"suite": key[0], "task_id": key[1], "task_episode_idx": key[2],
         "initial_state_sha256": None}
        for key in sorted(source_keys)
    ]
    evidence = {
        "relation": "exact_equal",
        "identity_fields": ["suite", "task_id", "task_episode_idx"],
        "initial_state_identity": "unavailable_in_source_and_null_in_head_sample",
        "num_episodes": len(population),
        "episodes_per_task": per_task_counts,
        "expected_episodes_per_task": expected_episodes_per_task,
        "population_hash": discovery_population_hash(population),
    }
    return population, evidence


def build_exact_task_subset(
    source_path: str | Path,
    sample_manifest_path: str | Path,
    head_scores_path: str | Path,
    output_path: str | Path,
    *,
    task_ids: Sequence[int],
    suite: str,
    expected_source_sha256: str,
    expected_sae_sha256: str,
    expected_episodes_per_task: int,
) -> dict[str, Any]:
    """Filter task rows and bind them only to an exactly matching sample."""
    import torch

    source_path = Path(source_path).expanduser().resolve()
    sample_path = Path(sample_manifest_path).expanduser().resolve()
    head_path = Path(head_scores_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    tasks = _task_ids(task_ids)
    if not isinstance(suite, str) or not suite:
        raise ValueError("suite must be a nonempty string")
    if output_path.exists():
        raise FileExistsError(output_path)
    source_hash = sha256_file(source_path)
    if source_hash != expected_source_sha256:
        raise ValueError(f"Source SHA-256 mismatch: {source_hash} != {expected_source_sha256}")
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Source score artifact must be a mapping")
    payload = dict(payload)
    rows = payload.get("row_keys")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Source score artifact requires nonempty row_keys")
    row_tasks = [_task_id(row.get("task_id")) for row in rows]
    keep = [index for index, task in enumerate(row_tasks) if task in set(tasks)]
    if not keep or sorted(set(row_tasks[index] for index in keep)) != tasks:
        raise ValueError("Source score rows do not cover every requested task")

    sample = read_json(sample_path)
    head_scores = read_json(head_path)
    scope = dict(head_scores.get("scope", {}))
    required_scope = ("task_ids", "dictionary_size", "discovery_manifest_hash",
                      "sample_manifest_hash", "sae_sha256")
    missing_scope = [name for name in required_scope if scope.get(name) is None]
    if missing_scope:
        raise ValueError(f"Head score scope missing required fields: {', '.join(missing_scope)}")
    if sorted(scope["task_ids"]) != tasks:
        raise ValueError("Head score task scope differs from requested task subset")
    if scope["sae_sha256"] != expected_sae_sha256:
        raise ValueError("Head score SAE SHA-256 differs from the expected Spatial-500 SAE")
    if sample.get("sample_hash") != scope["sample_manifest_hash"]:
        raise ValueError("Head scores are not bound to the supplied sample manifest")
    recomputed_sample_population = discovery_population_hash(sample.get("discovery_episodes", []))
    if sample.get("discovery_manifest_hash") != recomputed_sample_population:
        raise ValueError("Sample discovery_manifest_hash does not match its episode population")
    _, population_evidence = _exact_population(
        payload.get("selected_events"), sample.get("discovery_episodes"),
        task_ids=tasks, suite=suite,
        expected_episodes_per_task=expected_episodes_per_task,
    )
    if population_evidence["population_hash"] != scope["discovery_manifest_hash"]:
        raise ValueError("Exact source population hash differs from the head-score discovery scope")

    result = copy.deepcopy(payload)
    result["row_keys"] = [copy.deepcopy(rows[index]) for index in keep]
    dimension = None
    for name in ("matrix_raw", "matrix_window_mean", "matrix_task_mean", "matrix"):
        if name not in payload:
            continue
        matrix = torch.as_tensor(payload[name], device="cpu")
        if matrix.ndim != 2 or matrix.shape[0] != len(rows) or not torch.isfinite(matrix).all():
            raise ValueError(f"Invalid {name} shape/values")
        if dimension is not None and dimension != int(matrix.shape[1]):
            raise ValueError("Source matrices disagree on dictionary size")
        dimension = int(matrix.shape[1])
        result[name] = matrix[keep].clone()
    if "matrix_raw" not in result or dimension is None:
        raise ValueError("Source score artifact requires matrix_raw")
    if dimension != scope["dictionary_size"]:
        raise ValueError("Source dictionary size differs from head scores")
    if isinstance(payload.get("row_results"), list):
        if len(payload["row_results"]) != len(rows):
            raise ValueError("row_results does not align with row_keys")
        result["row_results"] = [copy.deepcopy(payload["row_results"][index]) for index in keep]
    result["selected_events"] = [copy.deepcopy(row) for row in payload["selected_events"]
                                 if _task_id(row.get("task_id")) in set(tasks)]
    source_counts = copy.deepcopy(payload.get("selection_counts", {}))
    task_counts = source_counts.get("task_timestep_counts", {}) if isinstance(source_counts, Mapping) else {}
    filtered_counts = {str(task): task_counts.get(str(task), task_counts.get(task)) for task in tasks}
    if any(value is None for value in filtered_counts.values()):
        filtered_counts = {}
    result["selection_counts"] = {"task_timestep_counts": filtered_counts}
    # The episode population is shared, but the timestep sampling is not: the
    # legacy Event score used every retained timestep whereas the head score
    # used sampled readouts.  Do not copy ``sample_manifest_hash`` into the
    # comparison scope, because that would falsely claim identical timesteps.
    result["scope"] = {
        "task_ids": tasks,
        "dictionary_size": dimension,
        "sae_sha256": scope["sae_sha256"],
        "discovery_manifest_hash": scope["discovery_manifest_hash"],
        "split_manifest_hash": scope.get("split_manifest_hash"),
        "population_binding": population_evidence,
        "timestep_scope": "legacy_event_all_retained_timesteps",
    }
    result["synthetic"] = False
    result["derivation"] = {
        "schema_version": SCHEMA_VERSION,
        "operation": "row_filter_only_no_episode_reaggregation",
        "source_artifact_sha256": source_hash,
        "source_sae_binding": {
            "status": "external_pinned_artifact_attestation",
            "expected_sae_sha256": expected_sae_sha256,
            "basis": "source_artifact_sha256_and_published_artifact_provenance",
        },
        "source_row_count": len(rows),
        "derived_row_count": len(keep),
        "source_selected_event_count": len(payload["selected_events"]),
        "derived_selected_event_count": len(result["selected_events"]),
        "source_selection_counts_hash": fingerprint(source_counts),
        "sample_manifest_sha256": sha256_file(sample_path),
        "head_sample_manifest_hash": scope["sample_manifest_hash"],
        "head_scores_sha256": sha256_file(head_path),
        "task_ids": tasks,
        "suite": suite,
        "population_binding": population_evidence,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    os.close(descriptor)
    try:
        torch.save(result, temporary)
        os.link(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {
        "status": "created",
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "scope": result["scope"],
        "derivation": result["derivation"],
    }
