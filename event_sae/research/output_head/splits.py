"""Freeze episode/initial-state splits before looking at evaluation labels.

This module intentionally depends only on the standard library. A declaration
about prior label use is provenance, not something code can infer from files.
Unknown history remains unknown and cannot qualify as confirmatory evidence.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _source(value: Any) -> tuple[list[dict], dict]:
    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as stream:
                value = [json.loads(line) for line in stream if line.strip()]
        else:
            with path.open(encoding="utf-8") as stream:
                value = json.load(stream)
    if isinstance(value, dict):
        rows = value.get("episodes", value.get("records"))
        metadata = {key: val for key, val in value.items() if key not in {"episodes", "records"}}
    else:
        rows, metadata = value, {}
    if not isinstance(rows, list) or not rows:
        raise ValueError("source_episode_manifest must contain nonempty episodes")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("episode records must be mappings")
    return [dict(row) for row in rows], metadata


def episode_key(row: Mapping[str, Any]) -> tuple[str, int, int]:
    """Task-qualified original trial identity; run-local episode IDs do not join."""
    suite = row.get("suite")
    if not isinstance(suite, str) or not suite:
        raise ValueError("episode suite is required")
    return suite, _int(row.get("task_id"), "task_id"), _int(
        row.get("task_episode_idx"), "task_episode_idx")


def _state_key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    suite, task, _ = episode_key(row)
    state_hash = row.get("initial_state_sha256")
    if not isinstance(state_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", state_hash):
        raise ValueError("initial_state_sha256 must be a SHA-256 digest")
    # The same vector under a different task is a different task-conditioned state.
    return suite, task, state_hash.lower()


def _label_history(row: Mapping[str, Any], default: Any = None) -> bool | None:
    value = row.get("evaluation_labels_previously_used", default)
    if value is not None and not isinstance(value, bool):
        raise ValueError("evaluation_labels_previously_used must be true, false, or null")
    return value


def build_split_manifest(source_episode_manifest: Any, split_spec: Mapping[str, Any]) -> dict:
    """Make deterministic exact-count task splits without using success labels.

Input is a list, JSON/JSONL path, or ``{episodes: [...], ...provenance}``.
Required record fields are suite/task_id/task_episode_idx/initial_state_sha256.
Hash provenance must be explicit on the source manifest or each record.
``evaluation_labels_previously_used=None`` is permitted when freezing, but
``validate_split_manifest(..., require_confirmatory=True)`` will reject it.
Whether freezing preceded the pilot must also be explicitly declared; creating
a manifest now cannot establish that historical fact.
"""
    records, metadata = _source(source_episode_manifest)
    if not isinstance(split_spec, Mapping):
        raise ValueError("split_spec must be a mapping")
    frozen_before_pilot = split_spec.get("frozen_before_pilot")
    if frozen_before_pilot is not None and not isinstance(frozen_before_pilot, bool):
        raise ValueError("frozen_before_pilot must be true, false, or null")
    names = ("discovery", "validation", "evaluation")
    counts = {name: _int(split_spec.get(f"{name}_per_task"), f"{name}_per_task") for name in names}
    if counts["discovery"] == 0 or counts["evaluation"] == 0:
        raise ValueError("discovery and evaluation must each have at least one episode")
    if split_spec.get("require_initial_state_disjoint", True) is not True:
        raise ValueError("initial-state disjointness cannot be disabled")
    seed = _int(split_spec.get("split_seed", 2026), "split_seed")
    default_history = _label_history(split_spec)
    default_provenance = metadata.get("initial_state_hash_provenance")
    default_verified = metadata.get("initial_state_hash_verified") is True
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    seen_episodes, seen_states = set(), set()
    for row in records:
        if "suite" not in row and metadata.get("suite"):
            row["suite"] = metadata["suite"]
        key, state = episode_key(row), _state_key(row)
        if key in seen_episodes:
            raise ValueError(f"duplicate task/original trial identity: {key}")
        if state in seen_states:
            raise ValueError(f"duplicate initial state within task: {state[:2]}")
        provenance = row.get("initial_state_hash_provenance", default_provenance)
        if not (row.get("initial_state_hash_verified") is True or default_verified or provenance):
            raise ValueError("initial-state hash provenance is unknown")
        seen_episodes.add(key)
        seen_states.add(state)
        row["initial_state_sha256"] = state[2]
        row["initial_state_hash_provenance"] = provenance or "source_declares_hash_verified"
        row["evaluation_labels_previously_used"] = _label_history(row, default_history)
        grouped[key[:2]].append(row)
    assignments: dict[str, list[dict]] = {name: [] for name in names}
    for task, rows in sorted(grouped.items()):
        if len(rows) != sum(counts.values()):
            raise ValueError(f"task {task} has {len(rows)} states; split counts require {sum(counts.values())}")
        rows.sort(key=episode_key)
        task_seed = int(_digest([seed, list(task)]), 16)
        random.Random(task_seed).shuffle(rows)
        start = 0
        for name in names:
            assignments[name].extend(rows[start:start + counts[name]])
            start += counts[name]
    evaluation_history = [row["evaluation_labels_previously_used"] for row in assignments["evaluation"]]
    history = True if True in evaluation_history else (None if None in evaluation_history else False)
    manifest = {
        "schema_version": "output_head_split_v1", "frozen": True,
        "frozen_before_pilot": frozen_before_pilot,
        "split_seed": seed, "algorithm": "python_random_per_task_sha256_seed_v1",
        "counts_per_task": counts, "source_manifest_hash": _digest({"records": sorted(records, key=episode_key), "metadata": metadata}),
        "splits": assignments, "evaluation_labels_previously_used": history,
        "initial_state_identity": "suite/task_id/initial_state_sha256",
        "synthetic": metadata.get("synthetic", False),
        "generalization_scope": "ranking_selection_with_fixed_pretrained_sae",
    }
    manifest["manifest_hash"] = _digest(manifest)
    validate_split_manifest(manifest)
    return manifest


def build_followup_split(source_episode_manifest: Any, split_spec: Mapping[str, Any]) -> dict:
    """Freeze a new, explicitly non-confirmatory study after an earlier pilot.

    Current LIBERO state hashes protect the upcoming runs. They do not establish
    the state bytes used by historical activation collection. Preserve both
    identities so this limitation cannot disappear when scores are exported.
    """
    records, metadata = _source(source_episode_manifest)
    if split_spec.get("frozen_before_pilot") is True:
        raise ValueError("A followup split cannot be backdated before the old pilot")
    for row in records:
        current = row.get("current_runtime_state_sha256")
        if current != row.get("initial_state_sha256"):
            raise ValueError("Followup requires matching current runtime registry state hashes")
        historical = row.get("historical_initial_state_sha256")
        if historical is not None and historical != current:
            raise ValueError("Historical source state differs from current registered state")
        if not row.get("initial_state_hash_provenance", metadata.get("initial_state_hash_provenance")):
            raise ValueError("Followup requires explicit current-state registry provenance")
    verified = metadata.get("historical_state_identity_verified") is True
    if verified and any(row.get("historical_initial_state_sha256") is None for row in records):
        raise ValueError("Historical state verification requires original hashes for every episode")
    result = build_split_manifest({**metadata, "episodes": records},
                                  {**split_spec, "frozen_before_pilot": False})
    result.update(schema_version="output_head_followup_split_v1", mode="followup",
                  frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                  frozen_before_followup_evaluation=True,
                  historical_state_identity_verified=verified,
                  historical_source_state_status="verified" if verified else "unavailable",
                  initial_state_identity="suite/task_id/current_runtime_state_sha256",
                  generalization_scope="post_pilot_followup_fixed_pretrained_sae_not_confirmatory")
    result["manifest_hash"] = _digest({key: value for key, value in result.items() if key != "manifest_hash"})
    validate_split_manifest(result, require_followup=True)
    return result


def validate_split_manifest(manifest: Mapping[str, Any], *, require_confirmatory: bool = False,
                            require_followup: bool = False) -> dict:
    """Check immutable membership, duplicate states, exact counts, and history."""
    schemas = {"output_head_split_v1", "output_head_followup_split_v1"}
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") not in schemas:
        raise ValueError("unsupported split manifest")
    is_followup = manifest.get("schema_version") == "output_head_followup_split_v1"
    if require_confirmatory and is_followup:
        raise ValueError("Followup splits cannot qualify as confirmatory")
    if require_followup and not is_followup:
        raise ValueError("Followup execution requires an explicitly frozen followup split")
    expected = manifest.get("manifest_hash")
    if not expected or _digest({k: v for k, v in manifest.items() if k != "manifest_hash"}) != expected:
        raise ValueError("split manifest hash mismatch")
    if manifest.get("frozen") is not True:
        raise ValueError("split is not frozen")
    if is_followup:
        if (manifest.get("mode") != "followup" or manifest.get("frozen_before_pilot") is not False
                or manifest.get("frozen_before_followup_evaluation") is not True):
            raise ValueError("Followup must be frozen now without backdating the old pilot")
        try:
            frozen_at = datetime.fromisoformat(manifest["frozen_at_utc"])
            if frozen_at.tzinfo is None or frozen_at > datetime.now(timezone.utc):
                raise ValueError("invalid freeze time")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Followup requires a valid UTC freeze timestamp") from exc
        if not isinstance(manifest.get("historical_state_identity_verified"), bool):
            raise ValueError("Followup must declare historical state verification")
        expected_status = "verified" if manifest["historical_state_identity_verified"] else "unavailable"
        if (manifest.get("historical_source_state_status") != expected_status
                or manifest.get("initial_state_identity") != "suite/task_id/current_runtime_state_sha256"
                or manifest.get("generalization_scope") != "post_pilot_followup_fixed_pretrained_sae_not_confirmatory"):
            raise ValueError("Followup state identity or study scope is inconsistent")
    seen_episodes, seen_states = set(), set()
    task_sets = []
    splits = manifest.get("splits", {})
    for name in ("discovery", "validation", "evaluation"):
        rows = splits.get(name)
        if not isinstance(rows, list):
            raise ValueError(f"missing {name} split")
        counts: dict[tuple, int] = defaultdict(int)
        for row in rows:
            key, state = episode_key(row), _state_key(row)
            if is_followup:
                if row.get("current_runtime_state_sha256") != state[2]:
                    raise ValueError("Followup current runtime state hash mismatch")
                historical = row.get("historical_initial_state_sha256")
                if historical is not None and historical != state[2]:
                    raise ValueError("Followup historical state contradicts the registry")
                if manifest["historical_state_identity_verified"] and historical is None:
                    raise ValueError("Followup historical state verification lacks source evidence")
                if not row.get("initial_state_hash_provenance"):
                    raise ValueError("Followup current-state provenance is missing")
            if key in seen_episodes or state in seen_states:
                raise ValueError("episode or initial state overlaps splits")
            seen_episodes.add(key)
            seen_states.add(state)
            counts[key[:2]] += 1
            _label_history(row)
        expected_count = manifest["counts_per_task"][name]
        if any(value != expected_count for value in counts.values()):
            raise ValueError(f"{name} per-task counts do not match")
        if expected_count:
            task_sets.append(set(counts))
    if not task_sets or any(tasks != task_sets[0] for tasks in task_sets):
        raise ValueError("task coverage differs across splits")
    history = [_label_history(row) for row in splits["evaluation"]]
    aggregated = True if True in history else (None if None in history else False)
    if manifest.get("evaluation_labels_previously_used") is not aggregated:
        raise ValueError("evaluation label history is inconsistent")
    if require_confirmatory:
        if manifest.get("frozen_before_pilot") is not True:
            raise ValueError("evaluation split was not frozen before pilot")
        if aggregated is not False:
            raise ValueError("confirmatory evaluation label history is unknown or previously used")
    return {"valid": True, "manifest_hash": expected, "num_episodes": len(seen_episodes),
            "evaluation_labels_previously_used": aggregated}


def validate_selection_split(manifest: Mapping[str, Any], episode_rows: Sequence[Mapping[str, Any]],
                             *, stage: str = "pilot", labels_used: bool | None = None,
                             mode: str | None = None) -> dict:
    """Enforce discovery-only scoring and sealed evaluation states in pilot."""
    validate_split_manifest(manifest, require_confirmatory=mode != "followup",
                            require_followup=mode == "followup")
    allowed = {"score": ("discovery",), "pilot": ("discovery", "validation"),
               "validation": ("validation",), "evaluation": ("evaluation",)}
    if stage not in allowed:
        raise ValueError(f"unknown stage: {stage}")
    membership = {episode_key(row): row for name in allowed[stage] for row in manifest["splits"][name]}
    if not episode_rows:
        raise ValueError("empty selection")
    for row in episode_rows:
        key = episode_key(row)
        if key not in membership or _state_key(row) != _state_key(membership[key]):
            raise ValueError(f"{stage} selection uses a forbidden split or mismatched state: {key}")
    if stage == "evaluation" and labels_used is True:
        raise ValueError("evaluation labels cannot be used for method selection")
    return {"valid": True, "stage": stage, "num_episodes": len(episode_rows),
            "split_manifest_hash": manifest["manifest_hash"]}
