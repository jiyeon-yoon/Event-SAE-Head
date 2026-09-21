"""Recompute inherited Event scores on the exact Head-KL discovery population.

The event vocabulary/clusters remain frozen inherited artifacts. Filtering their
episode inputs does not make clustering or the pretrained SAE held-out. Current
LIBERO state hashes establish the follow-up registry, not historical collection
states. Neither limitation is erased when binding the new score artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .config import load_config, validate_config
from .provenance import atomic_write_json, fingerprint, read_json, read_jsonl, sha256_file
from .readouts import _identity, discovery_population_hash
from .splits import episode_key, validate_selection_split, validate_split_manifest


SCHEMA = "output_head_discovery_event_scores_v1"


def _path(value: Any, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f"Missing {name}")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _episode_map(rows: list[dict], name: str) -> dict[int, dict]:
    result = {}
    identities = set()
    for row in rows:
        number = row.get("episode_num")
        if type(number) is not int or number < 0 or number in result:
            raise ValueError(f"Invalid/duplicate episode_num in {name}")
        identity = (row.get("task_id"), row.get("task_episode_idx"))
        if any(type(value) is not int or value < 0 for value in identity) or identity in identities:
            raise ValueError(f"Invalid/duplicate task/trial in {name}")
        if type(row.get("source_run_idx")) is not int or row["source_run_idx"] < 0:
            raise ValueError(f"Missing source_run_idx in {name}")
        result[number] = row
        identities.add(identity)
    return result


def _same_origin(first: dict, second: dict, name: str) -> None:
    for field in ("episode_num", "task_id", "task_episode_idx", "source_run_idx"):
        if first.get(field) != second.get(field):
            raise ValueError(f"{name}: mismatched {field}")


def _index_evidence(index: Path, sample: dict, prompts: dict[int, dict],
                    allowed: set[int]) -> dict:
    """Stream the large index, verify readouts, and count every discovery step."""
    rows = sample["readouts"]
    wanted = {}
    for row in rows:
        key = (row.get("source_run_idx"), row.get("global_forward_idx"))
        if key in wanted:
            raise ValueError("Duplicate readout forward identity")
        wanted[key] = row
    seen = set()
    steps = set()
    canonical = hashlib.sha256(b"[")
    first = True
    with index.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not first:
                canonical.update(b",")
            canonical.update(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                        allow_nan=False).encode())
            first = False
            number = row.get("episode_num")
            if number in allowed:
                _same_origin(row, prompts[number], "activation index/prompt")
                if row.get("layer_idx") != sample["generation_spec"]["layer_idx"]:
                    raise ValueError("Discovery index contains another layer")
                step = row.get("step_in_episode")
                if type(step) is not int or step < 0:
                    raise ValueError("Invalid discovery timestep")
                steps.add((number, step))
            key = (row.get("source_run_idx"), row.get("global_forward_idx"))
            if key not in wanted:
                continue
            if key in seen:
                raise ValueError("Duplicate selected forward in activation index")
            seen.add(key)
            selected = wanted[key]
            _same_origin(row, selected, "activation index/readout")
            for field in ("row_start", "row_end", "layer_idx", "step_in_episode", "shard_path"):
                if row.get(field) != selected.get(field):
                    raise ValueError(f"activation index/readout: mismatched {field}")
    canonical.update(b"]")
    if canonical.hexdigest() != sample.get("source_index_hash"):
        raise ValueError("Sample source_index_hash differs from current activation index")
    if seen != set(wanted):
        raise ValueError("Sample readouts are missing from activation index")
    if {number for number, _ in steps} != allowed:
        raise ValueError("Discovery episodes are missing activation timesteps")
    counts = Counter(prompts[number]["task_id"] for number, _ in steps)
    return {"num_readouts_verified": len(seen), "num_discovery_timesteps": len(steps),
            "task_timestep_counts": {str(key): value for key, value in sorted(counts.items())}}


def _source_signature(paths: dict[str, Path], topk: dict) -> dict:
    files = {name: sha256_file(path) for name, path in paths.items()}
    shard_hashes = {}
    for shard in topk.get("shards", []):
        relative = shard.get("path")
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Top-K shard paths must be relative and confined to topk/")
        if relative in shard_hashes:
            raise ValueError("Duplicate Top-K shard")
        path = paths["topk_manifest"].parent / relative
        shard_hashes[relative] = sha256_file(path)
    if not shard_hashes or topk.get("num_shards") != len(shard_hashes):
        raise ValueError("Top-K shard count mismatch")
    package = Path(__file__).resolve().parents[2]
    code_paths = [package / "scoring/score_matrix.py", package / "events/io.py"]
    code_paths.extend(Path(__file__).with_name(name) for name in (
        "discovery_scores.py", "splits.py", "readouts.py", "state_registry.py", "config.py", "provenance.py"))
    return {"files": files, "topk_shards": shard_hashes,
            "code": {str(path.relative_to(package)): sha256_file(path) for path in code_paths}}


def _binding(config: dict, event_root: Path) -> dict:
    dense = _path(config["inputs"]["dense_dir"], "inputs.dense_dir")
    merged = dense.parents[1]
    root = _path(config["output"]["root_dir"], "output.root_dir")
    weights = _path(config["inputs"]["sae_checkpoint"], "inputs.sae_checkpoint")
    if weights.is_dir():
        weights = _path(weights / "ae.pt", "SAE weights")
    paths = {
        "topk_manifest": event_root / "topk/manifest.json",
        "event_features": event_root / "events/event_features.jsonl",
        "cluster_assignments": event_root / "clusters/cluster_assignments.jsonl",
        "clusters": event_root / "clusters/clusters.jsonl",
        "prompts": merged / "prompt_records.jsonl",
        "activation_index": dense / "activation_index.jsonl",
        "sample": root / "sample_manifest.json",
        "head_scores": root / "scores/scores.json",
        "source_registry": _path(config["inputs"]["source_episode_manifest"], "source_episode_manifest"),
        "split": _path(config["sampling"]["split_manifest"], "sampling.split_manifest"),
        "sae": weights,
    }
    paths = {name: _path(path, name) for name, path in paths.items()}
    sample, head, split = (read_json(paths[name]) for name in ("sample", "head_scores", "split"))
    if config["sampling"]["mode"] != "followup":
        raise ValueError("Discovery Event binding requires sampling.mode=followup")
    validate_split_manifest(split, require_followup=True)
    discovery = sample.get("discovery_episodes")
    if not isinstance(discovery, list) or not discovery:
        raise ValueError("Sample discovery population is empty")
    validate_selection_split(split, discovery, stage="score", mode="followup")
    tasks = sorted(config["sampling"]["task_ids"])
    expected = {episode_key(row): row for row in split["splits"]["discovery"] if row["task_id"] in tasks}
    actual = {episode_key(row): row for row in discovery}
    if len(actual) != len(discovery) or set(actual) != set(expected):
        raise ValueError("Head sample must select the complete discovery split for every requested task")
    if sample.get("synthetic") or head.get("synthetic") or split.get("synthetic"):
        raise ValueError("Synthetic artifacts cannot bind a real Event comparator")
    mapping_hash = fingerprint({"index": sample["source_index_hash"], "generation": sample["generation_spec"],
                                "mapping_version": sample["mapping_version"]})
    sample_hash = fingerprint({"mapping_hash": mapping_hash, "sample": sample["sample_spec"],
                               "readouts": sample["readouts"]})
    if sample.get("mapping_hash") != mapping_hash or sample.get("sample_hash") != sample_hash:
        raise ValueError("Sample manifest hash mismatch")
    population_hash = discovery_population_hash(discovery)
    scope = head.get("scope", {})
    if (sample.get("discovery_manifest_hash") != population_hash
            or scope.get("discovery_manifest_hash") != population_hash
            or scope.get("sample_manifest_hash") != sample_hash):
        raise ValueError("Head scores do not match the discovery sample")
    if (sample.get("split_manifest_hash") != split["manifest_hash"]
            or scope.get("split_manifest_hash") != split["manifest_hash"]):
        raise ValueError("Sample/head split binding mismatch")
    if sorted(scope.get("task_ids", [])) != tasks:
        raise ValueError("Head score task scope differs from config")
    topk = read_json(paths["topk_manifest"])
    source_signature = _source_signature(paths, topk)
    files = source_signature["files"]
    if topk.get("format") != "token_topk_sparse_v1":
        raise ValueError("Unsupported Top-K format")
    capture_target = topk.get("capture_target")
    if capture_target not in (None, "post_mlp_residual"):
        raise ValueError("Unsupported Top-K capture target")
    capture_binding = "explicit_topk_manifest_capture_target"
    if capture_target is None:
        # The pinned public Spatial-500 export predates capture_target metadata.
        # Its original dense path names the hook; exact SAE/index hashes below
        # must also match before this legacy provenance is accepted.
        source_dense = Path(topk.get("dense_dir") or "")
        suffix = ("sae_activations", "post_mlp_residual")
        if source_dense.parts[-2:] != suffix or dense.parts[-2:] != suffix:
            raise ValueError("Legacy null Top-K capture target requires post_mlp_residual dense-path evidence")
        capture_binding = "legacy_source_dense_path_plus_exact_index_and_sae_hashes"
    if topk.get("layer") != config["scope"]["layer_idx"]:
        raise ValueError("Top-K layer differs from Head-KL")
    if topk.get("dict_size") != scope.get("dictionary_size"):
        raise ValueError("Top-K dictionary size differs from Head-KL")
    if topk.get("sae_sha256") != files["sae"] or scope.get("sae_sha256") != files["sae"]:
        raise ValueError("Top-K/head SAE SHA-256 differs from actual weights")
    if topk.get("activation_index_sha256") != files["activation_index"]:
        raise ValueError("Top-K activation index SHA-256 differs from the merged run")
    prompts = _episode_map(read_jsonl(paths["prompts"]), "prompts")
    registry = read_json(paths["source_registry"])
    registry_hash = registry.get("manifest_hash")
    if (registry.get("schema_version") != "output_head_current_state_registry_v1"
            or not registry_hash
            or fingerprint({key: value for key, value in registry.items() if key != "manifest_hash"}) != registry_hash):
        raise ValueError("Source registry manifest hash/schema mismatch")
    for source_name, path_name in (("prompt_records", "prompts"), ("activation_index", "activation_index")):
        if registry.get("source_files", {}).get(source_name, {}).get("sha256") != files[path_name]:
            raise ValueError(f"Source registry {source_name} hash differs from current input")
    normalized_records = []
    for original in registry.get("episodes", []):
        row = dict(original)
        row.setdefault("suite", registry.get("suite"))
        row["initial_state_sha256"] = row["initial_state_sha256"].lower()
        row["initial_state_hash_provenance"] = row.get(
            "initial_state_hash_provenance", registry.get("initial_state_hash_provenance")) or "source_declares_hash_verified"
        row["evaluation_labels_previously_used"] = row.get(
            "evaluation_labels_previously_used", config["splits"]["evaluation_labels_previously_used"])
        normalized_records.append(row)
    registry_metadata = {key: value for key, value in registry.items() if key not in ("episodes", "records")}
    if split.get("source_manifest_hash") != fingerprint({
            "records": sorted(normalized_records, key=episode_key), "metadata": registry_metadata}):
        raise ValueError("Frozen split source manifest differs from the current source registry")
    generation = sample["generation_spec"]
    if generation.get("schema_version") == "output_head_generation_interpretation_v1":
        evidence = generation.get("evidence", {})
        if (evidence.get("source_registry_hash") != registry_hash
                or evidence.get("source_index_sha256") != files["activation_index"]
                or evidence.get("historical_runtime_flags_verified") is not False):
            raise ValueError("Generation interpretation is not bound to this current-state registry")
    registered = _episode_map(registry.get("episodes", []), "source registry")
    registry_keys = {episode_key(row): row for row in registered.values()}
    allowed = set()
    for key, selected in actual.items():
        original = registry_keys.get(key)
        if original is None:
            raise ValueError("Discovery episode is absent from source registry")
        number = original["episode_num"]
        if number not in prompts:
            raise ValueError("Registry episode is absent from prompts")
        _same_origin(original, prompts[number], "registry/prompt")
        for row in (original, expected[key], selected):
            if row.get("initial_state_sha256") != expected[key]["initial_state_sha256"]:
                raise ValueError("Source registry/discovery split state identity mismatch")
        if original.get("current_runtime_state_sha256") != original.get("initial_state_sha256"):
            raise ValueError("Registry must distinguish current-runtime state identity")
        for field in ("historical_initial_state_sha256", "initial_state_hash_provenance", "source_run_idx", "episode_num"):
            if original.get(field) != expected[key].get(field):
                raise ValueError(f"Source registry/split {field} mismatch")
        if _identity(prompts[number], sample["generation_spec"]) != selected.get("source_run_id"):
            raise ValueError("Sample source-run namespace differs from prompt origin")
        allowed.add(number)
    selected_numbers = set()
    for row in sample["readouts"]:
        number = row.get("episode_num")
        if number not in allowed:
            raise ValueError("Readout is outside exact discovery population")
        _same_origin(row, prompts[number], "readout/prompt")
        selected_numbers.add(number)
    if selected_numbers != allowed:
        raise ValueError("Some discovery episodes have no readouts")
    for row in read_jsonl(paths["event_features"]):
        number = row.get("episode_num")
        if number in allowed:
            for field in ("task_id", "task_episode_idx"):
                if row.get(field) != prompts[number].get(field):
                    raise ValueError(f"Event metadata/prompt {field} mismatch")
            if "source_run_idx" in row and row["source_run_idx"] != prompts[number]["source_run_idx"]:
                raise ValueError("Event metadata source_run_idx mismatch")
    index_evidence = _index_evidence(paths["activation_index"], sample, prompts, allowed)
    new_scope = {key: scope[key] for key in ("task_ids", "dictionary_size", "sae_sha256",
                                            "discovery_manifest_hash", "split_manifest_hash")}
    new_scope.update(timestep_scope="all_retained_discovery_episode_timesteps",
                     cluster_scope="frozen_inherited_full500_clusters_not_discovery_fitted",
                     initial_state_scope="current_runtime_registry_historical_collection_state_unverified",
                     historical_runtime_flags_verified=False,
                     capture_target="post_mlp_residual", capture_target_binding=capture_binding)
    signature = {"schema_version": SCHEMA, "sources": source_signature,
                 "allowed_episode_nums": sorted(allowed), "scope": new_scope,
                 "window_size": 5, "step_mapping": "inference_step", "top_n": 20}
    return {"paths": paths, "signature": signature, "signature_hash": fingerprint(signature),
            "scope": new_scope, "allowed": allowed, "index_evidence": index_evidence,
            "episodes_per_task": dict(Counter(str(prompts[num]["task_id"]) for num in allowed))}


def _verify_scored(payload: dict, context: dict) -> None:
    import torch

    allowed, scope = context["allowed"], context["scope"]
    if payload.get("source", {}).get("allowed_episode_nums") != sorted(allowed):
        raise ValueError("Scorer did not apply the exact discovery filter")
    counts = payload.get("selection_counts", {})
    for field in ("discovery_episode_count", "discovery_episode_count_with_timesteps"):
        if counts.get(field) != len(allowed):
            raise ValueError(f"Incomplete scored discovery population: {field}")
    normalized = {str(key): value for key, value in counts.get("task_timestep_counts", {}).items()}
    if normalized != context["index_evidence"]["task_timestep_counts"]:
        raise ValueError("Scorer timestep coverage differs from discovery activation index")
    normalized = {str(key): value for key, value in counts.get("discovery_task_episode_counts", {}).items()}
    if normalized != context["episodes_per_task"]:
        raise ValueError("Scorer per-task episode coverage differs from discovery split")
    if counts.get("skipped_missing_window_vectors") != 0:
        raise ValueError("Scorer skipped missing discovery activation windows")
    rows, events = payload.get("row_keys", []), payload.get("selected_events", [])
    if not rows or not events or {row.get("task_id") for row in rows} != set(scope["task_ids"]):
        raise ValueError("Scorer does not cover every discovery task")
    if any(row.get("episode_num") not in allowed for row in events):
        raise ValueError("Scored events include excluded episodes")
    if counts.get("selected_events_after_activation_filter") != len(events):
        raise ValueError("Scored event count mismatch")
    for key in ("matrix_raw", "matrix_window_mean", "matrix_task_mean"):
        tensor = torch.as_tensor(payload.get(key), device="cpu")
        if tuple(tensor.shape) != (len(rows), scope["dictionary_size"]) or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid scored {key}")


def build_discovery_event_scores(config_path: str | Path, event_root: str | Path,
                                  *, output_path: str | Path | None = None) -> dict:
    """Write a verified comparator and bind config; exact repeats reuse it."""
    import yaml

    config_path = _path(config_path, "config")
    original_config_hash = sha256_file(config_path)
    config = load_config(config_path)
    default_parent = _path(config["inputs"]["source_episode_manifest"], "source registry").parent
    output = Path(output_path or default_parent / "discovery_event_scores.pt").expanduser().resolve()
    current = config["inputs"]["event_scores_path"]
    if current and Path(current).expanduser().resolve() != output:
        raise ValueError("Refusing to replace a different inputs.event_scores_path")
    updated = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    updated.setdefault("inputs", {})["event_scores_path"] = str(output)
    validate_config(updated)
    context = _binding(config, _path(event_root, "event-root"))
    receipt = output.with_suffix(output.suffix + ".binding.json")
    import torch

    if output.exists() or receipt.exists():
        if not output.is_file() or not receipt.is_file():
            raise ValueError("Incomplete existing Event score/receipt pair; use a new output path")
        proof = read_json(receipt)
        if (proof.get("signature_hash") != context["signature_hash"]
                or proof.get("output_sha256") != sha256_file(output)):
            raise ValueError("Existing Event score binding is stale or modified")
        payload = torch.load(output, map_location="cpu", weights_only=False)
        _verify_scored(payload, context)
        if payload.get("scope") != context["scope"] or payload.get("derivation", {}).get("signature_hash") != context["signature_hash"]:
            raise ValueError("Existing Event score content binding mismatch")
        status = "reused"
    else:
        from event_sae.scoring.score_matrix import score_cluster_features

        output.parent.mkdir(parents=True, exist_ok=True)
        paths = context["paths"]
        with tempfile.TemporaryDirectory(prefix=".discovery-event-", dir=output.parent) as directory:
            temporary = Path(directory) / "scores.pt"
            score_cluster_features(
                topk_run_dir=paths["topk_manifest"].parent, event_features_path=paths["event_features"],
                cluster_assignments_path=paths["cluster_assignments"], cluster_annotations_path=None,
                clusters_path=paths["clusters"], output_path=temporary,
                prompt_records_path=paths["prompts"], allowed_episode_nums=context["allowed"],
                window_size=5, top_n=20, step_mapping="inference_step")
            payload = torch.load(temporary, map_location="cpu", weights_only=False)
            _verify_scored(payload, context)
            payload["scope"] = context["scope"]
            payload["synthetic"] = False
            payload["derivation"] = {"schema_version": SCHEMA, "operation": "discovery_only_reaggregation",
                                     "signature_hash": context["signature_hash"],
                                     "signature": context["signature"],
                                     "population_binding": context["index_evidence"],
                                     "event_formula": "unchanged_inherited_max_temporal_projection",
                                     "clustering_refit": False, "historical_state_identity_verified": False}
            torch.save(payload, temporary)
            os.link(temporary, output)
        atomic_write_json(receipt, {"schema_version": SCHEMA,
                                    "signature_hash": context["signature_hash"],
                                    "output_sha256": sha256_file(output)})
        status = "created"
    # Only publish a config change after the score artifact and receipt agree.
    from .provenance import atomic_write_text
    if sha256_file(config_path) != original_config_hash:
        raise ValueError("Config changed during Event scoring; artifact is saved but config was not updated")
    atomic_write_text(config_path, yaml.safe_dump(updated, sort_keys=False), overwrite=True)
    return {"status": status, "output": str(output), "receipt": str(receipt),
            "signature_hash": context["signature_hash"], "scope": context["scope"],
            "discovery_episodes": len(context["allowed"]),
            "coverage": context["index_evidence"], "updated_config": str(config_path)}
