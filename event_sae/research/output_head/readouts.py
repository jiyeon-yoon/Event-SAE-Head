"""Strict last-layer generation readouts and small, lossless derived caches.

Metadata audit and sampling do not import torch or any model/simulator package.
Tensor loading is explicit in ``prepare_readout_cache`` and never downloads.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


MAPPING_VERSION = "openvla_complete_forward_readout_v1"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _integer(value: Any, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _load_records(value: Any) -> tuple[list[dict], Path | None]:
    path = None
    if isinstance(value, (str, Path)):
        path = Path(value)
        with path.open(encoding="utf-8") as stream:
            value = ([json.loads(line) for line in stream if line.strip()]
                     if path.suffix == ".jsonl" else json.load(stream))
    if isinstance(value, Mapping):
        container = value
        value = container.get("records", container.get("episodes"))
        if isinstance(value, list):
            shared = {key: container[key] for key in ("source_run_id", "suite") if key in container}
            value = [{**shared, **row} if isinstance(row, dict) else row for row in value]
    if not isinstance(value, list) or not value or not all(isinstance(row, dict) for row in value):
        raise ValueError("index must contain a nonempty list of forward records")
    return [dict(row) for row in value], path


def _generation(spec: Mapping[str, Any]) -> dict:
    if not isinstance(spec, Mapping):
        raise ValueError("generation_spec is required")
    result = dict(spec)
    _integer(result.get("action_dim"), "action_dim", 1)
    _integer(result.get("layer_idx"), "layer_idx")
    if result.get("batch_size") != 1 or isinstance(result.get("batch_size"), bool):
        raise ValueError("verified batch_size=1 is required; row count is not evidence")
    if result.get("padding") != "none":
        raise ValueError("only verified padding='none' is supported")
    if result.get("use_cache") is not True:
        raise ValueError("verified cached autoregressive generation is required")
    evidence = result.get("evidence")
    if not evidence or not isinstance(evidence, (str, dict)):
        raise ValueError("batch/padding/generation evidence is required")
    if result.get("index_scope", "complete") not in {"complete", "subset"}:
        raise ValueError("index_scope must be complete or subset")
    return result


def _identity(row: Mapping[str, Any], spec: Mapping[str, Any]) -> str:
    explicit = row.get("source_run_id")
    if isinstance(explicit, str) and explicit:
        return explicit
    if row.get("source_run_idx") is not None:
        run_idx = _integer(row["source_run_idx"], "source_run_idx")
        namespace = spec.get("dataset_id")
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("merged source_run_idx requires an explicit dataset_id namespace")
        return f"{namespace}:run:{run_idx}"
    if isinstance(spec.get("source_run_id"), str) and spec["source_run_id"]:
        return spec["source_run_id"]
    raise ValueError("source_run_id is unknown; local episode/forward IDs are not global identities")


def _even_steps(values: list[int], count: int | None) -> list[int]:
    if count is None or count >= len(values):
        return values
    count = _integer(count, "max_steps_per_episode", 1)
    if count == 1:
        return [values[(len(values) - 1) // 2]]
    return [values[(i * (len(values) - 1)) // (count - 1)] for i in range(count)]


def discovery_population_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash original task/state identities, independently of source renumbering.

Unknown state hashes remain explicit nulls; this helper cannot establish that
an unverified legacy population is a strict held-out comparator.
"""
    population = {}
    for row in rows:
        identity = {key: row.get(key) for key in ("suite", "task_id", "task_episode_idx", "initial_state_sha256")}
        _integer(identity["task_id"], "task_id")
        _integer(identity["task_episode_idx"], "task_episode_idx")
        population[_digest(identity)] = identity
    return _digest([population[key] for key in sorted(population)])


def build_readout_manifest(index_path_or_records: Any, sample_spec: Mapping[str, Any],
                           generation_spec: Mapping[str, Any], *, dense_dir: str | Path | None = None,
                           source_episodes: Any = None, strict: bool = True) -> dict:
    """Map complete original forwards, then sample whole episodes/steps.

``generation_spec`` records verified batch/padding/cache/action dimension
evidence. Numeric gaps in global_forward_idx are reported, not guessed away.
Malformed records, ambiguous identities, and overlapping ranges always fail.
``strict=False`` may exclude malformed *step groups*, with explicit reporting.
"""
    spec = _generation(generation_spec)
    sample_spec = dict(sample_spec)
    records, index_path = _load_records(index_path_or_records)
    root = Path(dense_dir) if dense_dir is not None else (index_path.parent if index_path else None)
    episode_lookup: dict[tuple, dict] = {}
    if source_episodes is not None:
        episodes, _ = _load_records(source_episodes)
        for episode in episodes:
            key = (_identity(episode, spec), _integer(episode.get("episode_num"), "episode_num"))
            if key in episode_lookup:
                raise ValueError("duplicate source episode manifest identity")
            episode_lookup[key] = episode
    groups: dict[tuple, list[dict]] = defaultdict(list)
    intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    forwards: dict[str, set[int]] = defaultdict(set)
    episode_ids: dict[tuple, tuple] = {}
    shard_inventory = {}
    row_counts = Counter()
    for original in records:
        row = dict(original)
        run = _identity(row, spec)
        episode_num = _integer(row.get("episode_num"), "episode_num")
        extra = episode_lookup.get((run, episode_num), {})
        for field in ("task_id", "task_episode_idx", "initial_state_sha256", "suite"):
            if field in extra:
                if row.get(field) is not None and row[field] != extra[field]:
                    raise ValueError(f"index and source episode disagree on {field}")
                row[field] = extra[field]
        task = _integer(row.get("task_id"), "task_id")
        trial = _integer(row.get("task_episode_idx"), "task_episode_idx")
        step = _integer(row.get("step_in_episode"), "step_in_episode")
        forward = _integer(row.get("global_forward_idx"), "global_forward_idx")
        if _integer(row.get("layer_idx"), "layer_idx") != spec["layer_idx"]:
            raise ValueError("index contains another target layer")
        if "batch_size" in row and row["batch_size"] != 1:
            raise ValueError("record contradicts batch_size=1 evidence")
        if row.get("padding", "none") != "none":
            raise ValueError("record contradicts padding evidence")
        start = _integer(row.get("row_start"), "row_start")
        end = _integer(row.get("row_end"), "row_end", 1)
        if end <= start:
            raise ValueError("invalid end-exclusive row range")
        shard = row.get("shard_path")
        if not isinstance(shard, str) or not shard:
            raise ValueError("shard_path is missing")
        path = (root / shard) if root is not None else Path(shard)
        # Keep relative names when no filesystem root was supplied (CPU fixtures).
        shard_key = str(path.absolute()) if root is not None else shard
        if root is not None:
            if not path.is_file():
                raise FileNotFoundError(f"missing dense shard: {path}")
            stat = path.stat()
            shard_inventory[shard_key] = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                                          "identity_level": "stat_only_not_content_verified"}
        intervals[shard_key].append((start, end))
        if forward in forwards[run]:
            raise ValueError(f"duplicate global_forward_idx in source run {run}: {forward}")
        forwards[run].add(forward)
        episode_key = (run, episode_num)
        if episode_key in episode_ids and episode_ids[episode_key] != (task, trial):
            raise ValueError("one source episode maps to multiple task/original-trial identities")
        episode_ids[episode_key] = (task, trial)
        uid = _digest([run, task, trial, episode_num])
        row.update(source_run_id=run, episode_uid=uid, source_shard=shard_key,
                   source_row=end - 1, forward_idx=forward, task_id=task,
                   task_episode_idx=trial, step_in_episode=step)
        if "suite" not in row and spec.get("suite"):
            row["suite"] = spec["suite"]
        groups[(run, task, trial, episode_num, step)].append(row)
        row_counts[end - start] += 1
    range_gaps = []
    metadata = spec.get("shard_metadata", {})
    for shard, ranges in intervals.items():
        previous = 0
        for start, end in sorted(ranges):
            if start < previous:
                raise ValueError(f"overlapping or duplicate row ranges: {shard}")
            if start > previous:
                range_gaps.append({"shard": shard, "row_start": previous, "row_end": start})
            previous = end
        item = metadata.get(shard, metadata.get(Path(shard).name, {}))
        total_rows = item.get("num_rows") if isinstance(item, Mapping) else None
        if total_rows is not None and previous != total_rows:
            raise ValueError(f"index row coverage disagrees with shard metadata: {shard}")
    if range_gaps and spec.get("index_scope", "complete") == "complete":
        raise ValueError("noncontiguous shard row ranges in a complete index")
    mapped, excluded = [], []
    dimension = spec["action_dim"]
    step_counts = Counter(len(rows) for rows in groups.values())
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda row: row["global_forward_idx"])
        reason = None
        if len(rows) != dimension:
            reason = f"expected {dimension} prediction forwards, found {len(rows)}"
        elif any(row["row_end"] - row["row_start"] != 1 for row in rows[1:]):
            reason = "cached decode forward must contain exactly one row for batch size 1"
        elif any("action_dim_index" in row and row["action_dim_index"] != i for i, row in enumerate(rows)):
            reason = "action dimension metadata disagrees with generation order"
        if reason:
            excluded.append({"step_identity": list(key), "reason": reason})
            if strict:
                raise ValueError(f"invalid action query {key}: {reason}")
            continue
        for action_dim, row in enumerate(rows):
            mapped.append({**row, "action_dim_index": action_dim})
    if not mapped:
        raise ValueError("no complete action queries remain")
    mode = sample_spec.get("mode", "pilot")
    if mode not in {"pilot", "confirmatory"}:
        raise ValueError("sampling mode must be pilot or confirmatory")
    split = sample_spec.get("split_manifest")
    if mode == "confirmatory" and split is None:
        raise ValueError("confirmatory sampling requires a pre-frozen split manifest")
    if split is not None:
        from .splits import episode_key, validate_split_manifest
        if isinstance(split, (str, Path)):
            with Path(split).open(encoding="utf-8") as stream:
                split = json.load(stream)
        validate_split_manifest(split, require_confirmatory=True)
        # Score computation always uses discovery, even during a protected pilot.
        members = {episode_key(row): row for row in split["splits"]["discovery"]}
        filtered = []
        for row in mapped:
            member = members.get(episode_key(row))
            if member is not None:
                if row.get("initial_state_sha256") != member["initial_state_sha256"]:
                    raise ValueError("readout state hash does not match frozen discovery manifest")
                filtered.append(row)
        mapped = filtered
        if not mapped:
            raise ValueError("no discovery episodes available in the frozen split")
        sample_spec["split_manifest_hash"] = split["manifest_hash"]
    tasks = sorted({row["task_id"] for row in mapped})
    wanted_tasks = sample_spec.get("task_ids", tasks)
    if not isinstance(wanted_tasks, list) or not wanted_tasks or len(set(wanted_tasks)) != len(wanted_tasks):
        raise ValueError("task_ids must be a nonempty unique list")
    if not set(wanted_tasks) <= set(tasks):
        raise ValueError("requested sample task has no valid complete action queries")
    seed = _integer(sample_spec.get("sample_seed", 2026), "sample_seed")
    cap = sample_spec.get("max_episodes_per_task")
    if cap is not None:
        _integer(cap, "max_episodes_per_task", 1)
    if sample_spec.get("step_selection", "evenly_spaced") != "evenly_spaced":
        raise ValueError("only explicit evenly_spaced step sampling is supported")
    if sample_spec.get("require_complete_action_query", True) is not True:
        raise ValueError("complete action queries cannot be disabled")
    selected_episodes = set()
    for task in sorted(wanted_tasks):
        _integer(task, "task_id")
        episodes = sorted({row["episode_uid"] for row in mapped if row["task_id"] == task})
        rng = random.Random(int(_digest([seed, task]), 16))
        selected_episodes.update(rng.sample(episodes, min(cap or len(episodes), len(episodes))))
    selected_steps = {}
    for episode in sorted(selected_episodes):
        steps = sorted({row["step_in_episode"] for row in mapped if row["episode_uid"] == episode})
        selected_steps[episode] = set(_even_steps(steps, sample_spec.get("max_steps_per_episode")))
    selected = [row for row in mapped if row["episode_uid"] in selected_episodes
                and row["step_in_episode"] in selected_steps[row["episode_uid"]]]
    discovery_episodes = {}
    for row in selected:
        discovery_episodes[row["episode_uid"]] = {
            key: row.get(key) for key in ("source_run_id", "suite", "task_id", "task_episode_idx",
                                         "episode_uid", "initial_state_sha256")}
    discovery = [discovery_episodes[key] for key in sorted(discovery_episodes)]
    gap_counts = {}
    for run, values in forwards.items():
        ordered = sorted(values)
        gap_counts[run] = sum(max(0, second - first - 1) for first, second in zip(ordered, ordered[1:]))
    audit = {"num_tasks": len(tasks), "num_episodes": len(episode_ids), "num_steps": len(groups),
             "num_forwards": len(records), "forward_row_count_histogram": dict(sorted(row_counts.items())),
             "step_forward_count_histogram": dict(sorted(step_counts.items())),
             "mapped_steps": len(groups) - len(excluded), "excluded_steps": excluded,
             "exclusion_fraction": len(excluded) / len(groups), "row_range_gaps": range_gaps,
             "forward_id_gaps": gap_counts, "forward_id_gaps_are_not_proof_of_missing_rows": True,
             "shard_validation": "existence_and_index_ranges" if root is not None else "metadata_only",
             "generation_evidence": spec["evidence"], "num_selected_readouts": len(selected)}
    manifest = {"schema_version": "output_head_readout_manifest_v1", "mapping_version": MAPPING_VERSION,
                "source_index_hash": _digest(records), "generation_spec": spec,
                "source_index_path": str(index_path.absolute()) if index_path else None,
                "sample_spec": dict(sample_spec), "readouts": selected, "audit": audit,
                "discovery_episodes": discovery, "discovery_manifest_hash": discovery_population_hash(discovery),
                "split_manifest_hash": sample_spec.get("split_manifest_hash"),
                "source_shards": shard_inventory, "dense_dir": str(root.absolute()) if root else None,
                "indexed_shard_rows": {key: max(end for _, end in ranges) for key, ranges in intervals.items()},
                "synthetic": sample_spec.get("synthetic", False)}
    manifest["mapping_hash"] = _digest({"index": manifest["source_index_hash"], "generation": spec,
                                        "mapping_version": MAPPING_VERSION})
    manifest["sample_hash"] = _digest({"mapping_hash": manifest["mapping_hash"], "sample": sample_spec,
                                       "readouts": selected})
    return manifest


def audit_readout_index(index_path_or_records: Any, generation_spec: Mapping[str, Any], **kwargs: Any) -> dict:
    """Audit all metadata, reporting bad query groups without reading tensors."""
    return build_readout_manifest(index_path_or_records, {}, generation_spec, strict=False, **kwargs)["audit"]


def _atomic_tensor(path: Path, payload: dict) -> None:
    import torch
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_readout_cache(manifest: Mapping[str, Any], sae: Any, output_dir: str | Path | None = None,
                          *, dense_dir: str | Path | None = None, device: str = "cpu",
                          sae_id: str, hidden_dtype: str = "bfloat16",
                          encoder_mode: str = "original_forward", readout_batch_size: int = 32,
                          rows_per_shard: int = 1024, resume: bool = True,
                          sae_implementation_id: str | None = None) -> dict:
    """Encode original complete forwards and store every nonzero latent.

Default encoding exactly preserves original forward grouping. Readout-only
mode is allowed only if single-row and pooled encodings exactly match the
original-forward encoding for every selected row; failure is explicit.
The returned cache is directly usable by the offline scoring module.
"""
    import torch

    if encoder_mode not in {"original_forward", "readout_only"}:
        raise ValueError("unknown encoder_mode")
    if getattr(sae, "training", False):
        raise ValueError("SAE must be in eval mode")
    if not isinstance(sae_id, str) or not sae_id:
        raise ValueError("sae_id checkpoint/implementation identity is required")
    if sae_implementation_id is None:
        try:
            sae_implementation_id = hashlib.sha256(inspect.getsource(type(sae)).encode()).hexdigest()
        except (OSError, TypeError):
            raise ValueError("SAE source is unavailable; explicit sae_implementation_id is required") from None
    _integer(readout_batch_size, "readout_batch_size", 1)
    _integer(rows_per_shard, "rows_per_shard", 1)
    dtype = getattr(torch, hidden_dtype, None)
    if dtype not in {torch.float32, torch.float16, torch.bfloat16, torch.float64}:
        raise ValueError("unsupported hidden_dtype")
    if manifest.get("schema_version") != "output_head_readout_manifest_v1":
        raise ValueError("unsupported readout manifest")
    if manifest.get("source_index_path"):
        current_records, _ = _load_records(manifest["source_index_path"])
        if _digest(current_records) != manifest["source_index_hash"]:
            raise ValueError("source activation index changed since mapping")
    mapping_hash = _digest({"index": manifest["source_index_hash"], "generation": manifest["generation_spec"],
                            "mapping_version": manifest["mapping_version"]})
    if mapping_hash != manifest.get("mapping_hash"):
        raise ValueError("mapping manifest hash mismatch")
    records = manifest.get("readouts")
    if not isinstance(records, list) or not records:
        raise ValueError("empty readout manifest")
    expected_sample = _digest({"mapping_hash": manifest["mapping_hash"], "sample": manifest["sample_spec"],
                               "readouts": records})
    if expected_sample != manifest.get("sample_hash"):
        raise ValueError("sample manifest hash mismatch")
    root = Path(dense_dir or manifest.get("dense_dir") or ".")
    sources: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    identities = {}
    for index, row in enumerate(records):
        path = Path(row["source_shard"])
        if not path.is_absolute():
            path = root / path
        path = path.absolute()
        stat = path.stat()
        identity = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                    "identity_level": "stat_only_not_content_verified"}
        previous = manifest.get("source_shards", {}).get(str(path))
        if previous and identity != previous:
            raise ValueError(f"source shard changed since mapping: {path}")
        identities[str(path)] = identity
        sources[str(path)].append((index, row))
    signature_data = {"sample_hash": manifest["sample_hash"], "mapping_hash": manifest["mapping_hash"],
                      "discovery_manifest_hash": manifest["discovery_manifest_hash"],
                      "sae_id": sae_id, "hidden_dtype": hidden_dtype, "encoder_mode": encoder_mode,
                      "readout_batch_size": readout_batch_size, "source_shards": identities,
                      "split_manifest_hash": manifest["sample_spec"].get("split_manifest_hash"),
                      "encoder_implementation": sae_implementation_id,
                      "cache_implementation": _file_hash(Path(__file__)),
                      "torch_version": str(torch.__version__), "encoding_device": device,
                      "schema_version": "output_head_readout_cache_v1"}
    signature = _digest(signature_data)
    destination = Path(output_dir) if output_dir is not None else None
    if destination is not None:
        destination = destination.absolute()
        if dense_dir is not None or manifest.get("dense_dir"):
            input_root = root.resolve()
            if destination.resolve() == input_root or input_root in destination.resolve().parents:
                raise ValueError("readout output must be outside existing input directories")
        for path in sources:
            source_parent = Path(path).resolve().parent
            if destination.resolve() == source_parent or source_parent in destination.resolve().parents:
                raise ValueError("readout output must be outside existing input directories")
        cache_file = destination / "manifest.json"
        if cache_file.exists():
            if not resume:
                raise FileExistsError(cache_file)
            with cache_file.open(encoding="utf-8") as stream:
                cached_metadata = json.load(stream)
            if cached_metadata.get("cache_signature") != signature:
                raise ValueError("readout cache resume provenance mismatch")
            return load_readout_cache(destination)
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError("incomplete cache directory; choose a new output directory")
    hidden_rows = [None] * len(records)
    feature_ids, feature_values = [None] * len(records), [None] * len(records)
    grouping_equal, dict_size = True, None
    with torch.inference_mode():
        for path, entries in sorted(sources.items()):
            dense = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            if not isinstance(dense, torch.Tensor) or dense.ndim != 2 or not dense.is_floating_point():
                raise ValueError(f"dense shard must be a floating [rows, hidden] tensor: {path}")
            indexed_rows = manifest.get("indexed_shard_rows", {}).get(path)
            if (manifest["generation_spec"].get("index_scope", "complete") == "complete"
                    and indexed_rows is not None and dense.shape[0] != indexed_rows):
                raise ValueError("indexed row range exceeds or omits actual dense shard rows")
            for index, row in entries:
                start, end, readout = row["row_start"], row["row_end"], row["source_row"]
                if not (0 <= start < end <= dense.shape[0] and start <= readout < end):
                    raise ValueError("readout range exceeds actual dense shard")
                hidden = dense[start:end].to(device=device, dtype=dtype)
                if not torch.isfinite(hidden).all():
                    raise ValueError("nonfinite hidden state")
                encoded = sae.encode(hidden.to(torch.float32))
                if not isinstance(encoded, torch.Tensor) or encoded.ndim != 2 or encoded.shape[0] != end - start:
                    raise ValueError("SAE encode must return [forward_rows, dict_size]")
                if not torch.isfinite(encoded).all():
                    raise ValueError("nonfinite SAE latent")
                if dict_size is None:
                    dict_size = int(encoded.shape[1])
                if encoded.shape[1] != dict_size or dict_size < 1:
                    raise ValueError("inconsistent SAE dictionary size")
                offset = readout - start
                original_z = encoded[offset].detach().cpu().clone()
                one_hidden = hidden[offset:offset + 1].to(torch.float32)
                single = sae.encode(one_hidden)
                grouping_equal = grouping_equal and bool(torch.equal(single[0].detach().cpu(), original_z))
                hidden_rows[index] = hidden[offset].detach().cpu().clone()
                ids = torch.nonzero(original_z != 0, as_tuple=False).flatten()
                feature_ids[index] = ids
                feature_values[index] = original_z[ids].clone()
            del dense
        hidden = torch.stack(hidden_rows)
        for start in range(0, len(records), readout_batch_size):
            batch = hidden[start:start + readout_batch_size].to(device=device, dtype=torch.float32)
            pooled = sae.encode(batch).detach().cpu()
            reference = torch.zeros((len(batch), dict_size), dtype=pooled.dtype)
            for local, row_index in enumerate(range(start, min(start + readout_batch_size, len(records)))):
                reference[local, feature_ids[row_index]] = feature_values[row_index]
            grouping_equal = grouping_equal and bool(torch.equal(pooled, reference))
    if encoder_mode == "readout_only" and not grouping_equal:
        raise ValueError("readout-only encoding differs from original forward grouping")
    cache_metadata = {**signature_data, "cache_signature": signature, "num_readouts": len(records),
                      "provenance": signature_data,
                      "sample_manifest_hash": manifest["sample_hash"],
                      "dict_size": dict_size, "encoder_grouping": encoder_mode,
                      "grouping_check": {"readout_only_exact": grouping_equal,
                                         "scope": "all_selected_rows_single_and_pooled"},
                      "source_index_hash": manifest["source_index_hash"],
                      "synthetic": manifest.get("synthetic", False), "shards": []}
    cache = {"hidden": hidden, "records": records, "feature_ids": feature_ids,
             "feature_values": feature_values, "dict_size": dict_size,
             "encoder_grouping": encoder_mode, "grouping_check": cache_metadata["grouping_check"],
             "manifest": cache_metadata, "provenance": signature_data,
             "synthetic": manifest.get("synthetic", False)}
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(records), rows_per_shard):
            stop = min(start + rows_per_shard, len(records))
            path = destination / f"shard_{len(cache_metadata['shards']):06d}.pt"
            payload = {key: value[start:stop] for key, value in cache.items()
                       if key in {"hidden", "records", "feature_ids", "feature_values"}}
            _atomic_tensor(path, payload)
            cache_metadata["shards"].append({"path": path.name, "num_rows": stop - start,
                                               "sha256": _file_hash(path)})
        descriptor, temporary = tempfile.mkstemp(prefix=".manifest.", dir=destination)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(cache_metadata, stream, indent=2, allow_nan=False)
            os.replace(temporary, destination / "manifest.json")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return cache


def load_readout_cache(path: str | Path) -> dict:
    """Load a selected readout cache after checking every saved shard hash."""
    import torch

    path = Path(path)
    manifest_path = path / "manifest.json" if path.is_dir() else path
    with manifest_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("schema_version") != "output_head_readout_cache_v1":
        raise ValueError("unsupported readout cache schema")
    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict) or _digest(provenance) != metadata.get("cache_signature"):
        raise ValueError("cache provenance signature mismatch")
    if any(metadata.get(key) != value for key, value in provenance.items()):
        raise ValueError("cache metadata disagrees with signed provenance")
    output = {"hidden": [], "records": [], "feature_ids": [], "feature_values": []}
    for shard in metadata["shards"]:
        shard_path = manifest_path.parent / shard["path"]
        if shard_path.resolve().parent != manifest_path.parent.resolve():
            raise ValueError("cache shard path must stay within cache directory")
        if _file_hash(shard_path) != shard["sha256"]:
            raise ValueError("readout cache shard hash mismatch")
        payload = torch.load(shard_path, map_location="cpu", weights_only=True)
        count = shard["num_rows"]
        if not all(len(payload[key]) == count for key in output):
            raise ValueError("cache shard row counts disagree")
        if payload["hidden"].ndim != 2 or not torch.isfinite(payload["hidden"]).all():
            raise ValueError("invalid cached hidden tensor")
        for ids, values in zip(payload["feature_ids"], payload["feature_values"]):
            if ids.ndim != 1 or values.ndim != 1 or len(ids) != len(values):
                raise ValueError("invalid sparse latent shape")
            if ids.dtype != torch.int64 or (ids < 0).any() or (ids >= metadata["dict_size"]).any():
                raise ValueError("invalid sparse latent feature IDs")
            if len(torch.unique(ids)) != len(ids) or not torch.isfinite(values).all() or (values == 0).any():
                raise ValueError("cached sparse latent must contain distinct exact nonzeros")
        output["hidden"].append(payload["hidden"])
        for key in ("records", "feature_ids", "feature_values"):
            output[key].extend(payload[key])
    if not output["hidden"] or len(output["records"]) != metadata["num_readouts"]:
        raise ValueError("cache manifest row count mismatch")
    output["hidden"] = torch.cat(output["hidden"])
    output.update(dict_size=metadata["dict_size"], manifest=metadata, provenance=provenance,
                  encoder_grouping=metadata["encoder_grouping"], grouping_check=metadata["grouping_check"],
                  synthetic=metadata.get("synthetic", False))
    return output
