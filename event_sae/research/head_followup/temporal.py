"""Lossless full-sequence readout latents from existing dense activations.

No policy inference, simulator, training, downloads, or mutation of old inputs.
Original-forward SAE grouping is preserved. Only final prediction readouts
are saved, not all image/prompt token latents; all their nonzeros are retained.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch

from event_sae.research.output_head.provenance import (
    assert_safe_output, atomic_write_json, fingerprint, read_json, sha256_file,
)
from event_sae.research.output_head.readouts import _atomic_tensor, _digest, _load_records
from .metrics import GeneralityAccumulator


def validate_sequences(records, *, action_dim=7):
    if not records:
        raise ValueError("empty temporal readout records")
    groups, identities = defaultdict(lambda: defaultdict(set)), {}
    for row in records:
        uid, step, dim = row.get("episode_uid"), row.get("step_in_episode"), row.get("action_dim_index")
        if (not isinstance(uid, str) or not uid or isinstance(step, bool) or not isinstance(step, int) or step < 0
                or isinstance(dim, bool) or not isinstance(dim, int) or not 0 <= dim < action_dim):
            raise ValueError("invalid episode/step/dimension metadata")
        identity = (row.get("task_id"), row.get("source_run_id"), row.get("task_episode_idx"))
        if uid in identities and identities[uid] != identity:
            raise ValueError("episode identity changes within temporal sequence")
        identities[uid] = identity
        if dim in groups[uid][step]:
            raise ValueError("duplicate readout dimension")
        groups[uid][step].add(dim)
    lengths = {}
    for uid, steps in groups.items():
        ordered = sorted(steps)
        if ordered != list(range(len(ordered))):
            raise ValueError("missing/nonconsecutive temporal steps; never impute zero")
        if any(dims != set(range(action_dim)) for dims in steps.values()):
            raise ValueError("incomplete action dimensions")
        lengths[uid] = len(ordered)
    return lengths


def _validate_manifest(manifest):
    if manifest.get("schema_version") != "output_head_readout_manifest_v1":
        raise ValueError("unsupported readout mapping manifest")
    sampling = manifest["sample_spec"]
    if sampling.get("max_steps_per_episode") is not None:
        raise ValueError("full sequence required, not an eight-step Head cache")
    if manifest.get("audit", {}).get("excluded_steps"):
        raise ValueError("excluded temporal queries are not permitted")
    mapping = _digest({"index": manifest["source_index_hash"], "generation": manifest["generation_spec"],
                       "mapping_version": manifest["mapping_version"]})
    if manifest.get("mapping_hash") != mapping or manifest.get("sample_hash") != _digest({
            "mapping_hash": mapping, "sample": sampling, "readouts": manifest["readouts"]}):
        raise ValueError("mapping/sample hash mismatch")
    index_path = manifest.get("source_index_path")
    if index_path and _digest(_load_records(index_path)[0]) != manifest["source_index_hash"]:
        raise ValueError("source index changed")
    if manifest["generation_spec"].get("action_dim") != 7:
        raise ValueError("this adaptation requires seven prediction readouts")
    return validate_sequences(manifest["readouts"])


@torch.inference_mode()
def prepare_temporal_cache(manifest, sae, output, *, sae_sha256, device="cpu", hidden_dtype="bfloat16",
                           rows_per_shard=1024, max_readouts=500000, reference_cache=None, progress=None):
    lengths = _validate_manifest(manifest)
    records = manifest["readouts"]
    if (isinstance(max_readouts, bool) or not isinstance(max_readouts, int) or max_readouts < 1
            or len(records) > max_readouts):
        raise ValueError("temporal readout budget exceeded")
    if isinstance(rows_per_shard, bool) or not isinstance(rows_per_shard, int) or rows_per_shard < 1:
        raise ValueError("invalid shard row limit")
    if getattr(sae, "training", False):
        raise ValueError("SAE must be frozen in eval mode")
    if not isinstance(sae_sha256, str) or len(sae_sha256) != 64 or any(c not in "0123456789abcdef" for c in sae_sha256):
        raise ValueError("SAE content hash is required")
    dtype = getattr(torch, hidden_dtype, None)
    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("unsupported hidden dtype")
    # Re-encoding is not a new SAE. Prove it agrees with the saved scoring
    # latents wherever the old eight-step sample overlaps the full timeline.
    if not manifest.get("synthetic", False) and reference_cache is None:
        raise ValueError("real temporal encoding requires the frozen Head readout reference cache")
    reference, checked = {}, set()
    def key(row):
        return (row.get("suite"), row["task_id"], row["task_episode_idx"],
                row["step_in_episode"], row["action_dim_index"])
    if reference_cache is not None:
        from event_sae.research.output_head.sensitivity import _validate_sparse_cache
        _, ref_rows, ref_ids, ref_values, _ = _validate_sparse_cache(reference_cache)
        for row, ids, values in zip(ref_rows, ref_ids, ref_values):
            if key(row) in reference:
                raise ValueError("duplicate frozen reference readout")
            reference[key(row)] = (ids.cpu(), values.cpu())
        if not reference:
            raise ValueError("empty frozen Head reference")
        available = {key(row) for row in records}
        if not set(reference) <= available:
            raise ValueError("frozen Head reference is outside the full temporal population")
    root = Path(manifest.get("dense_dir") or ".")
    by_source = defaultdict(list)
    for row in records:
        path = Path(row["source_shard"])
        if not path.is_absolute():
            path = root / path
        by_source[str(path.resolve())].append(row)
    output = assert_safe_output(output, [root, *by_source.keys()])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("temporal output must be empty; no silent partial-cache resume")
    output.mkdir(parents=True, exist_ok=True)
    shards, source_hashes, count, width = [], {}, 0, None
    batch = {"records": [], "feature_ids": [], "feature_values": []}

    def flush():
        if not batch["records"]:
            return
        target = output / f"latents_{len(shards):06d}.pt"
        _atomic_tensor(target, batch)
        shards.append({"path": target.name, "num_rows": len(batch["records"]), "sha256": sha256_file(target)})
        for value in batch.values():
            value.clear()

    for path, selected in sorted(by_source.items()):
        before = Path(path).stat()
        previous = manifest.get("source_shards", {}).get(path)
        if previous and (before.st_size != previous["size_bytes"] or before.st_mtime_ns != previous["mtime_ns"]):
            raise ValueError("source stat changed after mapping")
        source_hashes[path] = sha256_file(path)
        dense = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(dense, torch.Tensor) or dense.ndim != 2 or not dense.is_floating_point():
            raise ValueError("dense source must be a floating matrix")
        indexed = manifest.get("indexed_shard_rows", {}).get(path)
        if indexed is None or indexed != dense.shape[0]:
            raise ValueError("dense source row coverage does not match complete index")
        for row in selected:
            start, end, selected_row = row["row_start"], row["row_end"], row["source_row"]
            if not 0 <= start <= selected_row < end <= len(dense):
                raise ValueError("invalid original forward range")
            hidden = dense[start:end].to(device=device, dtype=dtype)
            if not torch.isfinite(hidden).all():
                raise ValueError("nonfinite dense activation")
            z = sae.encode(hidden.float())
            if (not isinstance(z, torch.Tensor) or z.ndim != 2 or len(z) != len(hidden)
                    or z.dtype != torch.float32 or not torch.isfinite(z).all() or (z < 0).any()):
                raise ValueError("SAE must return finite nonnegative FP32 original-forward latents")
            if width is None:
                width = z.shape[1]
            if z.shape[1] != width or width < 1:
                raise ValueError("dictionary size changed")
            readout = z[selected_row - start].detach().cpu()
            ids = torch.nonzero(readout > 0, as_tuple=True)[0]
            if key(row) in reference:
                old_ids, old_values = reference[key(row)]
                if not torch.equal(ids, old_ids) or not torch.equal(readout[ids], old_values):
                    raise ValueError("re-encoded latent differs from frozen Head reference; investigate runtime/SAE grouping, do not relax silently")
                checked.add(key(row))
            batch["records"].append(dict(row))
            batch["feature_ids"].append(ids.clone())
            batch["feature_values"].append(readout[ids].clone())
            count += 1
            if len(batch["records"]) >= rows_per_shard:
                flush()
        after = Path(path).stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("source changed while encoding")
        del dense
        if progress:
            progress({"readouts_encoded": count, "total_readouts": len(records), "source": path})
    flush()
    if checked != set(reference):
        raise ValueError("not all frozen reference readouts were checked")
    metadata = {
        "schema_version": "head_followup_temporal_cache_v1", "synthetic": manifest.get("synthetic", False),
        "encoder_grouping": "original_forward", "truncated": False, "hidden_dtype": hidden_dtype,
        "time_scope": "all_recorded_consecutive_discovery_steps",
        "token_scope": "seven_final_prediction_readouts_not_all_prompt_or_image_tokens",
        "dict_size": width, "num_readouts": count, "episode_lengths": lengths,
        "sae_sha256": sae_sha256, "source_shard_hashes": source_hashes,
        "source_integrity_scope": "hashes_recorded_now_not_proof_of_unrecorded_historical_runtime",
        "sample_hash": manifest["sample_hash"], "discovery_manifest_hash": manifest["discovery_manifest_hash"],
        "split_manifest_hash": manifest.get("split_manifest_hash"),
        "state_identity_scope": manifest.get("state_identity_scope"),
        "generation_evidence": manifest["generation_spec"].get("evidence"),
        "torch_version": str(torch.__version__), "encoding_device": str(device),
        "frozen_head_reference_rows_checked": len(checked), "frozen_head_reference_exact": bool(reference),
        "implementation_sha256": sha256_file(__file__), "shards": shards,
    }
    metadata["manifest_hash"] = fingerprint(metadata)
    atomic_write_json(output / "manifest.json", metadata)
    return metadata


def load_temporal_cache(path):
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    if (manifest.get("schema_version") != "head_followup_temporal_cache_v1"
            or manifest.get("manifest_hash") != fingerprint({k: v for k, v in manifest.items() if k != "manifest_hash"})
            or manifest.get("encoder_grouping") != "original_forward" or manifest.get("truncated") is not False):
        raise ValueError("invalid/lossy temporal cache manifest")
    output = {"records": [], "feature_ids": [], "feature_values": []}
    for shard in manifest["shards"]:
        file = path / shard["path"]
        if file.resolve().parent != path.resolve() or sha256_file(file) != shard["sha256"]:
            raise ValueError("temporal shard path/hash mismatch")
        data = torch.load(file, map_location="cpu", weights_only=True)
        if any(len(data[k]) != shard["num_rows"] for k in output):
            raise ValueError("temporal shard row count mismatch")
        for ids, values in zip(data["feature_ids"], data["feature_values"]):
            if (ids.ndim != 1 or values.shape != ids.shape or ids.dtype != torch.int64 or values.dtype != torch.float32
                    or ids.unique().numel() != len(ids) or (ids < 0).any() or (ids >= manifest["dict_size"]).any()
                    or not torch.isfinite(values).all() or (values <= 0).any()):
                raise ValueError("invalid lossless sparse latents")
        for key in output:
            output[key].extend(data[key])
    if len(output["records"]) != manifest["num_readouts"] or validate_sequences(output["records"]) != manifest["episode_lengths"]:
        raise ValueError("temporal episode coverage mismatch")
    return {**output, "manifest": manifest}


def temporal_statistics(cache):
    metadata = cache["manifest"]
    if metadata.get("truncated") is not False or metadata.get("encoder_grouping") != "original_forward":
        raise ValueError("statistics require exact original-forward temporal cache")
    lengths = validate_sequences(cache["records"])
    by_episode = defaultdict(list)
    for i, row in enumerate(cache["records"]):
        by_episode[row["episode_uid"]].append(i)
    statistics = GeneralityAccumulator(metadata["dict_size"])
    for uid, indices in sorted(by_episode.items()):
        means = torch.zeros((lengths[uid], metadata["dict_size"]), dtype=torch.float64)
        for i in indices:
            row = cache["records"][i]
            means[row["step_in_episode"], cache["feature_ids"][i]] += cache["feature_values"][i].double() / 7
        statistics.add_episode(uid, list(range(lengths[uid])), means)
    result = statistics.result()
    result["identity"] = {key: metadata[key] for key in (
        "sae_sha256", "discovery_manifest_hash", "split_manifest_hash", "manifest_hash")}
    result["synthetic"] = metadata["synthetic"]
    return result
