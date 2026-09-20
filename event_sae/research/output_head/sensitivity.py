"""Reference single-feature edits and bounded fixed-prefix head scoring."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .head import (EDIT_PARITY_CHECKS, HEAD_PARITY_CHECKS, OutputHeadBundle,
                   decode_action_bins, resolve_dtype, validate_parity_report)

PRIMARY_METRIC = "head_full_vocab_kl_fixed_prefix"
METRICS = (PRIMARY_METRIC, "full_vocab_argmax_flip_rate", "decoded_bin_change_rate",
           "normalized_bin_abs_change", "mean_readout_activation",
           "readout_activation_frequency", "mean_edit_norm")


@torch.inference_mode()
def edit_feature_reference(hidden: torch.Tensor, sae: Any, feature_id: int, alpha: float,
                           hidden_dtype: str | torch.dtype | None = None, *,
                           encoded: torch.Tensor | None = None) -> torch.Tensor:
    """Match ``flat + (decode(edited) - decode(encoded))`` in the legacy hook.

    Supply original-forward ``encoded`` for cached readouts. If absent, the
    complete provided forward is encoded together, preserving its grouping.
    """
    if hidden.ndim < 2 or not hidden.is_floating_point() or not torch.isfinite(hidden).all():
        raise ValueError("Hidden must be a finite floating tensor with at least two dimensions")
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")
    dtype = resolve_dtype(hidden_dtype or hidden.dtype)
    original = hidden.to(dtype=dtype)
    flat = original.reshape(-1, original.shape[-1]).to(torch.float32)
    z = sae.encode(flat) if encoded is None else encoded
    if z.ndim != 2 or z.shape[0] != flat.shape[0] or z.dtype != torch.float32:
        raise ValueError("Encoded latents must be FP32 [forward rows, dictionary]")
    if z.device != flat.device or not torch.isfinite(z).all():
        raise ValueError("Encoded latents must be finite and on the hidden device")
    if isinstance(feature_id, bool) or not isinstance(feature_id, int) or not 0 <= feature_id < z.shape[1]:
        raise ValueError("feature_id is outside the dictionary")
    edited = z.clone()
    edited[:, feature_id] = z[:, feature_id] * float(alpha)
    decoded_edit, decoded_base = sae.decode(edited), sae.decode(z)
    if decoded_edit.shape != flat.shape or decoded_base.shape != flat.shape:
        raise ValueError("SAE decoder output does not match hidden shape")
    updated = flat + (decoded_edit - decoded_base)
    if not torch.isfinite(updated).all():
        raise ValueError("Feature edit produced non-finite hidden values")
    return updated.to(dtype=dtype).reshape_as(original)


def full_vocab_kl(base: torch.Tensor, edit: torch.Tensor, *, negative_tolerance: float = 1e-6,
                  reduction_dtype: str | torch.dtype = "float32") -> torch.Tensor:
    """KL(base || edit) over every head logit; no action-only renormalization."""
    if base.shape != edit.shape or base.ndim != 2 or base.shape[-1] < 1:
        raise ValueError("KL inputs must have identical [rows, full vocabulary] shapes")
    if not math.isfinite(negative_tolerance) or negative_tolerance < 0:
        raise ValueError("negative_tolerance must be finite and nonnegative")
    dtype = resolve_dtype(reduction_dtype)
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("KL reduction must use FP32 or FP64")
    if not torch.isfinite(base).all() or not torch.isfinite(edit).all():
        raise ValueError("KL logits contain NaN/Inf")
    logp = torch.log_softmax(base.to(dtype=dtype), dim=-1)
    logq = torch.log_softmax(edit.to(device=base.device, dtype=dtype), dim=-1)
    result = (logp.exp() * (logp - logq)).sum(-1)
    if not torch.isfinite(result).all() or (result < -negative_tolerance).any():
        raise ValueError("Invalid KL: non-finite or materially negative result")
    return result.clamp_min(0)


def aggregation_weights(records: Sequence[Mapping[str, Any]]) -> tuple[torch.Tensor, list[int]]:
    """Equal task → episode → step → dimension means; weights sum to one."""
    if not records:
        raise ValueError("Cannot aggregate an empty readout sample")
    groups: dict[int, dict[str, dict[int, list[int]]]] = {}
    dimension_sets = set()
    for index, record in enumerate(records):
        required = ("task_id", "episode_uid", "step_in_episode", "action_dim_index")
        if any(record.get(key) is None for key in required):
            raise ValueError(f"Readout metadata lacks required aggregation keys: {required}")
        task, episode = record["task_id"], str(record["episode_uid"])
        step, dimension = record["step_in_episode"], record["action_dim_index"]
        if not episode or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (task, step, dimension)):
            raise ValueError("Invalid task/episode/step/action dimension metadata")
        groups.setdefault(task, {}).setdefault(episode, {}).setdefault(step, []).append(index)
    weights = torch.zeros(len(records), dtype=torch.float64)
    for task, episodes in groups.items():
        for episode, steps in episodes.items():
            for step, indices in steps.items():
                dims = [records[index]["action_dim_index"] for index in indices]
                if sorted(dims) != list(range(len(dims))):
                    raise ValueError(f"Duplicate/incomplete action dimensions at {(task, episode, step)}")
                dimension_sets.add(tuple(sorted(dims)))
                weights[indices] = 1.0 / (len(groups) * len(episodes) * len(steps) * len(indices))
    if len(dimension_sets) != 1:
        raise ValueError("Action dimension count changes between readout queries")
    return weights, sorted(groups)


def _read_cache(cache: Mapping[str, Any] | str | Path) -> dict:
    if isinstance(cache, (str, Path)):
        from .readouts import load_readout_cache
        return load_readout_cache(cache)
    return dict(cache)


def _validate_sparse_cache(cache: dict) -> tuple[torch.Tensor, list, list, list, int]:
    hidden, records = cache.get("hidden"), cache.get("records")
    ids = cache.get("feature_ids", cache.get("sparse_feature_ids"))
    values = cache.get("feature_values", cache.get("sparse_feature_values"))
    width = cache.get("dict_size")
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 2 or not hidden.is_floating_point():
        raise ValueError("Readout cache must contain hidden [rows, width]")
    if not torch.isfinite(hidden).all():
        raise ValueError("Readout cache contains non-finite hidden")
    if not isinstance(width, int) or isinstance(width, bool) or width < 1:
        raise ValueError("Readout cache must declare dictionary size")
    if not all(isinstance(rows, (list, tuple)) and len(rows) == hidden.shape[0]
               for rows in (records, ids, values)):
        raise ValueError("Readout metadata/sparse latent rows must match hidden")
    for row_ids, row_values in zip(ids, values):
        if not isinstance(row_ids, torch.Tensor) or not isinstance(row_values, torch.Tensor):
            raise ValueError("Sparse IDs and values must be tensors")
        if row_ids.ndim != 1 or row_values.shape != row_ids.shape:
            raise ValueError("Sparse latent vectors must be matching one-dimensional tensors")
        if row_ids.dtype not in (torch.int32, torch.int64) or row_values.dtype != torch.float32:
            raise ValueError("Sparse IDs must be integer and original encoded values FP32")
        if row_ids.unique().numel() != row_ids.numel() or ((row_ids < 0) | (row_ids >= width)).any():
            raise ValueError("Duplicate or out-of-range sparse feature IDs")
        if not torch.isfinite(row_values).all() or (row_values <= 0).any():
            raise ValueError("Exact sparse cache must contain finite positive nonzeros only")
    if cache.get("encoder_grouping") not in ("original_forward", "readout_only_validated"):
        raise ValueError("Readout cache must certify original-forward encoder grouping")
    if cache.get("truncated", False):
        raise ValueError("Top-K truncated cache cannot establish exact feature absence")
    return hidden, list(records), list(ids), list(values), width


def score_features(readout_cache: Mapping[str, Any] | str | Path, head_bundle: OutputHeadBundle,
                   sae: Any, score_config: Mapping[str, Any]) -> dict:
    """Score exact active pairs with bounded decoder/head batches.

    All metrics use the same complete readout denominator. No dense matrix
    of all readout latents or per-row edited logits is stored or returned.
    """
    cache, config = _read_cache(readout_cache), dict(score_config)
    hidden, records, sparse_ids, sparse_values, dict_size = _validate_sparse_cache(cache)
    synthetic = cache.get("synthetic") is True or cache.get("manifest", {}).get("synthetic") is True
    if not synthetic:
        reports, identity = config.get("parity_reports", {}), config.get("current_identity", {})
        validate_parity_report(reports.get("head", {}), identity, HEAD_PARITY_CHECKS)
        validate_parity_report(reports.get("edit", {}), identity, EDIT_PARITY_CHECKS)
        expected = {"head_weights_hash": head_bundle.manifest.get("weights_sha256"),
                    "model_revision": head_bundle.manifest.get("model_revision"),
                    "code_revision": head_bundle.manifest.get("code_revision"),
                    "layer_idx": head_bundle.manifest.get("target_layer"),
                    "encoder_grouping": cache["encoder_grouping"],
                    "pair_batch_size": config.get("pair_batch_size", 1)}
        if any(value is None or identity.get(key) != value for key, value in expected.items()):
            raise ValueError("Current parity identity does not describe the actual scoring bundle/cache")
    if config.get("edit_backend", "reference") != "reference":
        raise ValueError("Only reference edit is implemented; no silent fast approximation")
    if config.get("primary_metric", PRIMARY_METRIC) != PRIMARY_METRIC or config.get("full_vocab", True) is not True:
        raise ValueError("This scorer requires fixed-prefix full-vocabulary KL")
    if config.get("aggregation", "equal_task_episode_step_dimension") != "equal_task_episode_step_dimension":
        raise ValueError("Unsupported aggregation")
    if config.get("include_inactive_as_zero", True) is not True:
        raise ValueError("Inactive readouts must remain in every feature denominator")
    arithmetic_mode = config.get("arithmetic_mode", "isolated_row_bf16_reference_v1")
    if arithmetic_mode != "isolated_row_bf16_reference_v1":
        raise ValueError("This scorer requires the explicit isolated-row BF16 reference protocol")
    reduction_dtype = resolve_dtype(config.get("reduction_dtype", "float32"))
    if reduction_dtype not in (torch.float32, torch.float64):
        raise ValueError("KL reduction must use FP32 or FP64")
    negative_tolerance = float(config.get("negative_tolerance", config.get("negative_kl_tolerance", 1e-6)))
    if not math.isfinite(negative_tolerance) or negative_tolerance < 0:
        raise ValueError("negative_kl_tolerance must be finite and nonnegative")
    alpha = float(config.get("alpha", 0.0))
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")
    batch_size, maximum = config.get("pair_batch_size", 1), config.get("max_scored_pairs", 20000)
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (batch_size, maximum)):
        raise ValueError("Batch size and pair budget must be positive integers")
    if not synthetic and batch_size != 1:
        raise ValueError("Real isolated-row reference scoring requires pair_batch_size=1")
    selected = config.get("feature_universe", "all_dictionary_features")
    if selected == "all_dictionary_features":
        features, universe = list(range(dict_size)), "all_dictionary_features"
    elif isinstance(selected, (list, tuple)):
        features, universe = sorted(selected), "screened_subset"
        if not features or len(set(features)) != len(features) or any(
            isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < dict_size for v in features
        ):
            raise ValueError("Feature subset must contain unique valid feature IDs")
    else:
        raise ValueError("Unsupported feature universe")
    selected_set = set(features)
    pair_count = sum(sum(int(v) in selected_set for v in row.tolist()) for row in sparse_ids)
    if pair_count > maximum:
        raise ValueError(f"Pair budget exceeded: {pair_count} active pairs > {maximum}; no score computed")
    weights, tasks = aggregation_weights(records)
    device = head_bundle.head_weight.device
    vocab, dim = head_bundle.head_weight.shape
    if hidden.shape[1] != dim:
        raise ValueError("Readout/head hidden width mismatch")
    # Conservative working estimate; excludes already allocated model/SAE and
    # allocator overhead. Every actual intermediate is row/pair-batch bounded.
    estimated_bytes = batch_size * (dict_size * 4 * 3 + dim * 4 * 4 + vocab * 4 * 8)
    memory_cap = config.get("max_working_memory_bytes")
    if memory_cap is not None and estimated_bytes > int(memory_cap):
        raise ValueError(f"Estimated working-memory budget exceeded: {estimated_bytes} bytes")
    task_index, feature_index = {t: i for i, t in enumerate(tasks)}, {f: i for i, f in enumerate(features)}
    totals = torch.zeros((len(tasks), len(features), len(METRICS)), dtype=torch.float64)
    counts = torch.zeros((len(tasks), len(features)), dtype=torch.int64)
    dimensions = max(record["action_dim_index"] for record in records) + 1
    dimension_totals = torch.zeros((dimensions, len(features), len(METRICS)), dtype=torch.float64)
    dimension_counts = torch.zeros((dimensions, len(features)), dtype=torch.int64)
    task_n = {task: sum(record["task_id"] == task for record in records) for task in tasks}
    action = head_bundle.manifest.get("action_decoding")
    has_action = isinstance(action, dict) and action.get("verified") is True
    if has_action:
        decode_action_bins(torch.tensor([0]), action)
    started, executed_pairs = time.monotonic(), 0
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            end = min(start + batch_size, len(records))
            h = hidden[start:end].to(device=device, dtype=head_bundle.hidden_dtype)
            z = torch.zeros((end - start, dict_size), dtype=torch.float32, device=device)
            pairs = []
            for local, row in enumerate(range(start, end)):
                z[local, sparse_ids[row].to(device=device, dtype=torch.int64)] = sparse_values[row].to(device)
                pairs.extend((local, fid) for fid in sparse_ids[row].tolist() if fid in selected_set)
            base_logits = head_bundle(h)
            if not torch.isfinite(base_logits).all():
                raise ValueError("Baseline head returned NaN/Inf")
            base_logp = torch.log_softmax(base_logits.to(reduction_dtype), dim=-1)
            for pair_start in range(0, len(pairs), batch_size):
                block = pairs[pair_start:pair_start + batch_size]
                local_rows = torch.tensor([local for local, _ in block], dtype=torch.int64, device=device)
                feature_ids = torch.tensor([fid for _, fid in block], dtype=torch.int64, device=device)
                pair_rows = torch.arange(len(block), device=device)
                original, encoded = h[local_rows], z[local_rows]
                perturbed = encoded.clone()
                perturbed[pair_rows, feature_ids] = encoded[pair_rows, feature_ids] * alpha
                # Each pair has its own original latent row and independently
                # edits one coordinate. Both decodes use the same batch shape.
                delta = sae.decode(perturbed) - sae.decode(encoded)
                if delta.shape != original.shape or not torch.isfinite(delta).all():
                    raise ValueError("SAE decoder produced an invalid edited hidden")
                edited = (original.float() + delta).to(dtype=head_bundle.hidden_dtype)
                changed_logits, raw = head_bundle(edited), base_logits[local_rows]
                if not torch.isfinite(changed_logits).all():
                    raise ValueError("Edited head returned NaN/Inf")
                logp = base_logp[local_rows]
                logq = torch.log_softmax(changed_logits.to(reduction_dtype), dim=-1)
                kl = (logp.exp() * (logp - logq)).sum(-1)
                if not torch.isfinite(kl).all() or (kl < -negative_tolerance).any():
                    raise ValueError("Invalid KL: non-finite or materially negative result")
                kl = kl.clamp_min(0)
                top_base, top_edit = raw.argmax(-1), changed_logits.argmax(-1)
                flip = (top_base != top_edit).float()
                if has_action:
                    bins_base, bins_edit = decode_action_bins(top_base, action), decode_action_bins(top_edit, action)
                    bin_change, bin_abs = (bins_base != bins_edit).float(), (bins_base - bins_edit).abs()
                else:
                    bin_change, bin_abs = torch.zeros_like(kl), torch.zeros_like(kl)
                values = torch.stack((kl, flip, bin_change, bin_abs, encoded[pair_rows, feature_ids],
                                      torch.ones_like(kl),
                                      (edited.float() - original.float()).norm(dim=-1)), dim=1).double().cpu()
                for i, (local, fid) in enumerate(block):
                    row, ti, fi = start + local, task_index[records[start + local]["task_id"]], feature_index[fid]
                    totals[ti, fi] += weights[row] * len(tasks) * values[i]
                    counts[ti, fi] += 1
                    dimension = records[row]["action_dim_index"]
                    dimension_totals[dimension, fi] += weights[row] * dimensions * values[i]
                    dimension_counts[dimension, fi] += 1
                executed_pairs += len(block)
    feature_rows, task_rows, dimension_rows, averages = [], [], [], totals.mean(0)
    for fi, fid in enumerate(features):
        count = int(counts[:, fi].sum())
        common = {"feature_id": fid, "alpha": alpha,
                  "score_status": "scored" if count else "inactive_in_sample"}
        row = {**common, **{metric: float(averages[fi, mi]) for mi, metric in enumerate(METRICS)},
               "num_readouts": len(records), "num_active_readouts": count, "sampled_task_count": len(tasks)}
        if not has_action:
            row.update(decoded_bin_change_rate=None, normalized_bin_abs_change=None)
        feature_rows.append(row)
        for ti, task in enumerate(tasks):
            task_row = {**common, "task_id": task,
                        **{metric: float(totals[ti, fi, mi]) for mi, metric in enumerate(METRICS)},
                        "num_readouts": task_n[task], "num_active_readouts": int(counts[ti, fi])}
            if not has_action:
                task_row.update(decoded_bin_change_rate=None, normalized_bin_abs_change=None)
            task_rows.append(task_row)
        for dimension in range(dimensions):
            dimension_row = {**common, "action_dim_index": dimension,
                             **{metric: float(dimension_totals[dimension, fi, mi])
                                for mi, metric in enumerate(METRICS)},
                             "num_readouts": len(records) // dimensions,
                             "num_active_readouts": int(dimension_counts[dimension, fi])}
            if not has_action:
                dimension_row.update(decoded_bin_change_rate=None, normalized_bin_abs_change=None)
            dimension_rows.append(dimension_row)
    manifest = cache.get("manifest", {})
    return {"schema_version": "output_head_scores_v1", "synthetic": synthetic,
            "feature_scores": feature_rows, "feature_task_scores": task_rows,
            "feature_dimension_scores": dimension_rows,
            "scope": {"task_ids": tasks,
                      "discovery_manifest_hash": cache.get("discovery_manifest_hash", manifest.get("discovery_manifest_hash")),
                      "sample_manifest_hash": cache.get("sample_manifest_hash", manifest.get("sample_manifest_hash"))},
            "diagnostics": {"num_readouts": len(records), "active_pair_count": pair_count,
                            "executed_pairs": executed_pairs, "dictionary_size": dict_size,
                            "feature_universe": universe, "num_scored_features": len(features),
                            "estimated_working_bytes": estimated_bytes,
                            "pair_batch_size": batch_size, "elapsed_seconds": time.monotonic() - started,
                            "action_decoding_status": "available" if has_action else "unavailable",
                            "action_decoding_reason": None if has_action else "verified action decoding metadata absent",
                            "encoder_grouping": cache["encoder_grouping"],
                            "arithmetic_mode": arithmetic_mode, "edit_backend": "reference"}}
