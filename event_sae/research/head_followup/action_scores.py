"""Bounded action-token rescoring of the existing sampled Head readouts."""
from __future__ import annotations

import torch
import time
from pathlib import Path

from event_sae.research.output_head.head import EDIT_PARITY_CHECKS, HEAD_PARITY_CHECKS, validate_parity_report
from event_sae.research.output_head.sensitivity import _validate_sparse_cache, aggregation_weights, edit_feature_reference
from .metrics import action_metrics


def action_workflow(config, reference_result, mapping_path, output, *, max_pairs=800000, progress=None):
    """Use fresh validation in config.output, read frozen cache from a separate root."""
    from event_sae.research.output_head.config import load_config
    from event_sae.research.output_head.workflow import numerical_identity, load_parity, _head_path, _sae_paths, sae_implementation_identity
    from event_sae.research.output_head.head import load_output_head
    from event_sae.research.output_head.readouts import load_readout_cache
    from event_sae.research.output_head.provenance import assert_safe_output, read_json, sha256_file, fingerprint, atomic_write_json
    from event_sae.openvla.activations import load_batch_topk_sae
    from .stage1 import validate_mapping

    cfg = load_config(config)
    root = Path(reference_result)
    weights, _ = _sae_paths(cfg)
    output = assert_safe_output(output, [root, weights, mapping_path, config])
    if output.exists():
        raise ValueError("action score output already exists; choose a new filename")
    assert_safe_output(cfg["output"]["root_dir"], [root])
    if cfg["scoring"]["alpha"] != 0 or cfg["scoring"]["pair_batch_size"] != 1:
        raise ValueError("Stage 1 uses complete removal and original isolated-row arithmetic")
    identity = numerical_identity(cfg)
    reports = load_parity(cfg, identity)  # Never borrow the frozen archive's old runtime certificate.
    sample = read_json(root / "sample_manifest.json")
    old = read_json(root / "scores/scores.json")
    if (sha256_file(root / "readouts/manifest.json") != old["identity"]["cache_manifest_hash"]
            or sample["sample_hash"] != old["scope"]["sample_manifest_hash"]
            or identity["sae_checkpoint_hash"] != old["scope"]["sae_sha256"]):
        raise ValueError("frozen cache/sample/SAE binding mismatch")
    cache = load_readout_cache(root / "readouts")
    if cache.get("synthetic") is not False:
        raise ValueError("action CLI needs real frozen readouts")
    sae, _ = load_batch_topk_sae(weights, device=cfg["scoring"]["device"])
    sae_id = fingerprint({"weights": sha256_file(weights), "encoder_implementation": sae_implementation_identity(type(sae))})
    if (cache["manifest"]["sae_id"] != sae_id or cache["manifest"]["sample_hash"] != sample["sample_hash"]
            or identity["head_weights_hash"] != old["identity"]["head_weights_hash"]):
        raise ValueError("current SAE/head differs from frozen scoring inputs")
    head = load_output_head(_head_path(cfg), device=cfg["scoring"]["device"])
    mapping = read_json(mapping_path)
    validate_mapping(mapping, head.manifest)
    started = time.perf_counter()
    result = score_action_cache(cache, head, sae, mapping, max_pairs=max_pairs,
                               current_identity=identity, parity_reports=reports, progress=progress)
    old_by_id = {row["feature_id"]: row for row in old["feature_scores"]}
    errors = [abs(row["full_vocab_kl"] - old_by_id[row["feature_id"]]["head_full_vocab_kl_fixed_prefix"])
              for row in result["feature_scores"]]
    # Diagnostic, not a fabricated claim of cross-hardware bit identity.
    result.update(scope=old["scope"], identity=identity, elapsed_seconds=time.perf_counter() - started,
                  implementation_sha256=sha256_file(__file__), metrics_sha256=sha256_file(Path(__file__).with_name("metrics.py")),
                  source_scores_sha256=sha256_file(root / "scores/scores.json"),
                  replay_full_vocab_max_feature_abs_error=max(errors), new_rollouts=0)
    atomic_write_json(output, result)
    return {"status": "scored", "output": str(output), "executed_pairs": result["executed_pairs"],
            "top3": result["top3"], "elapsed_seconds": result["elapsed_seconds"],
            "replay_full_vocab_max_feature_abs_error": max(errors), "new_rollouts": 0}


@torch.inference_mode()
def score_action_cache(cache, head, sae, mapping, *, max_pairs, current_identity=None, parity_reports=None, progress=None):
    hidden, records, ids, values, width = _validate_sparse_cache(cache)
    synthetic = cache.get("synthetic") is True or cache.get("manifest", {}).get("synthetic") is True
    if not synthetic:
        if (mapping.get("verified") is not True or not mapping.get("evidence") or not mapping.get("tokenizer_sha256")
                or mapping.get("model_revision") != head.manifest.get("model_revision")
                or mapping.get("head_weights_sha256") != head.manifest.get("weights_sha256")):
            raise ValueError("action token mapping must be independently verified against this tokenizer/policy/head")
        if not current_identity or not parity_reports:
            raise ValueError("current head/edit runtime parity is required")
        for key, expected in {"head_weights_hash": head.manifest.get("weights_sha256"),
                              "model_revision": head.manifest.get("model_revision"),
                              "code_revision": head.manifest.get("code_revision"),
                              "layer_idx": head.manifest.get("target_layer"),
                              "encoder_grouping": "original_forward", "pair_batch_size": 1}.items():
            if expected is None or current_identity.get(key) != expected:
                raise ValueError("parity identity differs from actual action scorer inputs")
        validate_parity_report(parity_reports.get("head", {}), current_identity, HEAD_PARITY_CHECKS)
        validate_parity_report(parity_reports.get("edit", {}), current_identity, EDIT_PARITY_CHECKS)
    pairs = sum(len(row) for row in ids)
    if isinstance(max_pairs, bool) or not isinstance(max_pairs, int) or max_pairs < 1 or pairs > max_pairs:
        raise ValueError("action-score active-pair budget exceeded")
    token_ids = mapping.get("action_token_ids", [])
    # Validate indices before the first head/decoder call.
    vocab = head.head_weight.shape[0]
    if (not isinstance(token_ids, list) or not token_ids or len(set(token_ids)) != len(token_ids)
            or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < vocab for i in token_ids)):
        raise ValueError("invalid action token mapping")
    metrics = ("action_conditional_kl", "full_vocab_kl", "action_mass_abs_change", "conditional_action_argmax_flip")
    totals = torch.zeros((width, len(metrics)), dtype=torch.float64)
    signed_mass = torch.zeros(width, dtype=torch.float64)
    counts = torch.zeros(width, dtype=torch.int64)
    weights, _ = aggregation_weights(records)
    base_mass, executed = 0.0, 0
    device = head.head_weight.device
    for index in range(len(records)):
        h = hidden[index:index + 1].to(device=device, dtype=head.hidden_dtype)
        z = torch.zeros((1, width), dtype=torch.float32, device=device)
        z[0, ids[index].to(device)] = values[index].to(device)
        base = head(h)
        base_diagnostic = action_metrics(base, base, token_ids)
        base_mass += float(weights[index]) * float(base_diagnostic["base_action_mass"][0])
        for fid in ids[index].tolist():
            changed = edit_feature_reference(h, sae, fid, 0., encoded=z)
            scored = action_metrics(base, head(changed), token_ids)
            totals[fid] += weights[index] * torch.tensor([float(scored[key][0]) for key in metrics], dtype=torch.float64)
            signed_mass[fid] += weights[index] * float(scored["edited_action_mass"][0] - scored["base_action_mass"][0])
            counts[fid] += 1
            executed += 1
        if progress and (index + 1) % 100 == 0:
            progress({"readouts": index + 1, "total_readouts": len(records), "executed_pairs": executed, "total_pairs": pairs})
    result = [{"feature_id": fid, **{name: float(totals[fid, i]) for i, name in enumerate(metrics)},
               "mean_edited_action_mass": base_mass + float(signed_mass[fid]),
               "num_active_readouts": int(counts[fid]), "num_readouts": len(records)} for fid in range(width)]
    return {"schema_version": "head_action_scores_v1", "synthetic": synthetic, "feature_scores": result,
            "baseline_mean_action_mass": base_mass, "executed_pairs": executed,
            "scope": "same_fixed_prefix_sample_conditional_token_distribution_not_physical_action_error",
            "action_mapping": mapping, "aggregation": "equal_task_episode_step_dimension",
            "source_cache_signature": cache.get("manifest", {}).get("cache_signature"),
            "top3": [r["feature_id"] for r in sorted(result, key=lambda r: (-r["action_conditional_kl"], r["feature_id"]))[:3]]}
