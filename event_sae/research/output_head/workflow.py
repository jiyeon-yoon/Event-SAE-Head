"""Bounded research workflow, with cheap planning separated from execution."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from .provenance import (atomic_write_json, atomic_write_text, fingerprint,
                         implementation_fingerprint, read_json, read_jsonl, sha256_file)


REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSOR_IDENTITY_FILES = (
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
)
REQUIRED_PROCESSOR_IDENTITY_FILES = ("preprocessor_config.json", "tokenizer_config.json")
TOKENIZER_IDENTITY_FILES = ("tokenizer.json", "tokenizer.model")


def output_root(cfg: dict) -> Path:
    value = cfg["output"]["root_dir"]
    if not value:
        raise ValueError("Set output.root_dir to a separate research output directory")
    return Path(value).expanduser().resolve()


def required_input(cfg: dict, key: str) -> Path:
    value = cfg["inputs"].get(key)
    if not value:
        raise ValueError(f"Missing local input: inputs.{key}")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"Missing local input: inputs.{key} ({path})")
    return path


def _sae_paths(cfg: dict) -> tuple[Path, Path]:
    checkpoint = required_input(cfg, "sae_checkpoint")
    directory = checkpoint if checkpoint.is_dir() else checkpoint.parent
    weights = directory / "ae.pt" if checkpoint.is_dir() else checkpoint
    config = directory / "config.json"
    if not config.is_file():
        config = directory.parent / "config.json"
    if not weights.is_file() or not config.is_file():
        raise ValueError("SAE requires existing ae.pt and trainer config.json")
    trainer = read_json(config).get("trainer", {})
    if (trainer.get("dict_class") != "BatchTopKSAE" or
            trainer.get("submodule_name") != "post_mlp_residual" or
            trainer.get("layer") != cfg["scope"]["layer_idx"]):
        raise ValueError("SAE class/layer/capture target does not match the experiment")
    return weights, config


def sae_implementation_identity(cls: type) -> str:
    import inspect
    return fingerprint({"class": inspect.getsource(cls), "encode": inspect.getsource(cls.encode),
                        "decode": inspect.getsource(cls.decode)})


def write_once_json(path: Path, payload: Any, *, resume: bool = True) -> None:
    if path.exists() and resume:
        if read_json(path) != payload:
            raise ValueError(f"Refusing incompatible existing artifact: {path}; use a new output root")
        return
    atomic_write_json(path, payload)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list))
                         else value for key, value in row.items()})
    text = stream.getvalue()
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"Refusing incompatible existing CSV: {path}")
    else:
        atomic_write_text(path, text)


def audit_inputs(cfg: dict) -> dict:
    """Metadata only: does not load torch, scan shard tensors, or contact a hub."""
    required = {"dense_dir", "sae_checkpoint", "generation_manifest"}
    if set(cfg["selection"]["methods"]) & {"event_aligned", "window_mean", "task_mean"}:
        required.add("event_scores_path")
    if cfg["sampling"]["mode"] == "confirmatory":
        required.add("source_episode_manifest")
    inventory, blockers = {}, []
    for name, value in cfg["inputs"].items():
        if name == "existing_result_dirs":
            inventory[name] = [{"path": str(x), "exists": Path(x).expanduser().exists()} for x in value]
            continue
        exists = bool(value and Path(value).expanduser().exists())
        inventory[name] = {"path": value, "exists": exists, "required": name in required}
        if name in required and not exists:
            blockers.append(f"inputs.{name}: missing")
    if not any(inventory[name]["exists"] for name in ("local_model_snapshot", "output_head_bundle")):
        blockers.append("Provide existing local_model_snapshot or output_head_bundle; no download will occur")
    if not cfg["output"]["root_dir"]:
        blockers.append("output.root_dir: missing")
    final_layer = {"status": "pending", "reason": "actual loaded model verification belongs to M4"}
    snapshot = cfg["inputs"]["local_model_snapshot"]
    if snapshot and (Path(snapshot) / "config.json").is_file():
        config = read_json(Path(snapshot) / "config.json")
        language = config.get("text_config", config.get("llm_config", config))
        layers = language.get("num_hidden_layers")
        if layers is not None:
            final_layer.update(local_num_layers=layers,
                               local_config_match=cfg["scope"]["layer_idx"] == layers - 1)
            if not final_layer["local_config_match"]:
                blockers.append("Target layer is not the final decoder layer in local config")
    if inventory["sae_checkpoint"]["exists"]:
        try:
            _, sae_config = _sae_paths(cfg)
            inventory["sae_trainer"] = read_json(sae_config).get("trainer")
        except ValueError as exc:
            blockers.append(str(exc))
    readout_audit = None
    if inventory["dense_dir"]["exists"] and inventory["generation_manifest"]["exists"]:
        from .readouts import audit_readout_index
        try:
            dense = required_input(cfg, "dense_dir")
            readout_audit = audit_readout_index(dense / "activation_index.jsonl",
                                                read_json(required_input(cfg, "generation_manifest")),
                                                dense_dir=dense,
                                                source_episodes=cfg["inputs"]["source_episode_manifest"])
            if readout_audit["excluded_steps"]:
                blockers.append("Malformed action-query groups found; strict prepare will reject them")
        except (ValueError, FileNotFoundError) as exc:
            blockers.append(f"Readout metadata audit: {exc}")
    return {"schema_version": "output_head_audit_v1", "status": "blocked" if blockers else "ready_for_metadata_mapping",
            "synthetic": False, "inventory": inventory, "blockers": blockers,
            "readout_audit": readout_audit,
            "final_layer": final_layer, "runtime_validation": "not_run", "model_download": False,
            "implementation_fingerprint": implementation_fingerprint()}


def split_workflow(cfg: dict) -> dict:
    from .splits import build_split_manifest
    result = build_split_manifest(required_input(cfg, "source_episode_manifest"), cfg["splits"])
    write_once_json(output_root(cfg) / "split_manifest.json", result)
    return result


def setup_head_workflow(cfg: dict, config_path: str | Path, snapshot_path: str | Path,
                        metadata_output: str | Path | None = None) -> dict:
    """Create verified local head metadata and connect it to a local YAML.

    This is a config-only setup step: it hashes processor assets and inspects
    snapshot metadata/indexes, but does not load OpenVLA weights or use CUDA.
    """
    import yaml
    from .config import validate_config
    from .head import NORM_IMPLEMENTATION, _resolve_snapshot_text_config

    config_file = Path(config_path).expanduser().resolve()
    snapshot = Path(snapshot_path).expanduser().resolve()
    if not config_file.is_file():
        raise FileNotFoundError(f"Local experiment config not found: {config_file}")
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Local model snapshot not found: {snapshot}")
    metadata_path = (Path(metadata_output).expanduser().resolve() if metadata_output else
                     snapshot.parent / "head.json")
    if metadata_path == snapshot or snapshot in metadata_path.parents:
        raise ValueError("Head metadata must be outside the read-only model snapshot")

    model_config_path = snapshot / "config.json"
    index_path = snapshot / "model.safetensors.index.json"
    model_config = read_json(model_config_path)
    text_config, resolution = _resolve_snapshot_text_config(model_config)
    layers = text_config.get("num_hidden_layers")
    eps = text_config.get("rms_norm_eps")
    if cfg["scope"]["layer_idx"] != layers - 1:
        raise ValueError("Configured target is not the final decoder layer")
    if text_config.get("torch_dtype", model_config.get("torch_dtype")) != cfg["model"]["expected_live_hidden_dtype"]:
        raise ValueError("Snapshot text dtype differs from expected live hidden dtype")

    tensor_keys = {"norm_weight": "language_model.model.norm.weight",
                   "head_weight": "language_model.lm_head.weight"}
    weight_map = read_json(index_path).get("weight_map", {})
    missing_tensors = [value for value in tensor_keys.values() if value not in weight_map]
    if missing_tensors:
        raise ValueError(f"Snapshot index lacks required output-head tensors: {missing_tensors}")

    processor_hashes = {}
    for name in PROCESSOR_IDENTITY_FILES:
        path = snapshot / name
        processor_hashes[name] = sha256_file(path) if path.is_file() else None
    missing_processor = [name for name in REQUIRED_PROCESSOR_IDENTITY_FILES
                         if processor_hashes[name] is None]
    if all(processor_hashes[name] is None for name in TOKENIZER_IDENTITY_FILES):
        missing_processor.append("tokenizer.json or tokenizer.model")
    if missing_processor:
        raise FileNotFoundError(f"Required processor identity files missing: {missing_processor}")
    base_eval_path = Path(cfg["rollout"]["base_eval_config"]).expanduser().resolve()
    base_eval = yaml.safe_load(base_eval_path.read_text(encoding="utf-8")) or {}
    preprocessing_code = REPO_ROOT / "event_sae/openvla/eval/model.py"
    import transformers
    processor_evidence = {
        "schema_version": "processor_identity_v1",
        "model_revision": cfg["model"]["revision"],
        "code_revision": cfg["model"]["code_revision"],
        "files": processor_hashes,
        "transformers_version": transformers.__version__,
        "preprocessing": {
            "center_crop": base_eval.get("model", {}).get("center_crop"),
            "base_eval_config_sha256": sha256_file(base_eval_path),
            "eval_model_code_sha256": sha256_file(preprocessing_code),
        },
    }
    metadata = {
        "model_revision": cfg["model"]["revision"],
        "code_revision": cfg["model"]["code_revision"],
        "target_layer": cfg["scope"]["layer_idx"],
        "hidden_dtype": cfg["model"]["expected_live_hidden_dtype"],
        "norm_dtype": cfg["model"]["expected_live_hidden_dtype"],
        "head_dtype": cfg["model"]["expected_live_hidden_dtype"],
        "processor_identity": f"sha256:{fingerprint(processor_evidence)}",
        "processor_identity_evidence": processor_evidence,
        "norm_spec": {"implementation": NORM_IMPLEMENTATION, "eps": float(eps)},
        "tensor_keys": tensor_keys,
        "setup_text_config_resolution": resolution,
    }

    document = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    document.setdefault("inputs", {})["local_model_snapshot"] = str(snapshot)
    document["inputs"]["head_export_metadata"] = str(metadata_path)
    document.setdefault("scoring", {})["device"] = "cuda:0"
    validate_config(document)
    write_once_json(metadata_path, metadata)
    atomic_write_text(config_file, yaml.safe_dump(document, sort_keys=False), overwrite=True)
    return {
        "status": "configured",
        "config": str(config_file),
        "local_model_snapshot": str(snapshot),
        "head_export_metadata": str(metadata_path),
        "processor_identity": metadata["processor_identity"],
        "num_layers": layers,
        "target_layer": cfg["scope"]["layer_idx"],
        "norm_eps": float(eps),
        "model_execution": False,
    }


def export_workflow(cfg: dict) -> dict:
    from .head import export_local_output_head
    metadata = read_json(required_input(cfg, "head_export_metadata"))
    for key, expected in (("model_revision", cfg["model"]["revision"]),
                          ("code_revision", cfg["model"]["code_revision"]),
                          ("target_layer", cfg["scope"]["layer_idx"]),
                          ("hidden_dtype", cfg["model"]["expected_live_hidden_dtype"])):
        if metadata.get(key) != expected:
            raise ValueError(f"Head export metadata mismatch: {key}")
    return export_local_output_head(required_input(cfg, "local_model_snapshot"), output_root(cfg) / "head",
                                    metadata=metadata, norm_spec=metadata.get("norm_spec"),
                                    tensor_keys=metadata.get("tensor_keys"))


def _head_path(cfg: dict) -> Path:
    explicit = cfg["inputs"]["output_head_bundle"]
    return Path(explicit).expanduser().resolve() if explicit else output_root(cfg) / "head"


def numerical_identity(cfg: dict) -> dict:
    """Build current identity from real files and implementation, never old reports."""
    from .readouts import MAPPING_VERSION
    import torch
    import transformers
    from event_sae.openvla.eval.config import load_config as load_eval_config
    from dictionary_learning.trainers.batch_top_k import BatchTopKSAE
    weights, sae_config = _sae_paths(cfg)
    head_dir = _head_path(cfg)
    if head_dir.is_file():
        head_dir = head_dir.parent
    head_manifest = read_json(head_dir / "output_head_manifest.json")
    policy = load_eval_config(cfg["rollout"]["base_eval_config"])
    if policy.model.load_in_8bit or policy.model.load_in_4bit:
        raise ValueError("Quantized policy heads are outside the reference MVP")
    if policy.model.revision != cfg["model"]["revision"] or policy.model.code_revision != cfg["model"]["code_revision"]:
        raise ValueError("Base policy model identity differs from research config")
    if cfg["model"]["expected_live_hidden_dtype"] != "bfloat16":
        raise ValueError("Native OpenVLA runtime currently supports the existing BF16 protocol only")
    snapshot = cfg["inputs"]["local_model_snapshot"]
    snapshot_identity = {}
    if snapshot:
        for path in sorted(Path(snapshot).glob("*")):
            if path.is_file() and path.suffix in (".safetensors", ".json", ".py"):
                stat = path.stat()
                snapshot_identity[path.name] = ({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
                                                 if path.suffix == ".safetensors" else {"sha256": sha256_file(path)})
    for key, expected in (("model_revision", cfg["model"]["revision"]),
                          ("code_revision", cfg["model"]["code_revision"]),
                          ("target_layer", cfg["scope"]["layer_idx"])):
        if head_manifest.get(key, head_manifest.get("metadata", {}).get(key)) != expected:
            raise ValueError(f"Head identity does not match current config: {key}")
    return {
        "model_revision": cfg["model"]["revision"], "code_revision": cfg["model"]["code_revision"],
        "model_checkpoint": cfg["model"]["checkpoint"], "unnorm_key": cfg["model"]["unnorm_key"],
        "preprocessing": {"center_crop": policy.model.center_crop, "num_steps_wait": policy.env.num_steps_wait},
        "local_snapshot_identity": snapshot_identity,
        "local_snapshot_verification_level": "metadata_and_head_content_hashes_not_full_weight_rescan",
        "attention_backend": "sdpa", "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "device": cfg["scoring"]["device"], "cuda_build": str(torch.version.cuda),
        "processor_identity": head_manifest.get("processor_identity", head_manifest.get("metadata", {}).get("processor_identity")),
        "head_manifest_hash": sha256_file(head_dir / "output_head_manifest.json"),
        "head_weights_hash": sha256_file(head_dir / "output_head.safetensors"),
        "sae_checkpoint_hash": sha256_file(weights), "sae_config_hash": sha256_file(sae_config),
        "encoder_implementation_hash": sae_implementation_identity(BatchTopKSAE),
        "layer_idx": cfg["scope"]["layer_idx"], "capture_target": "post_mlp_residual",
        "mapping_version": MAPPING_VERSION,
        "generation_manifest_hash": sha256_file(required_input(cfg, "generation_manifest")),
        "hidden_dtype": cfg["model"]["expected_live_hidden_dtype"],
        "reduction_dtype": cfg["scoring"]["reduction_dtype"],
        "edit_backend": cfg["scoring"]["edit_backend"], "encoder_grouping": "original_forward",
        "arithmetic_mode": cfg["scoring"]["arithmetic_mode"],
        "pair_batch_size": cfg["scoring"]["pair_batch_size"],
        "negative_tolerance": cfg["scoring"]["negative_tolerance"],
        "alpha": cfg["scoring"]["alpha"],
        "score_definition": cfg["scoring"]["primary_metric"],
        "validation_tolerances": {key: cfg["validation"][key] for key in ("atol", "rtol", "max_argmax_mismatch")},
        "implementation_fingerprint": implementation_fingerprint(),
    }


def load_parity(cfg: dict, identity: dict) -> dict:
    from .head import validate_parity_report, HEAD_PARITY_CHECKS, EDIT_PARITY_CHECKS
    reports = {}
    for name, filename, checks in (("head", "runtime_parity.json", HEAD_PARITY_CHECKS),
                                   ("edit", "edit_parity.json", EDIT_PARITY_CHECKS)):
        path = output_root(cfg) / "validation" / filename
        if not path.is_file():
            raise ValueError(f"Runtime {name} parity not_run: {path}")
        report = read_json(path)
        validate_parity_report(report, identity, checks)
        reports[name] = report
    return reports


def prepare_workflow(cfg: dict) -> dict:
    import time
    from event_sae.openvla.activations import load_batch_topk_sae
    from .readouts import build_readout_manifest, prepare_readout_cache
    started = time.perf_counter()
    weights, _ = _sae_paths(cfg)
    dense = required_input(cfg, "dense_dir")
    generation = read_json(required_input(cfg, "generation_manifest"))
    if generation.get("layer_idx") != cfg["scope"]["layer_idx"]:
        raise ValueError("Generation manifest target layer mismatch")
    episodes = cfg["inputs"]["source_episode_manifest"]
    manifest = build_readout_manifest(dense / "activation_index.jsonl", cfg["sampling"], generation,
                                     dense_dir=dense, source_episodes=episodes)
    root = output_root(cfg)
    write_once_json(root / "sample_manifest.json", manifest)
    sae, _ = load_batch_topk_sae(weights, device=cfg["scoring"]["device"])
    sae_id = fingerprint({"weights": sha256_file(weights),
                          "encoder_implementation": sae_implementation_identity(type(sae))})
    cache = prepare_readout_cache(manifest, sae, root / "readouts", dense_dir=dense,
                                  device=cfg["scoring"]["device"], sae_id=sae_id,
                                  hidden_dtype=cfg["model"]["expected_live_hidden_dtype"],
                                  encoder_mode="original_forward", resume=cfg["output"]["resume"])
    return {"status": "prepared", "num_readouts": len(cache["records"]),
            "cache": str(root / "readouts"), "elapsed_seconds": time.perf_counter() - started,
            "grouping_check": cache["grouping_check"], "runtime_validation": "not_implied"}


def score_workflow(cfg: dict) -> dict:
    import time
    from event_sae.openvla.activations import load_batch_topk_sae
    from .head import load_output_head
    from .readouts import prepare_readout_cache
    from .sensitivity import score_features
    identity = numerical_identity(cfg)
    reports = load_parity(cfg, identity)
    root = output_root(cfg)
    sample = read_json(root / "sample_manifest.json")
    original_spec = {key: value for key, value in sample.get("sample_spec", {}).items()
                     if key != "split_manifest_hash"}
    if original_spec != cfg["sampling"]:
        raise ValueError("Current sampling differs from cached sample; prepare a new cache")
    if cfg["sampling"]["split_manifest"]:
        from .splits import validate_split_manifest
        current_split = read_json(cfg["sampling"]["split_manifest"])
        validate_split_manifest(current_split, require_confirmatory=True)
        if current_split["manifest_hash"] != sample.get("split_manifest_hash"):
            raise ValueError("Frozen split changed since readout preparation")
    if not (root / "readouts" / "manifest.json").is_file():
        raise ValueError("Prepare the readout cache before scoring")
    weights, _ = _sae_paths(cfg)
    sae, _ = load_batch_topk_sae(weights, device=cfg["scoring"]["device"])
    sae_id = fingerprint({"weights": sha256_file(weights),
                          "encoder_implementation": sae_implementation_identity(type(sae))})
    cache = prepare_readout_cache(sample, sae, root / "readouts", dense_dir=required_input(cfg, "dense_dir"),
                                  device=cfg["scoring"]["device"], sae_id=sae_id,
                                  hidden_dtype=cfg["model"]["expected_live_hidden_dtype"],
                                  encoder_mode="original_forward", resume=True)
    if cache.get("synthetic"):
        raise ValueError("The real score CLI cannot consume a synthetic cache")
    if cache["manifest"]["sae_id"] != sae_id or cache["manifest"]["sample_hash"] != sample["sample_hash"]:
        raise ValueError("Readout cache does not match current SAE/sample")
    scoring_identity = {**identity, "sample_hash": sample["sample_hash"],
                        "cache_manifest_hash": sha256_file(root / "readouts" / "manifest.json"),
                        "scoring": cfg["scoring"]}
    existing = root / "scores" / "scores.json"
    if existing.exists():
        previous = read_json(existing)
        if not cfg["output"]["resume"] or previous.get("identity") != scoring_identity:
            raise ValueError("Score resume provenance mismatch")
        return {"status": "resumed", "scores": str(existing), "num_features": len(previous["feature_scores"])}
    head = load_output_head(_head_path(cfg), device=cfg["scoring"]["device"])
    import torch
    cuda_device = head.head_weight.device if head.head_weight.device.type == "cuda" else None
    if cuda_device is not None:
        torch.cuda.reset_peak_memory_stats(cuda_device)
    started = time.perf_counter()
    result = score_features(cache, head, sae, {**cfg["scoring"], "current_identity": identity,
                                               "parity_reports": reports})
    result["identity"] = scoring_identity
    result["identity_hash"] = fingerprint(scoring_identity)
    result["scope"] = {"task_ids": sorted({row["task_id"] for row in cache["records"]}),
                       "dictionary_size": cache["dict_size"], "sae_sha256": sha256_file(weights),
                       "discovery_manifest_hash": sample["discovery_manifest_hash"],
                       "sample_manifest_hash": sample["sample_hash"],
                       "split_manifest_hash": sample.get("split_manifest_hash")}
    result["elapsed_seconds"] = time.perf_counter() - started
    result["diagnostics"]["peak_memory_allocated_bytes"] = (
        torch.cuda.max_memory_allocated(cuda_device) if cuda_device is not None else None)
    result["diagnostics"]["device"] = str(head.head_weight.device)
    write_once_json(existing, result)
    write_once_json(root / "scores" / "score_manifest.json", {
        "identity": scoring_identity, "identity_hash": result["identity_hash"],
        "scope": result["scope"], "synthetic": False,
        "elapsed_seconds": result["elapsed_seconds"], "logits_stage": "raw_head_before_processors"})
    _write_csv(root / "scores" / "feature_scores.csv", result["feature_scores"])
    _write_csv(root / "scores" / "feature_task_scores.csv", result.get("feature_task_scores", []))
    _write_csv(root / "scores" / "feature_dimension_scores.csv", result.get("feature_dimension_scores", []))
    return {"status": "scored", "scores": str(existing), "num_features": len(result["feature_scores"]),
            "elapsed_seconds": result["elapsed_seconds"], "diagnostics": result.get("diagnostics", {})}


def plan_workflow(cfg: dict) -> dict:
    from .candidates import build_evaluation_plan, load_ranking_scores
    root = output_root(cfg)
    scores_path = root / "scores" / "scores.json"
    if not scores_path.is_file() or not cfg["rollout"]["eval_manifest"]:
        return {"status": "blocked", "execute": False, "total_rollouts": None,
                "blockers": ["Need scores/scores.json and rollout.eval_manifest to compute an exact budget"]}
    scores = read_json(scores_path)
    evaluation = read_json(cfg["rollout"]["eval_manifest"])
    if scores.get("synthetic") is not False:
        raise ValueError("Real CLI planning cannot consume synthetic scores; use the synthetic test harness")
    if evaluation.get("mode", "pilot") != cfg["sampling"]["mode"]:
        raise ValueError("Evaluation manifest mode differs from config")
    if not isinstance(evaluation.get("discovery_eval_overlap"), bool):
        raise ValueError("Declare discovery_eval_overlap in the evaluation manifest")
    sample = read_json(root / "sample_manifest.json")
    discovery = sample.get("discovery_episodes", [])
    cases = evaluation.get("episodes", evaluation.get("eval_cases", evaluation.get("cases", [])))
    if discovery and cases:
        def state_key(row):
            return (row.get("suite", cfg["scope"]["suite"]), row["task_id"], row["task_episode_idx"])
        overlap = bool({state_key(row) for row in discovery} & {state_key(row) for row in cases})
        if overlap != evaluation["discovery_eval_overlap"]:
            raise ValueError("Declared discovery/evaluation overlap contradicts original trial IDs")
    if cfg["sampling"]["mode"] == "confirmatory":
        from .splits import validate_selection_split
        split_path = cfg["sampling"]["split_manifest"]
        if not split_path:
            raise ValueError("Confirmatory plan requires its frozen split")
        split = read_json(split_path)
        validate_selection_split(split, cases, stage="evaluation")
        if evaluation.get("split_manifest_hash") != split["manifest_hash"]:
            raise ValueError("Evaluation plan split hash mismatch")
    comparison_methods = [x for x in cfg["selection"]["methods"]
                          if x in ("event_aligned", "window_mean", "task_mean")]
    comparison = None
    if comparison_methods:
        # Full matrices are loaded only when a concrete panel can be built.
        comparison = load_ranking_scores(required_input(cfg, "event_scores_path"), comparison_methods,
                                         expected_scope=scores.get("scope"))
    budget = {**cfg["rollout"], "max_unique_features": cfg["selection"]["max_unique_features"],
              "identity_conditions": 1}
    plan = build_evaluation_plan(scores, comparison, evaluation, budget, selection=cfg["selection"])
    plan["experiment_config_hash"] = fingerprint(cfg)
    plan["score_artifact_hash"] = sha256_file(scores_path)
    plan["eval_manifest_hash"] = sha256_file(cfg["rollout"]["eval_manifest"])
    plan["implementation_fingerprint"] = implementation_fingerprint()
    plan["base_eval_config_hash"] = sha256_file(cfg["rollout"]["base_eval_config"])
    plan["execute"] = False
    plan.pop("plan_hash", None)
    plan["plan_hash"] = fingerprint(plan)
    if plan["status"] == "blocked_budget":
        write_once_json(root / "plan" / f"blocked-{plan['plan_hash'][:16]}.json", plan)
        return plan
    write_once_json(root / "plan" / "rollout_plan.json", plan)
    write_once_json(root / "plan" / "eval_manifest.json", evaluation)
    candidates = plan.get("candidates", [])
    candidate_path = root / "plan" / "candidates.jsonl"
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in candidates)
    if not candidate_path.exists():
        atomic_write_text(candidate_path, text)
    elif candidate_path.read_text(encoding="utf-8") != text:
        raise ValueError("Existing candidate panel differs; do not replace a frozen panel")
    return plan


def analyze_workflow(cfg: dict) -> dict:
    from .results import assemble_paired_effects, analyze_prediction
    root = output_root(cfg)
    scores = read_json(root / "scores" / "scores.json")
    plan = read_json(root / "plan" / "rollout_plan.json")
    if plan.get("plan_hash") != fingerprint({key: value for key, value in plan.items() if key != "plan_hash"}):
        raise ValueError("Analysis refused: frozen plan hash is invalid")
    if plan.get("score_artifact_hash") != sha256_file(root / "scores" / "scores.json"):
        raise ValueError("Analysis refused: score artifact changed after selection")
    if plan.get("experiment_config_hash") != fingerprint(cfg):
        raise ValueError("Analysis refused: config changed after selection")
    raw = read_json(root / "runs" / "raw" / "result.json")
    features = {int(row["feature_id"]): read_json(root / "runs" / f"feature-{row['feature_id']}" / "result.json")
                for row in plan["candidates"]}
    from .runtime import validate_condition_result, verify_identity_actions
    conditions = {row["condition_id"]: row for row in plan["conditions"]}
    for name, result in [("raw", raw), *[(f"feature-{key}", value) for key, value in features.items()]]:
        validate_condition_result(result, conditions[name], plan["eval_cases"], raw["protocol_id"],
                                  raw["head_sae_identity"], raw["code"]["commit"])
    for condition in plan["conditions"]:
        if condition["mode"] == "identity":
            identity_result = read_json(root / "runs" / condition["condition_id"] / "result.json")
            validate_condition_result(identity_result, condition, plan["eval_cases"], raw["protocol_id"],
                                      raw["head_sae_identity"], raw["code"]["commit"])
            verify_identity_actions(raw, identity_result)
    score_identity = scores.get("identity", {})
    for result in [raw, *features.values()]:
        identity = result.get("head_sae_identity", {})
        if not identity or any(score_identity.get(key) != value for key, value in identity.items()):
            raise ValueError("Scoring and rollout numerical identities differ")
    protocol = {**cfg["analysis"], "selection_eval_overlap": plan.get("selection_eval_overlap"),
                "evaluation_labels_previously_used": plan.get("evaluation_labels_previously_used"),
                "protocol_id": raw.get("protocol_id"), "alpha": cfg["scoring"]["alpha"],
                "synthetic": False, "mode": plan["mode"], "suite": cfg["scope"]["suite"],
                "eval_cases": plan["eval_cases"], "expected_feature_ids": plan["feature_ids"],
                "model_checkpoint": cfg["model"]["checkpoint"], "model_revision": cfg["model"]["revision"],
                "model_code_revision": cfg["model"]["code_revision"],
                "code_revision": raw["code"]["commit"], "sae_sha256": score_identity["sae_checkpoint_hash"],
                "layer_idx": cfg["scope"]["layer_idx"], "hook_start_step": cfg["rollout"]["hook_start_step"]}
    paired = assemble_paired_effects(raw, features, protocol)
    analysis = analyze_prediction(scores, paired, {**cfg["analysis"], "plan": plan,
                                                  "top_k": cfg["selection"]["top_k"]})
    write_once_json(root / "analysis" / "paired_effects.json", paired)
    write_once_json(root / "analysis" / "analysis.json", analysis)
    _write_csv(root / "analysis" / "paired_effects.csv", analysis.get("feature_effects", []))
    _write_csv(root / "analysis" / "predictor_comparison.csv", analysis.get("predictor_comparison", []))
    write_once_json(root / "analysis" / "uncertainty.json", analysis["uncertainty"])
    report = ("# Output-head sensitivity results\n\n"
              "These are fixed-prefix, raw full-vocabulary scores for one frozen OpenVLA/SAE. "
              "Closed-loop outcomes and signed drops are recorded separately.\n\n"
              f"Sampling mode: {cfg['sampling']['mode']}. Synthetic: {paired.get('synthetic')}.\n\n"
              f"Plan: `{plan['plan_hash']}`. Measured features: {len(features)}. "
              f"Approved total rollouts (including raw/identity): {plan['total_rollouts']}.\n\n"
              f"Discovery/evaluation overlap: {plan.get('selection_eval_overlap')}. "
              f"Previously used evaluation labels: {plan.get('evaluation_labels_previously_used')}.\n\n"
              "See paired_effects.json and analysis.json for measured pairs, missing/undefined "
              "statistics, shared-bootstrap uncertainty and scope. No unmeasured global recall is claimed.\n")
    report += "\n## Paired signed effects\n\n| Feature | Pairs | Drop | CI low | CI high | Status |\n|---|---:|---:|---:|---:|---|\n"
    for row in analysis["feature_effects"]:
        report += (f"| {row['feature_id']} | {row['num_valid_pairs']} | {row['drop']} | "
                   f"{row.get('drop_ci_low')} | {row.get('drop_ci_high')} | {row.get('bootstrap_status')} |\n")
    report += "\n## Predictor comparison\n\n| Predictor | Population | Spearman | Top-K coverage | Top-K mean drop |\n|---|---|---:|---:|---:|\n"
    for row in analysis["predictor_comparison"]:
        report += (f"| {row['predictor']} | {row['population_scope']} | {row['spearman']} | "
                   f"{row['topk_coverage']} | {row['topk_mean_drop']} |\n")
    report += ("\n## Scope and costs\n\n"
               f"Score scope: `{json.dumps(scores['scope'], sort_keys=True)}`. "
               f"Measured scoring seconds: {scores.get('elapsed_seconds')}. "
               f"Scoring diagnostics: `{json.dumps(scores.get('diagnostics', {}), sort_keys=True)}`.\n\n"
               "Preparation/I/O timing not present in these outcome artifacts is unavailable, not zero. "
               "Runtime parity certificates are in validation/. Exact action identity was rechecked. "
               "New rollouts use task/trial-derived episode RNG; inherited baseline runs are not silently reused. "
               "Negative drops indicate improvement; null correlation is undefined. Bootstrap is paired "
               "within the fixed tasks and conditional on this evaluated feature panel. The existing SAE "
               "was not retrained on a held-out-only dataset.\n")
    path = root / "analysis" / "report.md"
    if not path.exists():
        atomic_write_text(path, report)
    return analysis
