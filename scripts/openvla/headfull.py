#!/usr/bin/env python3
"""Pinned full10 download, merge and followup setup; never executes a policy."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
EVENT_REPO = "jiyeony/event-sae-libero-spatial-reproduction"
EVENT_REVISION = "f7eb3c8b6e7975481db7f50c82d06384d03c2e8a"
INDEX_SHA = "d6da7ab63bfd945e00a507584ba61beee0b2a9f4cb3b21de25bf616f5662ccea"
MODEL_REPO = "openvla/openvla-7b-finetuned-libero-spatial"
MODEL_REVISION = "962318cec55ac10993ff0f5f43eda9a270b4c873"
CODE_REVISION = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"
SAE_SHA = "18443083d320d2ad3c607431f307697639966ad92075bdfe031398557f3ba9d6"
PAIRS = ("0-1", "2-3", "4-5", "6-7", "8-9")


def paths(workspace: Path) -> dict[str, Path]:
    workspace = workspace.expanduser().resolve()
    inputs = workspace / "event-sae-spatial-inputs"
    return {"inputs": inputs, "merged": inputs / "merged",
            "sae": inputs / "checkpoint/trainer_0/ae.pt",
            "events": workspace / "head-inputs/event-public",
            "model": workspace / "head-inputs/openvla-spatial",
            "metadata": workspace / "head-inputs/full10",
            "results": workspace / "event-sae-head-results/full10-followup-v1"}


def write_once_json(path: Path, payload: dict) -> None:
    """Resume an identical artifact without replacing its frozen bytes."""
    from event_sae.research.output_head.provenance import atomic_write_json, fingerprint, read_json
    if path.exists():
        if fingerprint(read_json(path)) != fingerprint(payload):
            raise ValueError(f"Existing immutable setup artifact differs: {path}")
        return
    atomic_write_json(path, payload)


def pin_nested_code_reference(snapshot: Path) -> None:
    """Pin the nested AutoImageProcessor cache lookup in a dedicated HF_HOME.

    The nested loader can omit code_revision. Never replace a conflicting cached
    main reference; this alias points only to the explicitly downloaded revision.
    """
    if snapshot.name != CODE_REVISION or snapshot.parent.name != "snapshots":
        raise ValueError("Unexpected pinned custom-code cache path")
    from event_sae.research.output_head.provenance import atomic_write_text
    reference = snapshot.parent.parent / "refs/main"
    if reference.exists():
        if reference.read_text().strip() != CODE_REVISION:
            raise ValueError("Conflicting custom-code refs/main; use a fresh dedicated HF_HOME")
    else:
        atomic_write_text(reference, CODE_REVISION)


def download(workspace: Path, *, execute: bool) -> dict:
    p = paths(workspace)
    summary = {"status": "download_plan", "execute": execute,
               "source_datasets": 5, "expected_dense_shards": 369,
               "source_size_gib_approx": 281, "gpu_execution": False,
               "event_revision": EVENT_REVISION, "model_revision": MODEL_REVISION,
               "code_revision": CODE_REVISION, "paths": {k: str(v) for k, v in p.items()}}
    if not execute:
        return summary
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    os.environ.setdefault("HF_HOME", str(workspace.resolve() / "cache/head-full10-hf"))
    subprocess.run([sys.executable, str(ROOT / "scripts/openvla/download_libero_spatial_reproduction_inputs.py"),
                    "--output-root", str(p["inputs"])], check=True, cwd=ROOT)
    from huggingface_hub import snapshot_download
    snapshot_download(EVENT_REPO, repo_type="dataset", revision=EVENT_REVISION,
                      local_dir=p["events"], allow_patterns=["events/event_features.jsonl",
                      "clusters/cluster_assignments.jsonl", "clusters/clusters.jsonl",
                      "topk/manifest.json", "topk/shard_*.pt"])
    snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_dir=p["model"])
    code = snapshot_download("openvla/openvla-7b", revision=CODE_REVISION,
                             allow_patterns=["configuration_prismatic.py", "modeling_prismatic.py",
                                             "processing_prismatic.py"])
    pin_nested_code_reference(Path(code))
    summary.update(status="downloaded", hf_home=os.environ["HF_HOME"])
    return summary


def check_merged(p: dict[str, Path]) -> dict:
    from event_sae.research.output_head.provenance import read_json, sha256_file
    manifest = read_json(p["merged"] / "merge_manifest.json")
    expected_inputs = [str((p["inputs"] / f"pairs/tasks-{pair}").resolve()) for pair in PAIRS]
    if (manifest.get("num_episodes") != 500 or manifest.get("num_activation_shards") != 369 or
            manifest.get("input_dirs") != expected_inputs or manifest.get("link_mode") != "symlink"):
        raise ValueError("Merged source manifest differs from pinned full10 inputs")
    index = p["merged"] / "sae_activations/post_mlp_residual/activation_index.jsonl"
    if sha256_file(index) != INDEX_SHA:
        raise ValueError("Merged activation index differs from public Event TopK index")
    return manifest


def merge(workspace: Path) -> dict:
    from event_sae.openvla.merge_runs import MergeConfig, merge_openvla_runs
    p = paths(workspace)
    if not p["merged"].exists():
        merge_openvla_runs(MergeConfig(
            input_dirs=tuple(str(p["inputs"] / f"pairs/tasks-{pair}") for pair in PAIRS),
            output_dir=str(p["merged"]), trials_per_task=50, expected_tasks=10,
            expected_shards=369, link_mode="symlink"))
    manifest = check_merged(p)
    return {"status": "merged_and_verified", "episodes": manifest["num_episodes"],
            "dense_shards": manifest["num_activation_shards"], "merged": str(p["merged"])}


def setup(workspace: Path, config_path: Path, *, allow_libero: bool) -> dict:
    if not allow_libero:
        raise ValueError("Use --allow-libero to read current LIBERO states (no policy execution)")
    import yaml
    from event_sae.research.output_head.config import load_config, validate_config
    from event_sae.research.output_head.provenance import read_json, sha256_file, atomic_write_text
    from event_sae.research.output_head.state_registry import build_state_registry, build_generation_manifest
    from event_sae.research.output_head.splits import build_followup_split, validate_split_manifest

    p = paths(workspace)
    config_path = config_path.expanduser().resolve()
    check_merged(p)
    if sha256_file(p["sae"]) != SAE_SHA:
        raise ValueError("Pinned SAE checksum mismatch")
    public = read_json(p["events"] / "topk/manifest.json")
    if public.get("activation_index_sha256") != INDEX_SHA or public.get("sae_sha256") != SAE_SHA:
        raise ValueError("Public TopK source identity mismatch")
    cfg = load_config(ROOT / "configs/research/openvla/output_head_sensitivity_full10.yaml")
    metadata = p["metadata"]
    registry_path, generation_path = metadata / "source_episodes.json", metadata / "generation_manifest.json"
    split_path, eval_path = metadata / "split.json", metadata / "eval_manifest.json"
    registry = build_state_registry(p["merged"], expected_shards=369)
    generation = build_generation_manifest(registry)
    split = build_followup_split(registry, cfg["splits"])
    if split_path.exists():
        old = read_json(split_path)
        validate_split_manifest(old, require_followup=True)
        ignore = {"frozen_at_utc", "manifest_hash"}
        if {k: v for k, v in split.items() if k not in ignore} != {k: v for k, v in old.items() if k not in ignore}:
            raise ValueError("Existing frozen split differs; do not reselect after viewing results")
        split = old
    elif p["results"].exists() and any(p["results"].iterdir()):
        raise ValueError("Cannot freeze a new split over an existing experiment")
    evaluation = {key: split[key] for key in (
        "mode", "frozen_before_pilot", "frozen_at_utc", "frozen_before_followup_evaluation",
        "historical_state_identity_verified", "evaluation_labels_previously_used")}
    evaluation.update(split_manifest_hash=split["manifest_hash"], discovery_eval_overlap=False,
                      cases=split["splits"]["evaluation"])
    cfg["inputs"].update(dense_dir=str(p["merged"] / "sae_activations/post_mlp_residual"),
        source_episode_manifest=str(registry_path), sae_checkpoint=str(p["sae"]),
        generation_manifest=str(generation_path), local_model_snapshot=str(p["model"]),
        head_export_metadata=str(metadata / "head.json"))
    cfg["sampling"]["split_manifest"] = str(split_path)
    cfg["rollout"]["eval_manifest"] = str(eval_path)
    cfg["output"]["root_dir"] = str(p["results"])
    validate_config(cfg)
    if config_path.exists():
        existing = load_config(config_path)
        # Only these paths are filled by subsequent explicit pipeline steps.
        for key in ("event_scores_path", "output_head_bundle", "runtime_inputs"):
            cfg["inputs"][key] = existing["inputs"][key]
        if existing != cfg:
            raise ValueError("Existing local config differs; refusing to overwrite experiment settings")
    for path, payload in ((registry_path, registry), (generation_path, generation),
                          (split_path, split), (eval_path, evaluation)):
        write_once_json(path, payload)
    if not config_path.exists():
        atomic_write_text(config_path, yaml.safe_dump(cfg, sort_keys=False))
    return {"status": "configured", "config": str(config_path), "mode": "followup",
            "split": str(split_path), "split_manifest_hash": split["manifest_hash"],
            "discovery_episodes": len(split["splits"]["discovery"]),
            "validation_episodes": len(split["splits"]["validation"]),
            "evaluation_cases": len(evaluation["cases"]), "max_rollouts": 4000,
            "historical_state_identity_verified": registry["historical_state_identity_verified"],
            "historical_generation_flags_verified": False, "model_execution": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "merge", "setup"))
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/local/head-full10.yaml")
    parser.add_argument("--execute", action="store_true", help="Authorize pinned downloads only")
    parser.add_argument("--allow-libero", action="store_true", help="Inspect current benchmark states only")
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            result = download(args.workspace, execute=args.execute)
        elif args.command == "merge":
            result = merge(args.workspace)
        else:
            result = setup(args.workspace, args.config, allow_libero=args.allow_libero)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, OSError, ImportError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
