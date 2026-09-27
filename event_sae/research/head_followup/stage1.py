"""Stage 1 input restoration and token mapping; never runs a policy or downloads."""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import shutil
import tarfile

from event_sae.research.output_head.provenance import (
    assert_safe_output, atomic_write_json, fingerprint, read_json, sha256_file,
)

ARCHIVE_SHA = "07eeb5c182ee4e195eb526362db9232ae1717de4c94b9b6da2fd62ef288de8bc"
MODEL_REV = "962318cec55ac10993ff0f5f43eda9a270b4c873"
CODE_REV = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"
PREFIX = "results/full10-followup-v1/"
SOURCE_HASHES = {
    "config.json": "a2de1404c39c5608cc11917904ec610bf55a35b670b02db2ad71efb04a80b39e",
    "tokenizer.json": "8f5e2869e1807b8bb3c7717a294539c37e9de5d728ad60e31ef57e83cc5ea527",
    "modeling_prismatic.py": "9ce241c5ca09a4bed73654d0ca509baaff0fc5887d0bdd62e360ca8254f7794a",
}


def restore_stage1(archive, manifest_path, output):
    """Copy only allowlisted regular files from the pinned local backup, checking bytes.

    Original paths *inside* JSON/YAML stay historical. This is not a runnable config
    rewrite and never deserializes a tensor/pickle or treats old parity as current.
    """
    archive, manifest_path = Path(archive), Path(manifest_path)
    output = assert_safe_output(output, [archive, manifest_path])
    if output.exists():
        raise ValueError("restore needs a new output directory; existing files are never overwritten")
    manifest = read_json(manifest_path)
    if (manifest["archive"]["sha256"] != ARCHIVE_SHA or sha256_file(archive) != ARCHIVE_SHA
            or archive.stat().st_size != manifest["archive"]["size_bytes"]):
        raise ValueError("not the pinned full10 archive")
    allowed = ("head/", "readouts/", "validation/")
    exact = {"sample_manifest.json", "scores/scores.json", "scores/score_manifest.json",
             "analysis/paired_effects.json", "plan/rollout_plan.json"}
    wanted = {name: value for name, value in manifest["files"].items()
              if (name.startswith(PREFIX) and (name[len(PREFIX):] in exact
                  or name[len(PREFIX):].startswith(allowed)))
              or name.startswith("inputs/full10/") or name == "config/head-full10.yaml"}
    for name in wanted:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or str(path) != name:
            raise ValueError("unsafe manifest path")
    required = {PREFIX + name for name in ("head/output_head_manifest.json", "readouts/manifest.json",
                "scores/scores.json", "analysis/paired_effects.json", "plan/rollout_plan.json")}
    if not required.issubset(wanted):
        raise ValueError("incomplete Stage 1 backup")
    total = sum(value["size_bytes"] for value in wanted.values())
    if shutil.disk_usage(output.parent).free < total + 1024**3:
        raise ValueError("insufficient space for selective restore plus 1 GiB reserve")
    output.mkdir()
    copied = {}
    with tarfile.open(archive, "r|gz") as tf:
        for member in tf:
            if member.name not in wanted:
                continue
            expected = wanted[member.name]
            if member.name in copied or not member.isfile() or member.size != expected["size_bytes"]:
                raise ValueError("duplicate/nonregular/wrong-size archive member")
            target = output / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with tf.extractfile(member) as src, target.open("xb") as dst:
                for block in iter(lambda: src.read(8 * 1024**2), b""):
                    digest.update(block)
                    dst.write(block)
            if digest.hexdigest() != expected["sha256"]:
                raise ValueError("archive member digest mismatch: " + member.name)
            copied[member.name] = expected
    if set(copied) != set(wanted):
        raise ValueError("missing archive members; partial restore is not ready")
    receipt = {"schema_version": "head_stage1_restore_v1", "archive_sha256": ARCHIVE_SHA,
               "files": copied, "restored_bytes": total, "new_rollouts": 0,
               "historical_paths_rewritten": False, "old_runtime_parity_is_current": False}
    atomic_write_json(output / "stage1_restore.json", receipt)
    return {"status": "restored", "output": str(output), "files": len(copied),
            "restored_bytes": total, "new_rollouts": 0}


def build_action_mapping(source_dir, head_manifest):
    """Pinned OpenVLA canonical 256-token encoder support, not clipped decode domain.

    n_action_bins counts edges (255 centers). At the upper boundary the encoder
    emits bin 256; bins 255 and 256 both decode to center 254. Preserve both IDs
    for this token-space comparator; do not call it KL over physical actions.
    """
    source_dir = Path(source_dir)
    evidence = {name: sha256_file(source_dir / name) for name in SOURCE_HASHES}
    if evidence != SOURCE_HASHES:
        raise ValueError("mapping sources differ from independently checked pinned OpenVLA files")
    cfg, tokenizer = read_json(source_dir / "config.json"), read_json(source_dir / "tokenizer.json")
    vocab = tokenizer["model"]["vocab"]
    base = len(vocab)
    head_size, padding, bins = cfg["text_config"]["vocab_size"], cfg["pad_to_multiple_of"], cfg["n_action_bins"]
    if (sorted(vocab.values()) != list(range(base)) or base != head_size - padding
            or head_manifest.get("model_revision") != MODEL_REV or head_manifest.get("code_revision") != CODE_REV
            or head_manifest.get("vocab_size") != head_size
            or head_manifest.get("source_hashes", {}).get("config.json") != evidence["config.json"]
            or head_manifest.get("processor_identity_evidence", {}).get("files", {}).get("tokenizer.json") != evidence["tokenizer.json"]):
        raise ValueError("tokenizer/model/head provenance or vocabulary mismatch")
    result = {"schema_version": "head_action_mapping_v1", "verified": True,
              "model_revision": MODEL_REV, "code_revision": CODE_REV,
              "head_weights_sha256": head_manifest["weights_sha256"],
              "tokenizer_sha256": evidence["tokenizer.json"], "evidence": evidence,
              "head_vocab_size": head_size, "base_vocab_size": base, "n_action_bins": bins,
              "action_token_ids": list(range(base - bins, base)),
              "definition": "canonical_encoder_token_support_256_including_clipped_endpoint_alias",
              "not_claimed": "physical-action KL; all vocabulary IDs can be clipped by native decode",
              "encoder_rule": "token_id = base_vocab_size - digitize(clip(action,-1,1),linspace(-1,1,n_action_bins))",
              "source": {"model": f"openvla/openvla-7b-finetuned-libero-spatial@{MODEL_REV}",
                         "code": f"openvla/openvla-7b@{CODE_REV}"}}
    result["mapping_hash"] = fingerprint(result)
    return result


def validate_mapping(mapping, head_manifest):
    if (mapping.get("schema_version") != "head_action_mapping_v1"
            or mapping.get("mapping_hash") != fingerprint({k: v for k, v in mapping.items() if k != "mapping_hash"})
            or mapping.get("evidence") != SOURCE_HASHES
            or mapping.get("action_token_ids") != list(range(31744, 32000))
            or mapping.get("head_weights_sha256") != head_manifest.get("weights_sha256")
            or mapping.get("model_revision") != head_manifest.get("model_revision")
            or mapping.get("code_revision") != head_manifest.get("code_revision")):
        raise ValueError("unverified/stale action mapping")


def stage1_audit(restored, sae=None, dense=None):
    root = Path(restored)
    receipt = read_json(root / "stage1_restore.json")
    if receipt.get("archive_sha256") != ARCHIVE_SHA:
        raise ValueError("unexpected restored archive")
    for name, expected in receipt["files"].items():
        file = root / name
        if (file.resolve().is_relative_to(root.resolve()) is False
                or sha256_file(file) != expected["sha256"]):
            raise ValueError("restored input hash mismatch")
    head = read_json(root / PREFIX / "head/output_head_manifest.json")
    readouts = read_json(root / PREFIX / "readouts/manifest.json")
    sae_present = bool(sae and Path(sae).is_file())
    dense_files = list(Path(dense).rglob("layer_31_shard_*.pt")) if dense else []
    return {"schema_version": "head_stage1_audit_v1", "status": "audited_not_scored", "new_rollouts": 0,
            "restored_files_verified": len(receipt["files"]), "head_vocab_size": head["vocab_size"],
            "head_readouts": readouts["num_readouts"], "sae_checkpoint_present": sae_present,
            "dense_shards_present": len(dense_files), "free_disk_gib": shutil.disk_usage(root).free / 1024**3,
            "full_source_size_gib_approx": 281,
            "remaining": ["restore exact SAE and current-runtime numerical validation for action scoring",
                          "restore raw dense activations and re-encode full discovery timelines",
                          "independent generality labels and classifier calibration",
                          "rank/coverage comparison; missing behavior outcomes remain unavailable"]}


def setup_stage1(restored, output, *, sae, snapshot, dense=None, device="cuda:0"):
    """Write separate configs; never overwrite old configs or infer missing raw data.

    ④ needs no dense tensors. ⑤ config is produced only when a merged dense index
    is present. Numerical validation uses already saved preprocessed inputs, no
    simulator or new trajectory is needed.
    """
    import copy
    import yaml
    from event_sae.research.output_head.config import validate_config
    from event_sae.research.output_head.provenance import atomic_write_text

    root, sae, snapshot = Path(restored).resolve(), Path(sae).resolve(), Path(snapshot).resolve()
    output = assert_safe_output(output, [root, sae, snapshot, dense])
    if output.exists():
        raise ValueError("setup requires a new output directory")
    audit = stage1_audit(root, sae, dense)
    expected_sae = "18443083d320d2ad3c607431f307697639966ad92075bdfe031398557f3ba9d6"
    if not audit["sae_checkpoint_present"] or sha256_file(sae) != expected_sae or not (sae.parent / "config.json").is_file():
        raise ValueError("exact frozen SAE weights and adjacent trainer config.json required")
    if sha256_file(snapshot / "config.json") != SOURCE_HASHES["config.json"]:
        raise ValueError("wrong local model snapshot config")
    if not (snapshot / "model.safetensors.index.json").is_file() or not list(snapshot.glob("model-*.safetensors")):
        raise ValueError("restore model weights for current numerical validation (not new rollouts)")
    if dense and not (Path(dense) / "activation_index.jsonl").is_file():
        raise ValueError("merged dense activation_index.jsonl required for temporal setup")
    cfg = yaml.safe_load((root / "config/head-full10.yaml").read_text())
    reference = root / PREFIX
    inputs = root / "inputs/full10"
    for key, name in (("source_episode_manifest", "source_episodes.json"), ("event_scores_path", "discovery_event_scores.pt"),
                      ("generation_manifest", "generation_manifest.json"), ("head_export_metadata", "head.json"),
                      ("runtime_inputs", "runtime.pt")):
        cfg["inputs"][key] = str(inputs / name)
    cfg["inputs"].update(dense_dir=str(Path(dense).resolve()) if dense else None, sae_checkpoint=str(sae),
                          local_model_snapshot=str(snapshot), output_head_bundle=str(reference / "head"))
    cfg["sampling"]["split_manifest"] = str(inputs / "split.json")
    cfg["rollout"]["eval_manifest"] = str(inputs / "eval_manifest.json")
    cfg["rollout"]["base_eval_config"] = str(Path(__file__).resolve().parents[3] /
        "configs/reproduction/openvla/libero_spatial_intervention_layer31.yaml")
    cfg["scoring"]["device"] = device
    cfg["output"]["root_dir"] = str(output / "runtime")
    validate_config(cfg)
    atomic_write_text(output / "action-runtime.yaml", yaml.safe_dump(cfg, sort_keys=False))
    temporal_path = None
    if dense:
        temporal = copy.deepcopy(cfg)
        # Temporal CLI reads the reference sample/cache, and writes ONLY --output.
        temporal["output"]["root_dir"] = str(reference)
        temporal["inputs"]["output_head_bundle"] = None
        validate_config(temporal)
        temporal_path = output / "temporal-input.yaml"
        atomic_write_text(temporal_path, yaml.safe_dump(temporal, sort_keys=False))
    receipt = {"status": "configured_not_executed", "new_rollouts": 0, "reference_result": str(reference),
               "action_config": str(output / "action-runtime.yaml"),
               "temporal_config": str(temporal_path) if temporal_path else None,
               "runtime_validation": "must_run_on_current_device", "source_archive_sha256": ARCHIVE_SHA}
    atomic_write_json(output / "setup.json", receipt)
    return receipt
