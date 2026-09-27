#!/usr/bin/env python3
"""Opt-in offline follow-up tools; no rollout runner or automatic downloads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from event_sae.research.output_head.provenance import assert_safe_output, atomic_write_json, atomic_write_text, read_json, sha256_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    design = commands.add_parser("design", help="list conditions and cost, never executes")
    design.add_argument("--protocol", default="configs/research/openvla/head_followup_all5.json")
    restore = commands.add_parser("stage1-restore", help="verify and selectively restore existing local backup")
    restore.add_argument("--archive", required=True)
    restore.add_argument("--manifest", required=True)
    restore.add_argument("--output", required=True)
    audit = commands.add_parser("stage1-audit")
    audit.add_argument("--restored", required=True)
    audit.add_argument("--sae")
    audit.add_argument("--dense")
    audit.add_argument("--output", required=True)
    setup = commands.add_parser("stage1-setup", help="separate config paths, no execution/download")
    setup.add_argument("--restored", required=True)
    setup.add_argument("--output", required=True)
    setup.add_argument("--sae", required=True)
    setup.add_argument("--snapshot", required=True)
    setup.add_argument("--dense")
    setup.add_argument("--device", default="cuda:0")
    mapping = commands.add_parser("action-map", help="verify pinned model/tokenizer/code; no inference")
    mapping.add_argument("--source", required=True)
    mapping.add_argument("--head-manifest", required=True)
    mapping.add_argument("--output", required=True)
    action = commands.add_parser("action-score", help="new scores on frozen cache, fresh runtime parity required")
    action.add_argument("--config", required=True, help="separate runtime validation config, not frozen result config")
    action.add_argument("--reference-result", required=True)
    action.add_argument("--mapping", required=True)
    action.add_argument("--output", required=True)
    action.add_argument("--max-pairs", type=int, default=800000)
    action.add_argument("--execute", action="store_true")
    comparison = commands.add_parser("compare", help="join by feature ID; no new outcomes")
    comparison.add_argument("--reference-result", required=True)
    comparison.add_argument("--action-scores")
    comparison.add_argument("--drvla-scores")
    comparison.add_argument("--output", required=True, help="new output directory")
    labels = commands.add_parser("label-packet", help="blind discovery traces, not fabricated labels")
    labels.add_argument("--statistics", required=True)
    labels.add_argument("--cache", required=True)
    labels.add_argument("--output", required=True)
    labels.add_argument("--count", type=int, default=60)
    labels.add_argument("--protocol", default="configs/research/openvla/drvla_comparison_protocol_v1.json")
    temporal = commands.add_parser("temporal", help="re-encode saved dense data, not recollect or train")
    temporal.add_argument("--config", required=True, help="original full10 YAML, read only")
    temporal.add_argument("--output", required=True, help="new directory outside old inputs/results")
    temporal.add_argument("--execute", action="store_true")
    temporal.add_argument("--max-readouts", type=int, default=500000)
    temporal.add_argument("--device", default="cpu")
    temporal.add_argument("--protocol", default="configs/research/openvla/drvla_comparison_protocol_v1.json")
    stats = commands.add_parser("temporal-stats")
    stats.add_argument("--cache", required=True)
    stats.add_argument("--output", required=True)
    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("--statistics", required=True)
    calibrate.add_argument("--labels", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--protocol", default="configs/research/openvla/drvla_comparison_protocol_v1.json")
    args = parser.parse_args(argv)
    try:
        if args.command == "design":
            from event_sae.research.head_followup.design import design_summary
            result = design_summary(read_json(args.protocol))
        elif args.command == "stage1-restore":
            from event_sae.research.head_followup.stage1 import restore_stage1
            result = restore_stage1(args.archive, args.manifest, args.output)
        elif args.command == "stage1-audit":
            from event_sae.research.head_followup.stage1 import stage1_audit
            output = assert_safe_output(args.output, [args.restored, args.sae, args.dense])
            result = stage1_audit(args.restored, args.sae, args.dense)
            atomic_write_json(output, result)
        elif args.command == "stage1-setup":
            from event_sae.research.head_followup.stage1 import setup_stage1
            result = setup_stage1(args.restored, args.output, sae=args.sae, snapshot=args.snapshot,
                                  dense=args.dense, device=args.device)
        elif args.command == "action-map":
            from event_sae.research.head_followup.stage1 import build_action_mapping
            output = assert_safe_output(args.output, [args.source, args.head_manifest])
            result = build_action_mapping(args.source, read_json(args.head_manifest))
            atomic_write_json(output, result)
            ids = result["action_token_ids"]
            result = {"status": "verified", "output": str(output), "token_count": len(ids),
                      "first_token_id": ids[0], "last_token_id": ids[-1], "model_execution": False}
        elif args.command == "action-score":
            if not args.execute:
                result = {"status": "not_executed", "new_rollouts": 0,
                          "next": "restore inputs and obtain current runtime parity in separate config.output; then --execute"}
            else:
                from event_sae.research.head_followup.action_scores import action_workflow
                result = action_workflow(args.config, args.reference_result, args.mapping, args.output,
                    max_pairs=args.max_pairs, progress=lambda p: print(json.dumps(p), file=sys.stderr, flush=True))
        elif args.command == "compare":
            from event_sae.research.head_followup.comparison import compare_scores, comparison_markdown
            root = Path(args.reference_result)
            paths = {"old": root / "scores/scores.json", "plan": root / "plan/rollout_plan.json",
                     "paired": root / "analysis/paired_effects.json"}
            if args.action_scores:
                paths["action"] = Path(args.action_scores)
            if args.drvla_scores:
                paths["drvla"] = Path(args.drvla_scores)
            output = assert_safe_output(args.output, [root, *paths.values()])
            if output.exists():
                raise ValueError("comparison requires a new output directory")
            data = {key: read_json(path) for key, path in paths.items()}
            if sha256_file(paths["old"]) != data["plan"]["score_artifact_hash"]:
                raise ValueError("frozen score artifact differs from plan")
            result = compare_scores(**data)
            result["source_sha256"] = {key: sha256_file(path) for key, path in paths.items()}
            atomic_write_json(output / "comparison.json", result)
            atomic_write_text(output / "report.md", comparison_markdown(result))
            result = {"status": "compared", "output": str(output), "new_rollouts": 0,
                      "action_scores_present": result["action_scores_present"], "drvla_scores_present": result["drvla_scores_present"]}
        elif args.command == "label-packet":
            from event_sae.research.head_followup.temporal import load_temporal_cache
            from event_sae.research.head_followup.labeling import label_packet
            protocol, statistics = read_json(args.protocol), read_json(args.statistics)
            frozen = protocol["frozen_existing_experiment"]
            if any(statistics.get("identity", {}).get(key) != frozen[key] for key in ("sae_sha256", "split_manifest_hash")):
                raise ValueError("statistics do not match frozen SAE/split")
            output = assert_safe_output(args.output, [args.cache, args.statistics, args.protocol])
            packet = label_packet(statistics, load_temporal_cache(args.cache),
                excluded_feature_ids=protocol["calibrated_comparator"]["previously_measured_features"],
                statistics_sha256=sha256_file(args.statistics), count=args.count)
            atomic_write_json(output, packet)
            result = {"status": "review_packet_prepared", "output": str(output),
                      "candidates": len(packet["cards"]), "labels_created": 0}
        elif args.command == "temporal":
            if not args.execute:
                result = {"status": "not_executed", "next": "requires --execute, restored inputs, and new output directory",
                          "policy_execution": False, "sae_training": False}
            else:
                from event_sae.research.output_head.config import load_config
                from event_sae.research.output_head.workflow import _sae_paths, required_input
                from event_sae.research.output_head.readouts import build_readout_manifest, load_readout_cache
                from event_sae.openvla.activations import load_batch_topk_sae
                from event_sae.research.head_followup.temporal import prepare_temporal_cache
                cfg = load_config(args.config)
                frozen = read_json(args.protocol)["frozen_existing_experiment"]
                weights, _ = _sae_paths(cfg)
                sae_hash = sha256_file(weights)
                if sae_hash != frozen["sae_sha256"]:
                    raise ValueError("SAE differs from the frozen existing experiment")
                if cfg["model"]["revision"] != frozen["model_revision"] or cfg["scope"]["layer_idx"] != frozen["layer_idx"]:
                    raise ValueError("model/layer differs from the frozen existing experiment")
                dense = required_input(cfg, "dense_dir")
                output = assert_safe_output(args.output, [dense, cfg["output"]["root_dir"], weights.parent])
                sample = {**cfg["sampling"], "max_steps_per_episode": None, "max_episodes_per_task": None}
                manifest = build_readout_manifest(dense / "activation_index.jsonl", sample,
                    read_json(required_input(cfg, "generation_manifest")), dense_dir=dense,
                    source_episodes=cfg["inputs"]["source_episode_manifest"])
                original = read_json(Path(cfg["output"]["root_dir"]) / "sample_manifest.json")
                if (manifest["discovery_manifest_hash"] != original["discovery_manifest_hash"]
                        or manifest["split_manifest_hash"] != frozen["split_manifest_hash"]
                        or len(manifest["discovery_episodes"]) != frozen["discovery_episodes"]):
                    raise ValueError("full temporal discovery population differs from frozen Head sample/split")
                sae, _ = load_batch_topk_sae(weights, device=args.device)
                reference = load_readout_cache(Path(cfg["output"]["root_dir"]) / "readouts")
                result = prepare_temporal_cache(manifest, sae, output, sae_sha256=sae_hash, device=args.device,
                    hidden_dtype=cfg["model"]["expected_live_hidden_dtype"], max_readouts=args.max_readouts,
                    reference_cache=reference,
                    progress=lambda p: print(json.dumps(p), file=sys.stderr, flush=True))
                result = {"status": "prepared", "output": str(output), "num_readouts": result["num_readouts"],
                          "manifest_hash": result["manifest_hash"], "policy_execution": False, "sae_training": False}
        elif args.command == "temporal-stats":
            from event_sae.research.head_followup.temporal import load_temporal_cache, temporal_statistics
            output = assert_safe_output(args.output, [args.cache])
            result = temporal_statistics(load_temporal_cache(args.cache))
            atomic_write_json(output, result)
            result = {"status": "computed", "output": str(output), "num_episodes": result["num_episodes"],
                      "generality_score": "not_computed"}
        else:
            from event_sae.research.head_followup.metrics import calibrate_generality
            protocol, statistics, labels = read_json(args.protocol), read_json(args.statistics), read_json(args.labels)
            frozen = protocol["frozen_existing_experiment"]
            identity = statistics.get("identity", {})
            if (identity.get("sae_sha256") != frozen["sae_sha256"]
                    or identity.get("split_manifest_hash") != frozen["split_manifest_hash"]
                    or labels.get("statistics_sha256") != sha256_file(args.statistics)):
                raise ValueError("label/statistics/SAE/split binding mismatch")
            output = assert_safe_output(args.output, [args.statistics, args.labels, args.protocol])
            result = calibrate_generality(statistics, labels,
                excluded_feature_ids=protocol["calibrated_comparator"]["previously_measured_features"])
            result.update(identity=identity, labels_sha256=sha256_file(args.labels), synthetic=statistics.get("synthetic"))
            atomic_write_json(output, result)
            result = {"status": "calibrated", "output": str(output), "sae_retrained": False}
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
