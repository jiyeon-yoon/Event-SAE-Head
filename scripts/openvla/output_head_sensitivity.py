"""CPU-safe entry point for the independent output-head sensitivity study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from event_sae.research.output_head.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "split", "export-head", "prepare", "validate", "score", "plan", "run", "analyze"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", required=True, help="YAML; paths are relative to current working directory")
        if name == "validate":
            sub.add_argument("--allow-model-execution", action="store_true")
        elif name == "run":
            sub.add_argument("--approved-plan-hash")
            sub.add_argument("--expected-code-revision")
            sub.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        from event_sae.research.output_head import workflow
        if args.command == "validate":
            if not args.allow_model_execution:
                raise ValueError("validate requires --allow-model-execution; no model was loaded")
            from event_sae.research.output_head.runtime import validate_runtime
            result = validate_runtime(cfg)
        elif args.command == "run":
            if not args.execute or not args.approved_plan_hash or not args.expected_code_revision:
                raise ValueError("run requires --execute, --approved-plan-hash and --expected-code-revision")
            from event_sae.research.output_head.runtime import run_plan
            result = run_plan(cfg, args.approved_plan_hash, args.expected_code_revision)
        else:
            function = {"audit": workflow.audit_inputs, "split": workflow.split_workflow,
                        "export-head": workflow.export_workflow, "prepare": workflow.prepare_workflow,
                        "score": workflow.score_workflow, "plan": workflow.plan_workflow,
                        "analyze": workflow.analyze_workflow}[args.command]
            result = function(cfg)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 2 if result.get("status", "").startswith("blocked") else 0
    except (ValueError, FileNotFoundError, FileExistsError, ImportError, RuntimeError) as exc:
        print(json.dumps({"status": "blocked", "command": args.command,
                          "error": str(exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
