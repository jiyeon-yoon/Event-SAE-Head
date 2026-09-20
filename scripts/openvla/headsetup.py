#!/usr/bin/env python3
"""Configure and optionally export the pinned local OpenVLA output head."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from event_sae.research.output_head.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--metadata-output")
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args(argv)
    try:
        from event_sae.research.output_head import workflow
        cfg = load_config(args.config)
        result = {"setup": workflow.setup_head_workflow(
            cfg, args.config, args.snapshot, args.metadata_output,
        )}
        if args.export:
            result["export"] = workflow.export_workflow(load_config(args.config))
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, ImportError, RuntimeError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
