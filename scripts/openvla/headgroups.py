#!/usr/bin/env python3
"""Report fixed task 0/1 and task 2-9 subgroups from completed full10 analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from event_sae.research.output_head.groups import write_task_groups


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--root", help="Completed experiment output directory")
    source.add_argument("--config", help="Experiment YAML; reads output.root_dir")
    args = parser.parse_args(argv)
    try:
        root = args.root
        if args.config:
            from event_sae.research.output_head.config import load_config
            root = load_config(args.config)["output"]["root_dir"]
        if not root:
            raise ValueError("An experiment output.root_dir is required")
        result = write_task_groups(root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
