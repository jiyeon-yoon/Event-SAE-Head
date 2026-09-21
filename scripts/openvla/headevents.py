#!/usr/bin/env python3
"""Recompute and bind Event scores using only the frozen discovery episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--event-root", required=True, help="Local inherited Event artifact dataset")
    parser.add_argument("--output", help="Default: source registry directory/discovery_event_scores.pt")
    args = parser.parse_args(argv)
    try:
        from event_sae.research.output_head.discovery_scores import build_discovery_event_scores
        result = build_discovery_event_scores(args.config, args.event_root, output_path=args.output)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, OSError, RuntimeError, KeyError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
