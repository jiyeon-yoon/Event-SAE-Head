#!/usr/bin/env python3
"""Capture live LIBERO processor inputs for output-head runtime parity."""

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
    parser.add_argument("--output")
    parser.add_argument("--allow-simulator", action="store_true")
    args = parser.parse_args(argv)
    try:
        from event_sae.research.output_head.capture import capture_runtime_inputs
        result = capture_runtime_inputs(
            load_config(args.config), args.config,
            allow_simulator=args.allow_simulator, output_path=args.output,
        )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, ImportError, RuntimeError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
