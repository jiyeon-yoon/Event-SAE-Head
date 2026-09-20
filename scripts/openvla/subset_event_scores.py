#!/usr/bin/env python3
"""Bind an exact task subset of Event-SAE scores to a head-score sample."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from event_sae.research.output_head.event_score_subset import build_exact_task_subset


def _config_for_update(path: Path, event_scores_path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    inputs = data.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("Config requires an inputs mapping")
    current = inputs.get("event_scores_path")
    desired = str(event_scores_path)
    if current not in (None, "", desired):
        raise ValueError(f"Refusing to replace existing inputs.event_scores_path={current!r}")
    inputs["event_scores_path"] = desired
    return data


def _update_config(path: Path, event_scores_path: Path) -> None:
    data = _config_for_update(path, event_scores_path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yaml.safe_dump(data, stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--sample-manifest", required=True)
    parser.add_argument("--head-scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-id", action="append", type=int, required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-sae-sha256", required=True)
    parser.add_argument("--expected-episodes-per-task", required=True, type=int)
    parser.add_argument("--update-config", help="Set an empty inputs.event_scores_path after success")
    args = parser.parse_args(argv)
    try:
        config_path = None
        if args.update_config:
            config_path = Path(args.update_config).expanduser().resolve()
            # Refuse a conflicting config before creating the write-once subset.
            _config_for_update(config_path, Path(args.output).expanduser().resolve())
        result = build_exact_task_subset(
            args.source, args.sample_manifest, args.head_scores, args.output,
            task_ids=args.task_id, suite=args.suite,
            expected_source_sha256=args.expected_source_sha256,
            expected_sae_sha256=args.expected_sae_sha256,
            expected_episodes_per_task=args.expected_episodes_per_task,
        )
        if config_path is not None:
            _update_config(config_path, Path(args.output).expanduser().resolve())
            result["updated_config"] = str(config_path)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, OSError, RuntimeError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
