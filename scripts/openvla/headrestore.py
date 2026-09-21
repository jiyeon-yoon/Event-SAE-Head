"""Inspect or restore a downloaded output-head backup without model dependencies.

Download/authentication remain explicit terminal steps. Restoration preserves
artifact bytes, including historical absolute paths and provenance hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile


PILOT_SHA256 = "057e99bb74673c0ee28e599192875af4b338cfd274d8267d035988057067f1bf"
REQUIRED = (
    "sample_manifest.json", "scores/scores.json", "plan/rollout_plan.json",
    "analysis/report.md",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _members(archive: tarfile.TarFile, run_name: str) -> list[tarfile.TarInfo]:
    members, seen, total = [], set(), 0
    for member in archive:
        path = PurePosixPath(member.name)
        if (path.is_absolute() or ".." in path.parts or "\\" in member.name
                or not path.parts or path.parts[0] != run_name):
            raise ValueError(f"Archive member outside expected run: {member.name}")
        if not member.isfile() and not member.isdir():
            raise ValueError(f"Links/special files are not supported: {member.name}")
        if len(path.parts) == 1 and not member.isdir():
            raise ValueError("Archive run root must be a directory")
        key = str(path)
        if key in seen:
            raise ValueError(f"Duplicate archive member: {key}")
        seen.add(key)
        total += member.size
        members.append(member)
        if len(members) > 100_000 or total > 10 * 1024**3:
            raise ValueError("Backup exceeds the supported 100,000 files / 10 GiB limit")
    regular_files = {str(PurePosixPath(m.name)) for m in members if m.isfile()}
    missing = [name for name in REQUIRED if f"{run_name}/{name}" not in regular_files]
    if missing:
        raise ValueError(f"Backup lacks required result artifacts: {', '.join(missing)}")
    return members


def _json_member(archive: tarfile.TarFile, name: str) -> dict:
    member = next((item for item in archive.getmembers()
                   if str(PurePosixPath(item.name)) == name), None)
    if member is None:
        raise ValueError(f"Missing JSON artifact: {name}")
    if not member.isfile() or member.size > 64 * 1024**2:
        raise ValueError(f"Invalid or oversized JSON artifact: {name}")
    with archive.extractfile(member) as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON mapping: {name}")
    return value


def restore_backup(archive_path: str | Path, *, expected_sha256: str = PILOT_SHA256,
                   run_name: str = "matched-task01-v1",
                   destination: str | Path | None = None) -> dict:
    """Check the actual archive; optionally extract into a new directory.

This validates transport integrity and inventory, not scientific conclusions or
compatibility with the current code. Never rewrites old paths/certificates.
"""
    archive_path = Path(archive_path).expanduser().resolve()
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("expected_sha256 must be a SHA-256 digest")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name):
        raise ValueError("run_name must be a single directory name")
    actual = _sha256(archive_path)
    if actual != expected_sha256.lower():
        raise ValueError(f"Archive SHA-256 mismatch: {actual}")
    target = None
    if destination is not None:
        requested = Path(destination).expanduser()
        if requested.exists() or requested.is_symlink():
            raise FileExistsError(f"Destination must be new: {requested}")
        target = requested.resolve()
        if target == Path("/") or target == Path.home() or target in archive_path.parents:
            raise ValueError("Destination must be a separate new restore directory")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = _members(archive, run_name)
        plan = _json_member(archive, f"{run_name}/plan/rollout_plan.json")
        sample = _json_member(archive, f"{run_name}/sample_manifest.json")
        result = {
            "status": "verified", "archive": str(archive_path), "sha256": actual,
            "run_name": run_name, "files": sum(m.isfile() for m in members),
            "uncompressed_bytes": sum(m.size for m in members),
            "required_artifacts": list(REQUIRED),
            "recorded_plan": {
                "plan_hash": plan.get("plan_hash"), "mode": plan.get("mode"),
                "tasks": plan.get("scope", {}).get("task_ids"),
                "num_eval_cases": plan.get("num_eval_cases"),
                "num_unique_features": plan.get("num_unique_features"),
                "total_rollouts": plan.get("total_rollouts"),
                "selection_eval_overlap": plan.get("selection_eval_overlap"),
            },
            "discovery_episodes": len(sample.get("discovery_episodes", [])),
            "runtime_validation": "not_performed",
            "rollout_completion": "not_revalidated",
            "historical_paths": "preserved_without_rewriting",
        }
        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".headrestore-", dir=target.parent) as temporary:
                staged = Path(temporary) / "restored"
                staged.mkdir()
                for member in members:
                    output = staged.joinpath(*PurePosixPath(member.name).parts)
                    if member.isdir():
                        output.mkdir(parents=True, exist_ok=True)
                    else:
                        output.parent.mkdir(parents=True, exist_ok=True)
                        with archive.extractfile(member) as source, output.open("xb") as stream:
                            shutil.copyfileobj(source, stream)
                # Reserve the destination without replacing an existing file/dir.
                target.mkdir()
                try:
                    (staged / run_name).rename(target / run_name)
                except Exception:
                    target.rmdir()  # Only the empty directory reserved above.
                    raise
            result.update(status="restored", destination=str(target),
                          restored_run=str(target / run_name),
                          report=str(target / run_name / "analysis/report.md"))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--expected-sha256", default=PILOT_SHA256,
                        help="Default: independently recorded matched-task01-v1 checksum")
    parser.add_argument("--run-name", default="matched-task01-v1")
    parser.add_argument("--destination", help="Extract into this NEW directory; omit to inspect only")
    args = parser.parse_args(argv)
    try:
        result = restore_backup(args.archive, expected_sha256=args.expected_sha256,
                                run_name=args.run_name, destination=args.destination)
    except (OSError, ValueError, tarfile.TarError, EOFError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
