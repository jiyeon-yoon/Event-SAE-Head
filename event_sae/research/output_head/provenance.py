"""Small, dependency-free provenance and non-overwriting artifact utilities."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable


def fingerprint(payload: Any) -> str:
    """Hash strict canonical JSON; non-finite scientific values must be null."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def atomic_write_text(path: str | Path, text: str, *, overwrite: bool = False) -> None:
    """Publish a complete file. Default uses a no-clobber link, including races."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(path: str | Path, payload: Any, *, overwrite: bool = False) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n", overwrite=overwrite)


def assert_compatible(actual: dict, expected: dict) -> None:
    """Require all requested identity fields, not just an old passed flag."""
    differences = [key for key, value in expected.items()
                   if key not in actual or actual[key] != value]
    if differences:
        raise ValueError(f"Stale/incompatible provenance fields: {', '.join(differences)}")


def assert_safe_output(output: str | Path, inputs: Iterable[str | Path | None]) -> Path:
    """Reject source/output containment in either direction, including symlinks."""
    target = Path(output).expanduser().resolve()
    if target in (Path('/'), Path.home()):
        raise ValueError("A filesystem/home root is not a research output directory")
    for source in inputs:
        if not source:
            continue
        original = Path(source).expanduser().resolve()
        if target == original or target in original.parents or original in target.parents:
            raise ValueError(f"Research output overlaps read-only input: {original}")
    return target


def git_identity(root: str | Path) -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    return {"commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain"))}


def implementation_fingerprint() -> str:
    """Hash relevant source, including uncommitted edits, not output locations."""
    directory = Path(__file__).resolve().parent
    package_root = directory.parents[1]
    files = sorted(directory.glob("*.py")) + [
        package_root / "openvla/intervene.py",
        package_root / "openvla/activations.py",
        package_root / "openvla/eval/model.py",
        package_root / "openvla/eval/runner.py",
        package_root / "openvla/eval/config.py",
    ]
    return fingerprint({str(path.relative_to(package_root)): sha256_file(path)
                        for path in files if path.is_file()})


def reuse_json(path: str | Path, identity: dict) -> dict | None:
    """Resume only an intact artifact with an exactly matching identity."""
    path = Path(path)
    if not path.exists():
        return None
    payload = read_json(path)
    assert_compatible(payload.get("identity", {}), identity)
    if payload.get("identity_hash") != fingerprint(identity):
        raise ValueError("Artifact identity hash mismatch")
    return payload
