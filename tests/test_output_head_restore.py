import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from scripts.openvla.headrestore import REQUIRED, restore_backup


def backup(tmp_path, extras=(), prefix=""):
    path = tmp_path / "backup.tar.gz"
    payloads = {name: b"{}" for name in REQUIRED}
    payloads["plan/rollout_plan.json"] = json.dumps({
        "total_rollouts": 72, "num_eval_cases": 4, "selection_eval_overlap": True,
    }).encode()
    payloads["analysis/report.md"] = b"# Pilot result\n"
    with tarfile.open(path, "w:gz") as archive:
        for name, body in payloads.items():
            info = tarfile.TarInfo(f"{prefix}matched-task01-v1/{name}")
            info.size = len(body)
            archive.addfile(info, io.BytesIO(body))
        for info in extras:
            archive.addfile(info)
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), payloads


def test_inspect_reads_metadata_and_never_extracts(tmp_path):
    path, digest, _ = backup(tmp_path)
    before = list(tmp_path.iterdir())
    result = restore_backup(path, expected_sha256=digest)
    assert result["status"] == "verified"
    assert result["recorded_plan"]["total_rollouts"] == 72
    assert result["rollout_completion"] == "not_revalidated"
    assert list(tmp_path.iterdir()) == before


def test_restore_preserves_bytes_and_refuses_existing_destination(tmp_path):
    path, digest, payloads = backup(tmp_path)
    destination = tmp_path / "restored"
    result = restore_backup(path, expected_sha256=digest, destination=destination)
    assert result["status"] == "restored"
    for name, body in payloads.items():
        assert (destination / "matched-task01-v1" / name).read_bytes() == body
    with pytest.raises(FileExistsError):
        restore_backup(path, expected_sha256=digest, destination=destination)
    assert not list(tmp_path.glob(".headrestore-*"))


def test_wrong_archive_hash_prevents_all_writes(tmp_path):
    path, _, _ = backup(tmp_path)
    destination = tmp_path / "restored"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        restore_backup(path, expected_sha256="0" * 64, destination=destination)
    assert not destination.exists()


def test_dot_prefixed_tar_paths_are_supported(tmp_path):
    path, digest, _ = backup(tmp_path, prefix="./")
    result = restore_backup(path, expected_sha256=digest, destination=tmp_path / "restored")
    assert result["status"] == "restored"
    assert Path(result["report"]).is_file()


@pytest.mark.parametrize("kind", ["traversal", "absolute", "symlink", "hardlink", "duplicate", "outside"])
def test_unsafe_members_are_rejected_before_extraction(tmp_path, kind):
    info = tarfile.TarInfo({
        "traversal": "matched-task01-v1/../../escape",
        "absolute": "/tmp/escape", "symlink": "matched-task01-v1/link",
        "hardlink": "matched-task01-v1/link", "duplicate": "matched-task01-v1/sample_manifest.json",
        "outside": "another-run/test",
    }[kind])
    if kind in ("symlink", "hardlink"):
        info.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
        info.linkname = "/tmp/escape"
    path, digest, _ = backup(tmp_path, [info])
    with pytest.raises(ValueError):
        restore_backup(path, expected_sha256=digest, destination=tmp_path / "restore")
    assert not (tmp_path / "restore").exists()


def test_incomplete_archive_is_not_called_restored(tmp_path):
    path = tmp_path / "empty.tar.gz"
    with tarfile.open(path, "w:gz"):
        pass
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="required result artifacts"):
        restore_backup(path, expected_sha256=digest)


def test_cli_is_standard_library_only(tmp_path):
    path, digest, _ = backup(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts/openvla/headrestore.py"
    result = subprocess.run([sys.executable, "-S", str(script), "--archive", str(path),
                             "--expected-sha256", digest], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "verified"
