"""Round-trip a release through real Git source export and a dependency-free CLI."""

import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest

from scripts.openvla import headbundle as bundle
from test_output_head_bundle_verify import make_result_fixture


@pytest.fixture
def packed(tmp_path):
    workspace = tmp_path / "workspace"
    repo = workspace / "Event-SAE-Head"
    scripts = repo / "scripts/openvla"
    scripts.mkdir(parents=True)
    real_repo = Path(__file__).resolve().parents[1]
    for filename in ("headfull.py", "download_libero_spatial_reproduction_inputs.py"):
        shutil.copyfile(real_repo / "scripts/openvla" / filename, scripts / filename)
    config_module = repo / "event_sae/research/output_head/config.py"
    config_module.parent.mkdir(parents=True)
    shutil.copyfile(real_repo / "event_sae/research/output_head/config.py", config_module)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "scripts", "event_sae"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Bundle Test", "-c",
                    "user.email=bundle@example.invalid", "commit", "-qm", "fixture source"], check=True)
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    from event_sae.research.output_head.config import DEFAULTS
    run = make_result_fixture(workspace / "event-sae-head-results" / bundle.RUN_NAME, revision,
                              plan_config_hash=bundle.fingerprint(DEFAULTS))
    metadata = workspace / "head-inputs/full10"
    metadata.mkdir(parents=True)
    for name in ("source_episodes.json", "generation_manifest.json", "split.json", "eval_manifest.json",
                 "head.json", "runtime.pt", "discovery_event_scores.pt", "discovery_event_scores.pt.binding.json"):
        (metadata / name).write_bytes(b"{}\n")
    shutil.copyfile(run / "plan/eval_manifest.json", metadata / "eval_manifest.json")
    for name in ("split.json", "source_episodes.json"):
        (metadata / name).write_text(json.dumps({"manifest_hash": bundle.fingerprint({})}))
    (metadata / "discovery_event_scores.pt.binding.json").write_text(json.dumps({
        "output_sha256": bundle.sha256(metadata / "discovery_event_scores.pt")}))
    config = repo / "configs/local/head-full10.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("inputs: {}\n")
    receipt = workspace / "event-sae-spatial-inputs/input_manifest.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"fixture": true}\n')
    (run / "full10-run.log").write_text("Original run log\n")
    output = tmp_path / "release"
    result = bundle.pack(workspace, output)
    assert result["status"] == "packed" and result["total_rollouts"] == 12
    assert result["upload_performed"] is False
    return workspace, output


def test_pack_verify_restore_and_standalone_cli(packed, tmp_path):
    workspace, output = packed
    assert bundle.verify_or_restore(output)["status"] == "verified"
    restored = tmp_path / "new-machine"
    assert bundle.verify_or_restore(output, restored)["conditions"] == 4
    original = workspace / "event-sae-head-results" / bundle.RUN_NAME
    copied = restored / "results" / bundle.RUN_NAME
    for path in bundle.regular_files(original):
        assert path.read_bytes() == (copied / path.relative_to(original)).read_bytes()
    with tarfile.open(restored / "source/experiment-code.tar.gz") as archive:
        assert "Event-SAE-Head/scripts/openvla/headfull.py" in archive.getnames()
    pins = bundle.read_json(restored / "sources/public_inputs.json")
    assert pins["definitions"]["scripts/openvla/headfull.py"]["MODEL_REVISION"]
    assert len((output / "reports/episodes.csv").read_text().splitlines()) == 13
    # -S excludes site-packages: no torch, HF, numpy, yaml, or repository imports.
    result = subprocess.run([sys.executable, "-S", str(output / "tools/headbundle.py"),
                             "inspect", "--root", str(copied)], cwd=tmp_path,
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["total_rollouts"] == 12
    result = subprocess.run([sys.executable, "-S", str(output / "tools/headbundle.py"),
                             "verify", "--bundle", str(output)], cwd=tmp_path,
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["status"] == "verified"


def test_tampered_download_and_existing_destinations_rejected(packed, tmp_path):
    workspace, output = packed
    with pytest.raises(FileExistsError):
        bundle.pack(workspace, output)
    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(ValueError, match="NEW"):
        bundle.verify_or_restore(output, destination)
    report = output / "reports/summary.json"
    report.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        bundle.verify_or_restore(output)


@pytest.mark.parametrize("member_name,kind", [
    ("../escape", "file"), ("/absolute", "file"), ("payload", "symlink"),
    ("payload", "hardlink"), ("payload", "directory"), ("payload", "wrongsize"),
    ("payload", "wronghash"), ("other", "file"),
])
def test_unsafe_archive_rejected(tmp_path, member_name, kind):
    archive_path = tmp_path / "experiment.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        entry = tarfile.TarInfo(member_name)
        data = b"test"
        entry.size = len(data)
        if kind in {"symlink", "hardlink"}:
            entry.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
            entry.linkname = "../escape"
            entry.size = 0
        elif kind == "directory":
            entry.type = tarfile.DIRTYPE
            entry.size = 0
        archive.addfile(entry, io.BytesIO(data) if entry.isfile() else None)
    manifest = {"files": {"payload": {"size_bytes": 5 if kind == "wrongsize" else 4,
                                      "sha256": "0" * 64}},
                "uncompressed_bytes": 5 if kind == "wrongsize" else 4}
    with pytest.raises(ValueError):
        bundle.extract_verified(tmp_path, tmp_path / "destination", manifest)
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize("value", ["../x", "/x", "x/../y", "x//y", "x\\y", "x\ny", "./x", ""])
def test_unsafe_relative_paths(value):
    with pytest.raises(ValueError, match="Unsafe"):
        bundle.safe_relative(value)


def test_source_symlink_rejected(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "alias").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="Links"):
        bundle.regular_files(tree)


def test_changed_historical_config_rejected(packed, tmp_path):
    workspace, _ = packed
    config = workspace / "Event-SAE-Head/configs/local/head-full10.yaml"
    config.write_text("scoring:\n  device: cuda\n")
    with pytest.raises(ValueError, match="frozen experiment configuration"):
        bundle.pack(workspace, tmp_path / "different")


@pytest.mark.parametrize("filename", ["runtime.pt", "eval_manifest.json", "generation_manifest.json",
                                     "discovery_event_scores.pt", "split.json", "source_episodes.json"])
def test_changed_generated_input_rejected(packed, tmp_path, filename):
    workspace, _ = packed
    (workspace / "head-inputs/full10" / filename).write_text('{"changed": true}\n')
    with pytest.raises(ValueError):
        bundle.pack(workspace, tmp_path / "different")
