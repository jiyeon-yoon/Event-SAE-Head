"""Portable release of a completed output-head experiment.

Never changes experiment files, executes a policy, imports torch, or uploads.
The archive preserves original bytes and paths inside the recorded documents;
the companion verifier resolves action paths relative to the restored run.
Pack uses PyYAML from the experiment environment; verify/restore/inspect use only
the standard library.
"""

from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

try:
    from .headbundle_verify import audit_results, fingerprint
except ImportError:
    from headbundle_verify import audit_results, fingerprint


SCHEMA = "output_head_public_bundle_v1"
RUN_NAME = "full10-followup-v1"
HF_REPO = "jiyeony/event-sae-head-full10-results"
MAX_FILES = 100_000
MAX_BYTES = 100 * 1024**3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or ".." in path.parts or "\\" in value
            or str(path) != value or any(ord(char) < 32 for char in value)):
        raise ValueError(f"Unsafe bundle path: {value!r}")
    return value


def regular_files(root: Path) -> list[Path]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Expected a real directory: {root}")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise ValueError(f"Links/special files cannot enter a public bundle: {path}")
        if path.is_file():
            safe_relative(path.relative_to(root).as_posix())
            files.append(path)
    return files


def snapshot(path: Path) -> tuple[int, int, int]:
    state = path.stat()
    return state.st_size, state.st_mtime_ns, state.st_ino


def copy_stable(source: Path, target: Path) -> tuple[int, int, int]:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Expected a regular source file: {source}")
    before = snapshot(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, target.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=8 * 1024**2)
    if snapshot(source) != before:
        raise ValueError(f"Source changed during packaging: {source}")
    return before


def source_definitions(repo: Path, revision: str) -> dict:
    """Read pinned public-input constants without executing historical code."""
    definitions = {}
    for filename in ("download_libero_spatial_reproduction_inputs.py", "headfull.py"):
        relative = f"scripts/openvla/{filename}"
        source = subprocess.check_output(["git", "-C", str(repo), "show", f"{revision}:{relative}"], text=True)
        constants = {}
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in {"DATASETS", "PAIRS", "INDEX_SHA", "EVENT_REPO", "EVENT_REVISION",
                            "MODEL_REPO", "MODEL_REVISION", "CODE_REVISION", "SAE_SHA", "SAE_SHA256",
                            "SAE_REPO", "SAE_REVISION", "VISION_MODEL_REPO", "VISION_MODEL_REVISION"}:
                    constants[name] = ast.literal_eval(node.value)
        definitions[relative] = constants
    return {"experiment_code_revision": revision, "definitions": definitions,
            "excluded_downloadable_assets": ["original 281 GiB dense activations and videos", "OpenVLA full weights", "SAE checkpoint", "inherited full500 Event TopK tensors"],
            "included_derived_inputs": "All files from head-inputs/full10, including discovery Event scores and runtime inputs"}


def normalized_config(repo: Path, revision: str, path: Path, plan: dict) -> dict:
    """Bind YAML to the frozen plan using literal defaults from the recorded code."""
    try:
        import yaml
    except ImportError as error:
        raise ValueError("pack requires PyYAML in the original experiment environment") from error
    source = subprocess.check_output(["git", "-C", str(repo), "show",
                                      f"{revision}:event_sae/research/output_head/config.py"], text=True)
    defaults = [node for node in ast.parse(source).body
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                and node.target.id == "DEFAULTS"]
    if len(defaults) != 1:
        raise ValueError("Cannot identify historical literal configuration defaults")
    config = ast.literal_eval(defaults[0].value)

    def merge(base, updates):
        if not isinstance(updates, dict):
            raise ValueError("Configuration must be a mapping")
        for key, value in updates.items():
            if key not in base:
                raise ValueError(f"Unknown historical configuration field: {key}")
            if isinstance(base[key], dict):
                merge(base[key], value)
            else:
                base[key] = value

    merge(config, yaml.safe_load(path.read_text()) or {})
    if fingerprint(config) != plan.get("experiment_config_hash"):
        raise ValueError("Supplied YAML differs from the frozen experiment configuration")
    return config


def verify_preserved_inputs(payload: Path, run_name: str) -> None:
    """Bind copied generated inputs to saved evidence without opening tensor files."""
    run, inputs = payload / "results" / run_name, payload / "inputs/full10"
    plan = read_json(run / "plan/rollout_plan.json")
    normalized = read_json(payload / "config/normalized.json")
    if fingerprint(normalized) != plan.get("experiment_config_hash"):
        raise ValueError("Normalized configuration differs from frozen plan")
    evaluation = inputs / "eval_manifest.json"
    if sha256(evaluation) != plan["eval_manifest_hash"] or read_json(evaluation) != plan["eval_manifest"]:
        raise ValueError("Preserved evaluation input differs from frozen plan")
    split = read_json(inputs / "split.json")
    split_hash = fingerprint({k: v for k, v in split.items() if k != "manifest_hash"})
    if split.get("manifest_hash") != split_hash:
        raise ValueError("Preserved split hash mismatch")
    bound_split = plan["scope"].get("split_manifest_hash")
    if bound_split is not None and bound_split != split_hash:
        raise ValueError("Preserved split differs from score/plan scope")
    source = read_json(inputs / "source_episodes.json")
    if source.get("manifest_hash") != fingerprint({k: v for k, v in source.items() if k != "manifest_hash"}):
        raise ValueError("Source episode registry hash mismatch")
    for filename in ("runtime_parity.json", "edit_parity.json"):
        parity = read_json(run / "validation" / filename)
        if parity.get("sample_identity") != sha256(inputs / "runtime.pt"):
            raise ValueError("Preserved runtime inputs differ from parity evidence")
        if parity["identity"].get("generation_manifest_hash") != sha256(inputs / "generation_manifest.json"):
            raise ValueError("Preserved generation manifest differs from numerical identity")
    event_receipt = read_json(inputs / "discovery_event_scores.pt.binding.json")
    if event_receipt.get("output_sha256") != sha256(inputs / "discovery_event_scores.pt"):
        raise ValueError("Preserved Event scores differ from binding receipt")


def capture_environment(target: Path) -> None:
    packages = sorted(({"name": item.metadata.get("Name", "unknown"), "version": item.version}
                       for item in importlib.metadata.distributions()), key=lambda item: item["name"].lower())
    write_json(target / "packages.json", packages)
    write_json(target / "export_environment.json", {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "platform": platform.platform(), "scope": "export-time snapshot; experiment-time identities remain in original artifacts",
    })
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15)
        status = {"returncode": result.returncode, "output": result.stdout, "error": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        status = {"status": "unavailable", "reason": str(error)}
    write_json(target / "nvidia-smi.json", status)


def make_reports(result_root: Path, target: Path, audit: dict) -> None:
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "summary.json", audit)
    for path in sorted((result_root / "analysis").iterdir()):
        if path.suffix in {".md", ".csv"} and path.is_file():
            copy_stable(path, target / path.name)
    plan = read_json(result_root / "plan/rollout_plan.json")
    columns = ["condition_id", "mode", "feature_id", "alpha", "task_id", "task_episode_idx",
               "initial_state_sha256", "success", "num_actions", "caught_exception"]
    with (target / "episodes.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for condition in plan["conditions"]:
            result = read_json(result_root / "runs" / condition["condition_id"] / "result.json")
            for episode in result["episodes"]:
                row = {name: episode.get(name) for name in columns}
                row.update({name: condition.get(name) for name in ("condition_id", "mode", "feature_id", "alpha")})
                writer.writerow(row)


def dataset_card(audit: dict, run_name: str) -> str:
    hours = audit["condition_elapsed_seconds_sum"] / 3600
    return f'''---
language:
- en
- ko
tags:
- robotics
- interpretability
- sparse-autoencoder
- openvla
- libero
pretty_name: Event-SAE Head full10 completed experiment
---

# Event-SAE Head: completed LIBERO-Spatial experiment

This public release preserves **{audit['total_rollouts']} completed rollouts**, {audit['num_conditions']} conditions,
{audit['num_eval_cases']} paired evaluation cases and {audit['num_features']} ablated features.
The model and SAE were frozen. These are post-pilot follow-up results, not a fully held-out confirmatory study.

## Read results immediately

- [Main report](reports/report.md)
- [Task 0–1 and task 2–9](reports/task_groups_v1.md)
- [Machine-readable summary and actual timings](reports/summary.json)
- [Every evaluated episode](reports/episodes.csv)
- [Paired effects](reports/paired_effects.csv)
- [Predictor comparison](reports/predictor_comparison.csv)

The reports contain the measured Head-KL/Event results and their scope. Positive drop means
success deteriorated after removing a feature; values multiplied by 100 are percentage points.
Top-3 mean drop averages three **separate single-feature interventions**, not one joint intervention.
The archived analysis records the paired bootstrap confidence interval for the method difference.

## What is preserved

`experiment.tar.gz` contains the complete original result directory (all conditions, per-episode
results, action sequences, logs, scores, readout cache, head bundle, parity certificates, plan and
analysis), the complete generated `head-inputs/full10` metadata/input directory, and the exact
local experiment YAML (checked against the frozen normalized configuration hash).
Original documents are unchanged, including their historical absolute paths.

The archive also contains the exact source at commit `{audit['experiment_code_revision']}`,
public input repositories and pinned revisions, and an export-time package/GPU environment snapshot.
The summed recorded condition runtimes are **{hours:.3f} hours**. This is not an independently measured
Pod wall-clock duration or bill: preprocessing, downloads, analysis and idle time are separate.

Source public dense activations, full model weights, inherited TopK tensors and the SAE checkpoint
are referenced by pinned revisions/checksums, not duplicated in this release. The exact download
scripts are in the archived source. Videos are included only if they were actually saved;
the full10 preset disables new rollout video recording.

## Download, verify and restore anywhere (no GPU or model needed)

Install the Hugging Face CLI, then download the release. For citation/reproducibility use
`--revision <dataset-commit>` with the commit displayed in the Hugging Face Files history.

```bash
hf download {HF_REPO} --repo-type dataset --local-dir ./head-full10
python ./head-full10/tools/headbundle.py verify --bundle ./head-full10
python ./head-full10/tools/headbundle.py restore --bundle ./head-full10 --destination ./head-full10-restored
python ./head-full10/tools/headbundle.py inspect --root ./head-full10-restored/results/{run_name}
```

The tools require Python 3.10+ and its standard library only. Restoration requires a NEW destination.
Verification checks archive and per-file SHA-256, all frozen conditions and cases, original result
fingerprints, saved action hashes/identity, and recomputes observed success-rate drops from episodes.
It does not execute torch/pickle files, replay the simulator, or rerun GPU numerical parity.
The bootstrap intervals are preserved from the original analysis; checking their saved consistency
is distinct from recomputing all bootstrap draws.

`manifest.json` inventories archived files. `SHA256SUMS` inventories every published bundle file
except itself. Verification needs temporary free disk space for the uncompressed archive.
The preserved source archive is `source/experiment-code.tar.gz` after restoration.
To rerun the original analysis/rollouts, use that source, its documented dependencies and original
`/workspace` layout; the portable verifier intentionally does not rewrite provenance-bound paths.

## Research scope

The discovery/evaluation trial lists are separated. The SAE and Event clusters were inherited
from existing data; historical collection state bytes and generation flags are not independently
verified. Task 0–1 informed the earlier pilot. Rankings were computed across all ten tasks, so the
task 2–9 subgroup does not establish held-out-task generalization. Uncertainty is conditional on
the fixed tasks and evaluated feature panel. See the original reports for full limitations.

Only code, scientific artifacts and a curated environment inventory are packaged. Authentication
tokens, shell histories, Hugging Face caches and `.git` directories are not collected.
'''


def pack(workspace: Path, output: Path, *, run_name: str = RUN_NAME, config: Path | None = None) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name):
        raise ValueError("run-name must be one directory name")
    workspace, output = workspace.expanduser().resolve(), output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output must be a new directory: {output}")
    output = output.resolve()
    repo = workspace / "Event-SAE-Head"
    run = workspace / "event-sae-head-results" / run_name
    metadata = workspace / "head-inputs/full10"
    config = Path(config).expanduser().resolve() if config else repo / "configs/local/head-full10.yaml"
    for source in (repo, run, metadata):
        if source == output or source in output.parents or output in source.parents:
            raise ValueError("Release output must be separate from source directories")
    audit = audit_results(run)
    required_metadata = ("source_episodes.json", "generation_manifest.json", "split.json", "eval_manifest.json",
                         "head.json", "runtime.pt", "discovery_event_scores.pt", "discovery_event_scores.pt.binding.json")
    for name in required_metadata:
        if not (metadata / name).is_file():
            raise FileNotFoundError(metadata / name)
    if not config.is_file():
        raise FileNotFoundError(config)
    revision = audit["experiment_code_revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Experiment must record a full Git commit")
    subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{revision}^{{commit}}"], check=True)
    config_identity = normalized_config(repo, revision, config, read_json(run / "plan/rollout_plan.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    tracked = {}
    with tempfile.TemporaryDirectory(prefix=".headbundle-", dir=output.parent) as temporary:
        stage, payload = Path(temporary) / "release", Path(temporary) / "payload"
        stage.mkdir()
        payload.mkdir()
        trees = {run: f"results/{run_name}", metadata: "inputs/full10"}
        inventories = {source: regular_files(source) for source in trees}
        for source, relative in trees.items():
            for original in inventories[source]:
                target = payload / relative / original.relative_to(source)
                tracked[original] = copy_stable(original, target)
        tracked[config] = copy_stable(config, payload / "config/head-full10.yaml")
        if normalized_config(repo, revision, payload / "config/head-full10.yaml",
                             read_json(payload / "results" / run_name / "plan/rollout_plan.json")) != config_identity:
            raise ValueError("Configuration changed during packaging")
        write_json(payload / "config/normalized.json", config_identity)
        input_manifest = workspace / "event-sae-spatial-inputs/input_manifest.json"
        tracked[input_manifest] = copy_stable(input_manifest, payload / "sources/original_input_manifest.json")
        (payload / "source").mkdir()
        subprocess.run(["git", "-C", str(repo), "archive", "--format=tar.gz", "--prefix=Event-SAE-Head/",
                        f"--output={payload / 'source/experiment-code.tar.gz'}", revision], check=True)
        write_json(payload / "sources/public_inputs.json", source_definitions(repo, revision))
        capture_environment(payload / "environment")
        verify_preserved_inputs(payload, run_name)
        copied_audit = audit_results(payload / "results" / run_name)
        if copied_audit != audit:
            raise ValueError("Experiment changed during packaging")
        make_reports(payload / "results" / run_name, stage / "reports", copied_audit)
        tools_dir = stage / "tools"
        tools_dir.mkdir()
        for name in ("headbundle.py", "headbundle_verify.py"):
            copy_stable(Path(__file__).with_name(name), tools_dir / name)
        (stage / "README.md").write_text(dataset_card(audit, run_name), encoding="utf-8")
        files = {path.relative_to(payload).as_posix(): {"sha256": sha256(path), "size_bytes": path.stat().st_size}
                 for path in regular_files(payload)}
        if len(files) > MAX_FILES or sum(item["size_bytes"] for item in files.values()) > MAX_BYTES:
            raise ValueError("Bundle exceeds supported 100,000 files / 100 GiB")
        archive = stage / "experiment.tar.gz"
        with tarfile.open(archive, "w:gz", compresslevel=1, dereference=False) as stream:
            for relative in files:
                stream.add(payload / relative, arcname=relative, recursive=False)
        for original, before in tracked.items():
            if snapshot(original) != before:
                raise ValueError(f"Source changed during packaging: {original}")
        if any(regular_files(source) != inventories[source] for source in trees):
            raise ValueError("Source inventory changed during packaging")
        manifest = {"schema_version": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "run_name": run_name, "result_root": f"results/{run_name}", "experiment": audit,
                    "archive": {"path": archive.name, "sha256": sha256(archive), "size_bytes": archive.stat().st_size},
                    "files": files, "uncompressed_bytes": sum(item["size_bytes"] for item in files.values()),
                    "preservation": "original experiment bytes unchanged; historical paths preserved"}
        write_json(stage / "manifest.json", manifest)
        lines = [f"{sha256(path)}  {path.relative_to(stage).as_posix()}\n" for path in regular_files(stage)]
        (stage / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")
        output.mkdir()  # Reserve without replacing any existing user directory.
        stage.rename(output)
    return {"status": "packed", "bundle": str(output), "archive_sha256": manifest["archive"]["sha256"],
            "archive_bytes": manifest["archive"]["size_bytes"], "total_rollouts": audit["total_rollouts"],
            "conditions": audit["num_conditions"], "recorded_condition_hours": audit["condition_elapsed_seconds_sum"] / 3600,
            "upload_performed": False}


def verify_bundle_files(root: Path) -> dict:
    root = root.expanduser().resolve()
    if not (root / "SHA256SUMS").is_file():
        raise FileNotFoundError(root / "SHA256SUMS")
    recorded = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        if not re.fullmatch(r"[0-9a-f]{64}  .+", line):
            raise ValueError("Invalid SHA256SUMS line")
        digest, relative = line.split("  ", 1)
        safe_relative(relative)
        if relative in recorded or relative == "SHA256SUMS":
            raise ValueError("Duplicate or recursive checksum entry")
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Published file checksum mismatch: {relative}")
        recorded[relative] = digest
    required = {"experiment.tar.gz", "manifest.json", "README.md", "reports/summary.json",
                "tools/headbundle.py", "tools/headbundle_verify.py"}
    if not required <= set(recorded):
        raise ValueError("Checksum inventory omits required publication files")
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema_version") != SCHEMA or manifest.get("archive", {}).get("path") != "experiment.tar.gz":
        raise ValueError("Unsupported bundle schema/archive")
    if recorded["experiment.tar.gz"] != manifest["archive"]["sha256"]:
        raise ValueError("Archive digest differs between manifest and checksum file")
    if (root / "experiment.tar.gz").stat().st_size != manifest["archive"]["size_bytes"]:
        raise ValueError("Archive byte size differs from manifest")
    if read_json(root / "reports/summary.json") != manifest["experiment"]:
        raise ValueError("Published summary differs from manifest")
    return manifest


def extract_verified(bundle: Path, destination: Path, manifest: dict) -> dict:
    expected = manifest["files"]
    if not expected or len(expected) > MAX_FILES:
        raise ValueError("Invalid archive inventory size")
    for relative, item in expected.items():
        safe_relative(relative)
        if type(item["size_bytes"]) is not int or item["size_bytes"] < 0 or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise ValueError("Invalid archive member metadata")
    total = sum(item["size_bytes"] for item in expected.values())
    if total > MAX_BYTES or total != manifest.get("uncompressed_bytes"):
        raise ValueError("Invalid or oversized uncompressed archive")
    seen = set()
    with tarfile.open(bundle / "experiment.tar.gz", "r|gz") as archive:
        for member in archive:
            name = safe_relative(member.name)
            if not member.isfile() or name in seen or name not in expected or member.size != expected[name]["size_bytes"]:
                raise ValueError(f"Unexpected/unsafe archive member: {name}")
            seen.add(name)
            output = destination / name
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as incoming, output.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=8 * 1024**2)
            if sha256(output) != expected[name]["sha256"]:
                raise ValueError(f"Archived file checksum mismatch: {name}")
    if seen != set(expected):
        raise ValueError("Archive has missing files")
    relative_root = safe_relative(manifest["result_root"])
    if relative_root != f"results/{manifest['run_name']}":
        raise ValueError("Result root differs from bundle run name")
    verify_preserved_inputs(destination, manifest["run_name"])
    audit = audit_results(destination / relative_root)
    if audit != manifest["experiment"]:
        raise ValueError("Restored experiment differs from published summary")
    return audit


def verify_or_restore(bundle: Path, destination: Path | None = None) -> dict:
    bundle = bundle.expanduser().resolve()
    target = destination.expanduser().absolute() if destination else None
    if target is not None:
        if target.exists() or target.is_symlink():
            raise ValueError("Restore destination must be a separate NEW directory")
        target = target.resolve()
        if target == bundle or bundle in target.parents or target in bundle.parents:
            raise ValueError("Restore destination must be a separate NEW directory")
        target.parent.mkdir(parents=True, exist_ok=True)
    manifest = verify_bundle_files(bundle)
    parent = target.parent if target is not None else bundle.parent
    if shutil.disk_usage(parent).free < manifest["uncompressed_bytes"] + 128 * 1024**2:
        raise ValueError("Insufficient free disk for verified extraction")
    with tempfile.TemporaryDirectory(prefix=".headbundle-verify-", dir=parent) as temporary:
        restored = Path(temporary) / "restored"
        restored.mkdir()
        audit = extract_verified(bundle, restored, manifest)
        if target is not None:
            target.mkdir()
            restored.rename(target)
    return {"status": "restored" if target else "verified", "archive_sha256": manifest["archive"]["sha256"],
            "total_rollouts": audit["total_rollouts"], "conditions": audit["num_conditions"],
            "destination": str(target) if target else None, "result_root": str(target / manifest["result_root"]) if target else None,
            "model_execution": False, "scientific_scope": "stored outcome/action consistency; no simulator or GPU parity rerun"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("pack")
    create.add_argument("--workspace", type=Path, default=Path("/workspace"))
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--run-name", default=RUN_NAME)
    create.add_argument("--config", type=Path)
    for name in ("verify", "restore"):
        command = commands.add_parser(name)
        command.add_argument("--bundle", type=Path, required=True)
        if name == "restore":
            command.add_argument("--destination", type=Path, required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "pack":
            result = pack(args.workspace, args.output, run_name=args.run_name, config=args.config)
        elif args.command == "inspect":
            result = audit_results(args.root)
        else:
            result = verify_or_restore(args.bundle, getattr(args, "destination", None))
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError, EOFError, subprocess.CalledProcessError) as error:
        print(json.dumps({"status": "blocked", "command": args.command, "error": str(error)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
