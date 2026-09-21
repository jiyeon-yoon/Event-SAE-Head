"""Fixed full10 subgroups retain selection and recompute complete episode pairs."""

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from event_sae.research.output_head.groups import analyze_task_groups, subset_paired_effects, write_task_groups
from event_sae.research.output_head.provenance import fingerprint
from event_sae.research.output_head.results import analyze_prediction, assemble_paired_effects


def _freeze(plan):
    plan["plan_hash"] = fingerprint({k: v for k, v in plan.items() if k != "plan_hash"})
    return plan


def _fixture():
    raw = [{"suite": "synthetic_suite", "task_id": task, "task_episode_idx": trial,
            "initial_state_sha256": f"state-{task}-{trial}", "seed": 7,
            "success": task != 1, "caught_exception": None}
           for task in range(10) for trial in range(2 if task in (0, 2) else 1)]
    edits = {feature: copy.deepcopy(raw) for feature in (10, 11, 12)}
    for row in edits[10]:
        if row["task_id"] == 0 and row["task_episode_idx"] == 0:
            row["success"] = False
        if row["task_id"] == 1:
            row["success"] = True
    for row in edits[12]:
        if row["task_id"] >= 2:
            row["success"] = False
    cases = [{name: row[name] for name in ("suite", "task_id", "task_episode_idx", "initial_state_sha256")}
             for row in raw]
    protocol = {"synthetic": True, "protocol_id": "synthetic-full10", "suite": "synthetic_suite",
                "seed": 7, "selection_eval_overlap": False, "evaluation_labels_previously_used": None,
                "eval_cases": cases, "expected_feature_ids": [10, 11, 12]}
    paired = assemble_paired_effects(raw, edits, protocol)
    plan = _freeze({"status": "planned", "feature_ids": [10, 11, 12],
                    "candidates": [{"feature_id": feature} for feature in (10, 11, 12)],
                    "methods": ["head", "event"], "top_k": 1,
                    "method_topk": {"head": [10], "event": [11]},
                    "score_vectors": {"head": {"10": 3., "11": 2., "12": 1.},
                                      "event": {"10": 2., "11": 3., "12": 1.}},
                    "audit_feature_ids": [12], "feature_universe": [10, 11, 12],
                    "eval_cases": cases, "evaluation_labels_previously_used": None})
    analysis = analyze_prediction(plan, paired, {"bootstrap_replicates": 20, "bootstrap_seed": 42})
    return plan, paired, analysis


def test_subset_removes_unmatched_tasks_and_recomputes_task_weighted_means():
    plan, paired, _ = _fixture()
    source = copy.deepcopy(paired)
    result = subset_paired_effects(plan, paired, [0, 1])
    assert {row["task_id"] for row in result["paired_rows"]} == {0, 1}
    assert {row["task_id"] for row in result["protocol"]["eval_cases"]} == {0, 1}
    assert result["num_shared_cases"] == 3
    effect = next(row for row in result["feature_effects"] if row["feature_id"] == 10)
    assert effect["num_tasks"] == 2 and effect["num_valid_pairs"] == 3
    assert effect["drop"] == -.25  # mean(.5, -1), not the episode-weighted 0.
    assert effect["delta_sr"] == .25
    assert effect["sr_raw"] == .5 and effect["sr_edit"] == .75
    assert effect["outcome_switch_rate"] == .75
    assert effect["n_success_to_failure"] == effect["n_failure_to_success"] == 1
    assert effect["n_success_to_success"] == 1 and effect["n_failure_to_failure"] == 0
    assert [row["drop"] for row in effect["task_effects"]] == [.5, -1.]
    assert paired == source


def test_summary_means_are_recomputed_from_rows_not_copied():
    plan, paired, _ = _fixture()
    paired["feature_effects"][0].update(drop=999, sr_raw=999, task_effects=[])
    result = subset_paired_effects(plan, paired, [0, 1])
    assert result["feature_effects"][0]["drop"] == -.25


@pytest.mark.parametrize("mutation", ["one_missing", "all_missing", "duplicate", "wrong_drop", "baseline", "extra_task", "missing_feature"])
def test_invalid_or_missing_pairs_fail_even_outside_requested_group(mutation):
    plan, paired, _ = _fixture()
    row = next(row for row in paired["paired_rows"] if row["feature_id"] == 10 and row["task_id"] == 9)
    if mutation == "one_missing":
        paired["paired_rows"].remove(row)
    elif mutation == "all_missing":
        paired["paired_rows"] = [r for r in paired["paired_rows"] if r["task_id"] != 9]
    elif mutation == "duplicate":
        paired["paired_rows"].append(copy.deepcopy(row))
    elif mutation == "wrong_drop":
        row["drop"] = 999
    elif mutation == "baseline":
        row.update(raw_success=False, drop=-1)
    elif mutation == "extra_task":
        row["task_id"] = 20
    else:
        paired["paired_rows"] = [r for r in paired["paired_rows"] if r["feature_id"] != 12]
    with pytest.raises(ValueError):
        subset_paired_effects(plan, paired, [0, 1])


def test_frozen_topk_remains_fixed_when_another_feature_wins_subgroup():
    plan, paired, analysis = _fixture()
    snapshot = copy.deepcopy(plan)
    result = analyze_task_groups(plan, paired, analysis)
    assert plan == snapshot
    assert result["method_topk"] == plan["method_topk"]
    group = result["groups"]["followup_task2to9"]
    effects = {row["feature_id"]: row["drop"] for row in group["analysis"]["feature_effects"]}
    assert effects == {10: 0., 11: 0., 12: 1.}
    head = next(row for row in group["analysis"]["predictor_comparison"] if row["predictor"] == "head")
    assert head["topk_feature_ids"] == [10] and head["topk_mean_drop"] == 0.
    assert group["confirmatory"] is False
    assert group["evaluation_label_history"] == "not_independently_verified"
    assert group["declared_evaluation_labels_previously_used"] is None
    assert group["interpretation"] == "followup_not_confirmatory"


def test_missing_frozen_topk_cannot_trigger_automatic_reranking():
    plan, paired, _ = _fixture()
    del plan["method_topk"]["head"]
    _freeze(plan)
    with pytest.raises(ValueError, match="no subgroup reselection"):
        subset_paired_effects(plan, paired, [0, 1])


def test_tampered_plan_and_absent_tasks_fail():
    plan, paired, _ = _fixture()
    plan["top_k"] = 2
    with pytest.raises(ValueError, match="hash"):
        subset_paired_effects(plan, paired, [0, 1])
    plan, paired, _ = _fixture()
    with pytest.raises(ValueError, match="missing evaluation tasks"):
        subset_paired_effects(plan, paired, [10])


def _save_inputs(root):
    plan, paired, analysis = _fixture()
    files = {}
    for name, value in (("plan/rollout_plan.json", plan), ("analysis/paired_effects.json", paired),
                        ("analysis/analysis.json", analysis)):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(value, indent=2)
        path.write_text(text)
        files[path] = text
    return files


def test_files_are_separate_immutable_and_idempotent(tmp_path):
    originals = _save_inputs(tmp_path)
    first = write_task_groups(tmp_path)
    assert first == write_task_groups(tmp_path)
    assert all(path.read_text() == text for path, text in originals.items())
    output = Path(first["analysis"])
    payload = json.loads(output.read_text())
    assert set(payload["source_artifact_sha256"]) == {str(path.relative_to(tmp_path)) for path in originals}
    assert "followup_not_confirmatory" in Path(first["report"]).read_text()
    output.write_text("existing unrelated artifact")
    with pytest.raises(FileExistsError, match="immutable"):
        write_task_groups(tmp_path)
    assert output.read_text() == "existing unrelated artifact"
    assert all(path.read_text() == text for path, text in originals.items())


def test_cli_reads_root_without_loading_gpu_or_simulator(tmp_path):
    _save_inputs(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts/openvla/headgroups.py"
    result = subprocess.run([sys.executable, str(script), "--root", str(tmp_path)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["groups"]["followup_task2to9"]["num_shared_cases"] == 9
    check = subprocess.run([sys.executable, "-c", "import sys; import scripts.openvla.headgroups; assert not {'torch', 'tensorflow', 'libero', 'huggingface_hub'} & set(sys.modules)"],
                           cwd=script.parents[2], text=True, capture_output=True)
    assert check.returncode == 0, check.stderr


def test_cli_missing_artifacts_is_blocked(tmp_path):
    from scripts.openvla.headgroups import main
    assert main(["--root", str(tmp_path)]) == 2
