"""Describe fixed task subgroups without selecting another feature panel.

Consumes completed analysis artifacts only. This is a CPU-only follow-up of a
frozen plan, not evidence that evaluation labels were previously unseen.
"""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .candidates import _id
from .provenance import atomic_write_text, fingerprint, read_json, sha256_file
from .results import analyze_prediction


TASK_GROUPS = {
    "exploratory_task01": [0, 1],
    "followup_task2to9": list(range(2, 10)),
}


def _case(row: Mapping[str, Any]) -> tuple[str, int, int, str]:
    suite, state = row.get("suite"), row.get("initial_state_sha256")
    if not isinstance(suite, str) or not suite or not isinstance(state, str) or not state:
        raise ValueError("Every paired case needs suite and initial_state_sha256")
    return suite, _id(row["task_id"]), _id(row["task_episode_idx"]), state.lower()


def _validate_panel(plan: Mapping[str, Any], paired: Mapping[str, Any]) -> list[int]:
    if plan.get("plan_hash") != fingerprint({k: v for k, v in plan.items() if k != "plan_hash"}):
        raise ValueError("Frozen plan hash is invalid")
    if plan.get("status") != "planned":
        raise ValueError("Task groups require an approved planned panel")
    features = [_id(i) for i in plan["feature_ids"]]
    if not features or len(features) != len(set(features)):
        raise ValueError("Frozen panel must contain distinct feature IDs")
    candidates = [_id(row["feature_id"]) for row in plan["candidates"]]
    summaries = [_id(row["feature_id"]) for row in paired["feature_effects"]]
    for name, ids in (("candidates", candidates), ("feature summaries", summaries),
                      ("paired feature IDs", [_id(i) for i in paired["feature_ids"]])):
        if len(ids) != len(features) or set(ids) != set(features):
            raise ValueError(f"{name} differ from the frozen feature panel")
    methods = plan["methods"]
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("Frozen plan requires distinct predictor methods")
    top_k = _id(plan["top_k"])
    for method in methods:
        ids = [_id(i) for i in plan.get("method_topk", {}).get(method, [])]
        if not top_k or len(ids) != top_k or len(set(ids)) != top_k or not set(ids) <= set(features):
            raise ValueError("Complete frozen method_topk is required; no subgroup reselection is allowed")
        if method not in plan.get("score_vectors", {}):
            raise ValueError("Frozen predictor scores are required")
    if not set(_id(i) for i in plan.get("audit_feature_ids", [])) <= set(features):
        raise ValueError("Audit feature IDs differ from the frozen feature panel")
    return sorted(features)


def subset_paired_effects(
    plan: Mapping[str, Any], paired: Mapping[str, Any], task_ids: Sequence[int],
) -> dict[str, Any]:
    """Validate full shared pairs, then recompute equal-task subgroup effects.

Never slice precomputed whole-suite means: each task mean, transition count and
case count is derived again from the selected paired episode rows.
"""
    features = _validate_panel(plan, paired)
    tasks = [_id(task) for task in task_ids]
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("Task subgroup needs distinct nonempty task IDs")
    expected = [_case(row) for row in plan["eval_cases"]]
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("Frozen evaluation cases must be nonempty and distinct")
    if not set(tasks) <= {case[1] for case in expected}:
        raise ValueError("Requested task subgroup is missing evaluation tasks")
    protocol_id = paired["protocol_id"]
    if paired.get("protocol", {}).get("protocol_id", protocol_id) != protocol_id:
        raise ValueError("Paired protocol IDs disagree")
    rows_by_feature: dict[int, dict[tuple[Any, ...], dict]] = {feature: {} for feature in features}
    raw_outcomes: dict[tuple[Any, ...], bool] = {}
    for original in paired["paired_rows"]:
        row = dict(original)
        feature = _id(row["feature_id"])
        if feature not in rows_by_feature:
            raise ValueError("Paired row feature is outside the frozen panel")
        case = _case(row)
        key = (*case, _id(row["seed"]), row["protocol_id"])
        if row["protocol_id"] != protocol_id:
            raise ValueError("Paired row uses a different protocol")
        if key in rows_by_feature[feature]:
            raise ValueError("Duplicate paired case")
        if not isinstance(row.get("raw_success"), bool) or not isinstance(row.get("edit_success"), bool):
            raise ValueError("Paired success outcomes must be booleans")
        if row.get("drop") != int(row["raw_success"]) - int(row["edit_success"]):
            raise ValueError("Paired drop disagrees with success outcomes")
        if key in raw_outcomes and raw_outcomes[key] != row["raw_success"]:
            raise ValueError("Features must share the same raw baseline outcomes")
        raw_outcomes[key] = row["raw_success"]
        rows_by_feature[feature][key] = row
    shared_keys = set(rows_by_feature[features[0]])
    for rows in rows_by_feature.values():
        if set(rows) != shared_keys:
            raise ValueError("Every feature must share exactly the same paired case universe")
        cases = [key[:4] for key in rows]
        if len(cases) != len(expected) or set(cases) != set(expected):
            raise ValueError("Paired cases differ from the frozen evaluation manifest")
        if len({case[:3] for case in cases}) != len(cases) or len({(c[0], c[1], c[3]) for c in cases}) != len(cases):
            raise ValueError("Duplicate task/trial or initial-state case")
    if paired.get("num_shared_cases") != len(shared_keys):
        raise ValueError("Recorded shared case count disagrees with paired rows")
    selected_keys = sorted(key for key in shared_keys if key[1] in tasks)
    protocol = copy.deepcopy(paired.get("protocol", {}))
    protocol["eval_cases"] = [copy.deepcopy(row) for row in plan["eval_cases"] if _id(row["task_id"]) in tasks]
    protocol["expected_feature_ids"] = features
    protocol["task_group_ids"] = sorted(tasks)
    effects, selected_rows = [], []
    transitions = {(True, False): "n_success_to_failure", (False, True): "n_failure_to_success",
                   (True, True): "n_success_to_success", (False, False): "n_failure_to_failure"}
    for feature in features:
        by_task: dict[tuple[str, int], list[dict]] = defaultdict(list)
        counts = {name: 0 for name in transitions.values()}
        for key in selected_keys:
            row = copy.deepcopy(rows_by_feature[feature][key])
            by_task[(row["suite"], row["task_id"])].append(row)
            counts[transitions[(row["raw_success"], row["edit_success"])]] += 1
            selected_rows.append(row)
        task_effects = []
        for (suite, task), rows in sorted(by_task.items()):
            n = len(rows)
            task_effects.append({
                "suite": suite, "task_id": task, "num_valid_pairs": n,
                "sr_raw": sum(row["raw_success"] for row in rows) / n,
                "sr_edit": sum(row["edit_success"] for row in rows) / n,
                "drop": sum(row["drop"] for row in rows) / n,
                "outcome_switch_rate": sum(row["raw_success"] != row["edit_success"] for row in rows) / n,
            })
        means = {name: sum(row[name] for row in task_effects) / len(task_effects)
                 for name in ("sr_raw", "sr_edit", "drop", "outcome_switch_rate")}
        effects.append({"feature_id": feature, "alpha": 0.0, "num_tasks": len(task_effects),
                        "num_valid_pairs": len(selected_keys), **means, "delta_sr": -means["drop"],
                        **counts, "protocol_id": protocol_id, "task_effects": task_effects,
                        "selection_eval_overlap": protocol.get("selection_eval_overlap"),
                        "evaluation_labels_previously_used": protocol.get("evaluation_labels_previously_used")})
    return {"schema_version": "output_head_paired_effects_v1", "synthetic": bool(paired.get("synthetic", False)),
            "protocol_id": protocol_id, "feature_effects": effects, "paired_rows": selected_rows,
            "feature_ids": features, "num_shared_cases": len(selected_keys),
            "population_scope": "fixed_evaluated_panel", "protocol": protocol}


def analyze_task_groups(plan: dict, paired: dict, original_analysis: dict) -> dict:
    """Report the two preset groups with unchanged full-suite predictor ranks."""
    if {case["task_id"] for case in plan["eval_cases"]} != set(range(10)):
        raise ValueError("Full10 subgroup analysis requires all task IDs 0 through 9")
    if original_analysis.get("protocol_id") != paired.get("protocol_id"):
        raise ValueError("Original analysis and paired outcomes have different protocols")
    if {row["feature_id"] for row in original_analysis["feature_effects"]} != set(plan["feature_ids"]):
        raise ValueError("Original analysis differs from the frozen feature panel")
    uncertainty = original_analysis["uncertainty"]
    config = {name: uncertainty[name] for name in ("bootstrap_replicates", "bootstrap_seed", "confidence_level")}
    groups = {}
    for name, tasks in TASK_GROUPS.items():
        subset = subset_paired_effects(plan, paired, tasks)
        result = analyze_prediction(plan, subset, config)
        groups[name] = {
            "task_ids": tasks, "num_shared_cases": subset["num_shared_cases"],
            "interpretation": "exploratory" if tasks == [0, 1] else "followup_not_confirmatory",
            "confirmatory": False,
            "task_history": "prior_pilot_outcomes_examined" if tasks == [0, 1] else "not_used_in_reported_task01_pilot",
            "evaluation_label_history": "not_independently_verified",
            "declared_evaluation_labels_previously_used": plan.get("evaluation_labels_previously_used"),
            "analysis": result,
        }
    return {
        "schema_version": "output_head_task_groups_v1", "status": "analyzed",
        "plan_hash": plan["plan_hash"], "protocol_id": paired["protocol_id"],
        "synthetic": bool(paired.get("synthetic", False)),
        "selection": "unchanged_full_suite_plan_no_subgroup_reselection",
        "method_topk": copy.deepcopy(plan["method_topk"]),
        "feature_ids": copy.deepcopy(plan["feature_ids"]),
        "limitations": [
            "Task 0/1 informed the earlier pilot and are reported as exploratory.",
            "Task 2-9 is a follow-up subgroup, not automatically confirmatory: prior evaluation-label use is not independently verified.",
            "Predictor scores, feature panel, and top-K remain frozen to the original full-suite plan; no held-out-task generalization is established.",
            "Uncertainty resamples initial states within fixed tasks, conditional on the evaluated feature panel and the existing frozen SAE.",
        ],
        "groups": groups,
    }


def _report(payload: dict) -> str:
    lines = ["# Output-head sensitivity: fixed task groups", "", f"Frozen plan: `{payload['plan_hash']}`.", "",
             "Rankings and top-K are unchanged from the full-suite plan. These are descriptive subgroup results.", ""]
    for name, group in payload["groups"].items():
        lines += [f"## {name}", "", f"Tasks: {group['task_ids']}. Paired cases per feature: {group['num_shared_cases']}.",
                  f"Interpretation: `{group['interpretation']}`. Evaluation-label history: not independently verified.", "",
                  "| Predictor | Population | Spearman | Top-K mean drop | CI* |",
                  "|---|---|---:|---:|---|"]
        for row in group["analysis"]["predictor_comparison"]:
            lines.append(f"| {row['predictor']} | {row['population_scope']} | {row['spearman']} | {row['topk_mean_drop']} | {row['topk_mean_drop_ci']} |")
        level = group["analysis"]["uncertainty"]["confidence_level"]
        lines += ["", f"*Configured confidence level: {level}; intervals are conditional on these fixed tasks and feature panel.", ""]
    lines += ["## Scope", "", *[f"- {value}" for value in payload["limitations"]], ""]
    return "\n".join(lines)


def write_task_groups(root: str | Path) -> dict:
    """Write separate immutable task_groups_v1 artifacts; preserve source bytes."""
    root = Path(root).expanduser().resolve()
    paths = {"plan": root / "plan/rollout_plan.json", "paired": root / "analysis/paired_effects.json",
             "analysis": root / "analysis/analysis.json"}
    payload = analyze_task_groups(*(read_json(paths[name]) for name in ("plan", "paired", "analysis")))
    payload["source_artifact_sha256"] = {str(path.relative_to(root)): sha256_file(path) for path in paths.values()}
    payload["analysis_code_sha256"] = {path.name: sha256_file(path) for path in (Path(__file__), Path(__file__).with_name("results.py"))}
    outputs = {root / "analysis/task_groups_v1.json": json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
               root / "analysis/task_groups_v1.md": _report(payload)}
    for path, content in outputs.items():
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Existing immutable task-group artifact differs: {path}")
    for path, content in outputs.items():
        if not path.exists():
            atomic_write_text(path, content)
    return {"status": "analyzed", "plan_hash": payload["plan_hash"],
            "report": str(root / "analysis/task_groups_v1.md"),
            "analysis": str(root / "analysis/task_groups_v1.json"),
            "groups": {name: {"task_ids": group["task_ids"], "num_shared_cases": group["num_shared_cases"],
                               "interpretation": group["interpretation"]} for name, group in payload["groups"].items()}}
