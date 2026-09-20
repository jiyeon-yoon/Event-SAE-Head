"""Synthetic paired outcomes including improvements, missing cases and ties."""

from __future__ import annotations

import copy

import pytest

from event_sae.research.output_head.results import analyze_prediction, assemble_paired_effects, spearman_correlation


def _row(task, trial, success):
    return {"task_id": task, "task_episode_idx": trial, "initial_state_sha256": f"{task}-{trial}",
            "success": success, "caught_exception": None, "num_actions": 1}


def _protocol():
    return {"suite": "synthetic_suite", "seed": 7, "protocol_id": "synthetic-protocol", "synthetic": True,
            "selection_eval_overlap": False, "evaluation_labels_previously_used": False}


def _paired():
    raw = [_row(0, 0, True), _row(0, 1, True), _row(1, 0, False)]
    edits = {
        0: [_row(0, 0, False), _row(0, 1, True), _row(1, 0, True)],
        1: [_row(0, 0, False), _row(0, 1, True), _row(1, 0, True)],
        2: [_row(0, 0, True), _row(0, 1, True), _row(1, 0, False)],
    }
    return assemble_paired_effects(raw, edits, _protocol())


def _plan():
    return {"methods": ["head", "same", "different"], "top_k": 1,
            "method_topk": {"head": [0], "same": [1], "different": [2]},
            "score_vectors": {"head": {0: 3., 1: 2., 2: 1.}, "same": {0: 2., 1: 3., 2: 1.},
                              "different": {0: 1., 1: 2., 2: 3.}},
            "audit_feature_ids": [0, 2], "feature_universe": [0, 1, 2]}


def test_equal_task_weights_signed_drop_and_switch_counts():
    paired = _paired()
    feature = paired["feature_effects"][0]
    assert feature["drop"] == pytest.approx(-0.25)  # mean(+.5 task0, -1 task1), not token/episode weighted
    assert feature["delta_sr"] == pytest.approx(0.25)
    assert feature["sr_raw"] == 0.5 and feature["sr_edit"] == 0.75
    assert feature["n_success_to_failure"] == 1 and feature["n_failure_to_success"] == 1
    assert feature["outcome_switch_rate"] == 0.75
    assert paired["synthetic"] is True


@pytest.mark.parametrize("mutation", ["hash", "seed", "missing", "duplicate", "exception", "string_success"])
def test_invalid_pairs_fail_explicitly(mutation):
    raw = [_row(0, 0, True), _row(0, 1, False)]
    edits = copy.deepcopy(raw)
    if mutation == "hash": edits[0]["initial_state_sha256"] = "wrong"
    elif mutation == "seed": edits[0]["seed"] = 999
    elif mutation == "missing": edits.pop()
    elif mutation == "duplicate": edits.append(copy.deepcopy(edits[0]))
    elif mutation == "exception": edits[0]["caught_exception"] = "RuntimeError"
    elif mutation == "string_success": edits[0]["success"] = "false"
    with pytest.raises(ValueError):
        assemble_paired_effects(raw, {0: edits}, _protocol())


def test_protocol_mismatch_and_wrong_primary_baseline():
    raw = [_row(0, 0, True)]
    with pytest.raises(ValueError, match="protocol mismatch"):
        assemble_paired_effects({"episodes": raw, "protocol_id": "a"}, {0: {"episodes": raw, "protocol_id": "b"}}, _protocol())
    with pytest.raises(ValueError, match="baseline must be raw"):
        assemble_paired_effects({"episodes": raw, "mode": "reconstruction"}, {0: raw}, _protocol())


def test_results_match_frozen_case_manifest_and_explicit_overlap():
    rows = [_row(0, 0, True)]
    protocol = {**_protocol(), "require_declared_selection_eval_overlap": True, "selection_eval_overlap": None}
    with pytest.raises(ValueError, match="Declare selection_eval_overlap"):
        assemble_paired_effects(rows, {0: rows}, protocol)
    protocol.update(selection_eval_overlap=False, eval_cases=[{**rows[0], "initial_state_sha256": "other"}])
    with pytest.raises(ValueError, match="frozen evaluation manifest"):
        assemble_paired_effects(rows, {0: rows}, protocol)


def test_repeated_state_with_different_trial_index_is_not_independent_pair():
    rows = [_row(0, 0, True), _row(0, 1, True)]
    rows[1]["initial_state_sha256"] = rows[0]["initial_state_sha256"]
    with pytest.raises(ValueError, match="Duplicate"):
        assemble_paired_effects(rows, {0: rows}, _protocol())


def test_spearman_ties_and_undefined_vectors():
    assert spearman_correlation([1, 2, 3], [3, 2, 1]) == -1
    assert spearman_correlation([1, 1, 2], [3, 3, 1]) == -1
    assert spearman_correlation([1, 1], [1, 2]) is None
    assert spearman_correlation([], []) is None
    with pytest.raises(ValueError, match="finite"):
        spearman_correlation([1, float("nan")], [1, 2])


def test_bootstrap_is_shared_and_reproducible_and_groups_are_distinct():
    plan, paired = _plan(), _paired()
    config = {"bootstrap_replicates": 100, "bootstrap_seed": 4, "plan": plan}
    report = analyze_prediction({}, paired, config)
    assert report == analyze_prediction({}, paired, config)
    contrast = next(row for row in report["topk_contrasts"] if row["first"] == "head" and row["second"] == "same")
    assert contrast["low"] == contrast["high"] == contrast["difference"] == 0
    assert contrast["status"] == "degenerate_bootstrap"
    rows = [row for row in report["predictor_comparison"] if row["predictor"] == "head"]
    assert {row["population_scope"]: row["evaluated_feature_count"] for row in rows} == {"topk_union": 3, "audit_sample": 2}
    assert report["feature_effects"][2]["bootstrap_status"] == "degenerate_bootstrap"


def test_missing_topk_not_replaced_by_best_available():
    plan = _plan()
    plan["method_topk"]["head"] = [9]
    plan["score_vectors"]["head"][9] = 99
    report = analyze_prediction({}, _paired(), {"plan": plan, "bootstrap_replicates": 20})
    row = next(row for row in report["predictor_comparison"] if row["predictor"] == "head")
    assert row["topk_mean_drop"] is None and row["topk_coverage"] == 0
    assert row["missing_topk_feature_ids"] == [9]


def test_all_features_must_share_case_universe_even_if_summary_claims_complete():
    paired = _paired()
    paired["paired_rows"].pop()
    with pytest.raises(ValueError, match="exactly the same"):
        analyze_prediction({}, paired, {"plan": _plan(), "bootstrap_replicates": 20})


def test_legacy_fingerprints_may_differ_but_shared_provenance_must_match(tmp_path):
    import json

    raw = [_row(0, 0, True)]
    path = tmp_path / "episodes.jsonl"
    path.write_text(json.dumps(raw[0])+"\n", encoding="utf-8")
    common = {"episode_results_path": str(path), "completed_rollouts": 1, "task_suite": "synthetic_suite", "seed": 7,
              "model_checkpoint": "fixture", "model_revision": "m", "model_code_revision": "r", "code": {"commit": "c", "dirty": False},
              "run_config": {"env": {"seed": 7}, "logging": {"root_dir": "/somewhere"}}}
    base = {**common, "mode": "raw", "protocol_fingerprint": "raw-feature-dependent"}
    edit = {**common, "mode": "intervention", "feature_id": 0, "alpha": 0.0, "protocol_fingerprint": "edit-feature-dependent"}
    assert assemble_paired_effects(base, {0: edit}, _protocol())["feature_effects"][0]["drop"] == 0
    edit["model_revision"] = "other"
    with pytest.raises(ValueError, match="protocol mismatch"):
        assemble_paired_effects(base, {0: edit}, _protocol())
