"""Synthetic candidate contracts: strict population, stable selection and budgets."""

from __future__ import annotations

import copy

import pytest

from event_sae.research.output_head.candidates import build_evaluation_plan, load_ranking_scores


def _scores():
    return {"synthetic": True, "scope": {"task_ids": [0], "dictionary_size": 6, "discovery_manifest_hash": "discovery-a"},
            "feature_scores": [{"feature_id": i, "head_full_vocab_kl_fixed_prefix": float(5-i),
                                "mean_readout_activation": 1.0, "readout_activation_frequency": 1.0,
                                "num_active_readouts": 1} for i in range(6)]}


def _selection(**updates):
    return {"methods": ["head_full_vocab_kl_fixed_prefix", "mean_readout_activation"],
            "top_k": 2, "random_audit_features": 6, "random_seed": 3, "max_unique_features": 6, **updates}


def _eval():
    return {"suite": "synthetic_suite", "mode": "pilot", "cases": [
        {"task_id": 0, "task_episode_idx": i, "initial_state_sha256": f"state-{i}"} for i in range(2)]}


def test_dedup_ties_audit_overlap_and_identity_budget():
    plan = build_evaluation_plan(_scores(), None, _eval(), {"max_total_rollouts": 100}, selection=_selection())
    assert plan["method_topk"]["mean_readout_activation"] == [0, 1]
    assert plan["num_unique_features"] == 6
    assert plan["total_rollouts"] == (1+6+1)*2
    assert len(plan["audit_feature_ids"]) == 6
    assert plan["candidates"][0]["memberships"] == ["topk:head_full_vocab_kl_fixed_prefix", "topk:mean_readout_activation", "audit"]
    assert plan["execute"] is False and plan["synthetic"] is True
    repeat = build_evaluation_plan(_scores(), None, _eval(), {"max_total_rollouts": 100}, selection=_selection())
    assert plan == repeat


def test_budget_preserves_every_candidate_and_can_omit_identity_explicitly():
    plan = build_evaluation_plan(_scores(), None, _eval(), {"max_total_rollouts": 1, "identity_features": []},
                                 selection=_selection(max_unique_features=1))
    assert plan["status"] == "blocked_budget"
    assert len(plan["blocked_reasons"]) == 2
    assert len(plan["feature_ids"]) == 6
    assert plan["total_rollouts"] == 14


def test_audit_shortage_not_silently_reduced():
    scores = _scores()
    for row in scores["feature_scores"]:
        row["num_active_readouts"] = 0
    with pytest.raises(ValueError, match="Audit population smaller"):
        build_evaluation_plan(scores, None, _eval(), {}, selection=_selection())


def test_missing_feature_scores_not_imputed():
    scores = _scores()
    scores["feature_scores"][0]["head_full_vocab_kl_fixed_prefix"] = None
    with pytest.raises(ValueError, match="missing selected"):
        build_evaluation_plan(scores, None, _eval(), {}, selection=_selection())


def test_cross_task_or_discovery_comparison_rejected():
    scores = _scores()
    comparison = {"scores": {"event_aligned": {i: float(i) for i in range(6)}},
                  "scope": {**scores["scope"], "task_ids": [0, 1]}}
    settings = _selection(methods=["head_full_vocab_kl_fixed_prefix", "event_aligned"])
    with pytest.raises(ValueError, match="task_ids"):
        build_evaluation_plan(scores, comparison, _eval(), {}, selection=settings)
    comparison["scope"] = {**scores["scope"], "discovery_manifest_hash": "wrong"}
    with pytest.raises(ValueError, match="discovery_manifest_hash"):
        build_evaluation_plan(scores, comparison, _eval(), {}, selection=settings)


def test_confirmatory_requires_pre_pilot_frozen_disjoint_unseen_evaluation():
    manifest = {**_eval(), "mode": "confirmatory"}
    with pytest.raises(ValueError, match="pre-pilot"):
        build_evaluation_plan(_scores(), None, manifest, {}, selection=_selection())
    manifest.update(frozen_before_pilot=True, evaluation_labels_previously_used=False,
                    discovery_eval_overlap=False, split_manifest_hash="split-a")
    scores = _scores()
    with pytest.raises(ValueError, match="split_manifest_hash"):
        build_evaluation_plan(scores, None, manifest, {}, selection=_selection())
    scores["scope"]["split_manifest_hash"] = "split-a"
    assert build_evaluation_plan(scores, None, manifest, {}, selection=_selection())["eval_manifest"]["mode"] == "confirmatory"


def test_matrix_methods_are_optional_and_legacy_event_is_explicit():
    torch = pytest.importorskip("torch")
    payload = {"synthetic": True, "matrix_raw": torch.tensor([[1., 3.], [5., 1.]]),
               "row_keys": [{"task_id": 0, "episode_coverage": 1., "num_events": 1},
                            {"task_id": 1, "episode_coverage": 1., "num_events": 3}]}
    assert load_ranking_scores(payload, ["event_aligned"])["scores"]["event_aligned"] == {0: 3., 1: 2.}
    with pytest.raises(ValueError, match="matrix_window_mean"):
        load_ranking_scores(payload, ["window_mean"])
    legacy = copy.deepcopy(payload)
    legacy["matrix"] = legacy.pop("matrix_raw")
    with pytest.raises(ValueError, match="fallback"):
        load_ranking_scores(legacy, ["event_aligned"])
    assert load_ranking_scores(legacy, ["event_aligned"], legacy_event_verified=True)["provenance"]["population_scope"] == "legacy_reference"


def test_matrix_suite_aggregation_matches_original_weighting():
    torch = pytest.importorskip("torch")
    payload = {"matrix_raw": torch.tensor([[1., 3.], [5., 1.], [2., 2.]]),
               "matrix_window_mean": torch.tensor([[1., 3.], [5., 1.], [2., 2.]]),
               "matrix_task_mean": torch.tensor([[1., 3.], [5., 1.], [1., 3.]]),
               "row_keys": [{"task_id": 0, "episode_coverage": 1., "num_events": 1},
                            {"task_id": 1, "episode_coverage": 1., "num_events": 3},
                            {"task_id": 0, "episode_coverage": 1., "num_events": 2}],
               "selection_counts": {"task_timestep_counts": {"0": 2, "1": 6}}}
    result = load_ranking_scores(payload, ["event_aligned", "window_mean", "task_mean"])["scores"]
    assert result["event_aligned"][0] == pytest.approx(8/3)
    assert result["window_mean"][0] == pytest.approx(20/6)
    assert result["task_mean"] == {0: 4., 1: 1.5}
    payload["selection_counts"] = {}
    with pytest.raises(ValueError, match="timestep count"):
        load_ranking_scores(payload, ["task_mean"])


def test_eval_duplicate_state_and_unknown_metadata_rejected():
    manifest = _eval()
    manifest["cases"][1]["initial_state_sha256"] = "state-0"
    with pytest.raises(ValueError, match="Duplicate"):
        build_evaluation_plan(_scores(), None, manifest, {}, selection=_selection())
