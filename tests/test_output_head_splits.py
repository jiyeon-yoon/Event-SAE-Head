import copy
import hashlib

import pytest

from event_sae.research.output_head.splits import (
    _digest,
    build_split_manifest,
    validate_selection_split,
    validate_split_manifest,
)


def episode_source(tasks=2, trials=6):
    return {
        "synthetic": True,
        "initial_state_hash_provenance": "synthetic_fixture",
        "episodes": [
            {"suite": "toy", "task_id": task, "task_episode_idx": trial,
             "episode_num": task * trials + trial,
             "source_run_id": "toy-run",
             "initial_state_sha256": hashlib.sha256(f"{task}:{trial}".encode()).hexdigest()}
            for task in range(tasks) for trial in range(trials)
        ],
    }


def spec(**overrides):
    return {"discovery_per_task": 3, "validation_per_task": 1,
            "evaluation_per_task": 2, "split_seed": 11,
            "frozen_before_pilot": True,
            "evaluation_labels_previously_used": False, **overrides}


def test_splits_are_reproducible_from_task_original_ids_not_input_order():
    source = episode_source()
    result = build_split_manifest(source, spec())
    source["episodes"].reverse()
    assert result == build_split_manifest(source, spec())
    assert validate_split_manifest(result, require_confirmatory=True)["num_episodes"] == 12
    keys = []
    for name, expected in [("discovery", 6), ("validation", 2), ("evaluation", 4)]:
        assert len(result["splits"][name]) == expected
        keys.extend((r["task_id"], r["task_episode_idx"]) for r in result["splits"][name])
    assert len(set(keys)) == 12


@pytest.mark.parametrize("mutation", ["unknown_trial", "duplicate_trial", "duplicate_state", "unknown_hash", "unknown_provenance"])
def test_split_rejects_ambiguous_original_states(mutation):
    source = episode_source()
    if mutation == "unknown_trial":
        source["episodes"][0]["task_episode_idx"] = None
    elif mutation == "duplicate_trial":
        source["episodes"][1]["task_episode_idx"] = 0
    elif mutation == "duplicate_state":
        source["episodes"][1]["initial_state_sha256"] = source["episodes"][0]["initial_state_sha256"]
    elif mutation == "unknown_hash":
        source["episodes"][0]["initial_state_sha256"] = None
    else:
        source.pop("initial_state_hash_provenance")
    with pytest.raises(ValueError):
        build_split_manifest(source, spec())


def test_counts_cannot_silently_drop_source_states():
    with pytest.raises(ValueError, match="split counts require"):
        build_split_manifest(episode_source(), spec(discovery_per_task=2))


@pytest.mark.parametrize("history", [None, True])
def test_unknown_or_used_evaluation_labels_are_not_confirmatory(history):
    result = build_split_manifest(episode_source(), spec(evaluation_labels_previously_used=history))
    assert result["evaluation_labels_previously_used"] is history
    validate_split_manifest(result)
    with pytest.raises(ValueError, match="label history"):
        validate_split_manifest(result, require_confirmatory=True)


def test_late_split_freeze_does_not_qualify():
    result = build_split_manifest(episode_source(), spec(frozen_before_pilot=False))
    with pytest.raises(ValueError, match="before pilot"):
        validate_split_manifest(result, require_confirmatory=True)


@pytest.mark.parametrize("explicit_null", [False, True])
def test_unknown_freeze_history_is_not_invented(explicit_null):
    settings = spec()
    if explicit_null:
        settings["frozen_before_pilot"] = None
    else:
        settings.pop("frozen_before_pilot")
    result = build_split_manifest(episode_source(), settings)
    assert result["frozen_before_pilot"] is None
    validate_split_manifest(result)
    with pytest.raises(ValueError, match="before pilot"):
        validate_split_manifest(result, require_confirmatory=True)


@pytest.mark.parametrize("value", [0, 1, "true", "false", [], {}])
def test_freeze_history_requires_boolean_or_null(value):
    with pytest.raises(ValueError, match="frozen_before_pilot"):
        build_split_manifest(episode_source(), spec(frozen_before_pilot=value))


def test_hash_detects_evaluation_reselection():
    result = build_split_manifest(episode_source(), spec())
    result["splits"]["evaluation"][0]["task_episode_idx"] = 999
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_split_manifest(result)


def test_state_overlap_is_rejected_even_with_updated_hash():
    result = build_split_manifest(episode_source(), spec())
    result["splits"]["evaluation"][0] = copy.deepcopy(result["splits"]["discovery"][0])
    result["manifest_hash"] = _digest({k: v for k, v in result.items() if k != "manifest_hash"})
    with pytest.raises(ValueError, match="overlaps"):
        validate_split_manifest(result)


def test_pilot_and_scoring_cannot_read_sealed_evaluation_states():
    result = build_split_manifest(episode_source(), spec())
    validate_selection_split(result, result["splits"]["discovery"], stage="score")
    validate_selection_split(result, result["splits"]["validation"], stage="pilot")
    for stage in ("pilot", "score", "validation"):
        with pytest.raises(ValueError, match="forbidden split"):
            validate_selection_split(result, result["splits"]["evaluation"], stage=stage)
    with pytest.raises(ValueError, match="method selection"):
        validate_selection_split(result, result["splits"]["evaluation"], stage="evaluation", labels_used=True)


def test_per_episode_label_history_overrides_global_declaration():
    source = episode_source()
    for row in source["episodes"]:
        row["evaluation_labels_previously_used"] = True
    result = build_split_manifest(source, spec())
    assert result["evaluation_labels_previously_used"] is True
