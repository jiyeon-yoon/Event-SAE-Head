"""Stage-1 plumbing tests use toy fixtures, not claimed scientific results."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tarfile

import pytest
import torch

from event_sae.research.head_followup import stage1
from event_sae.research.head_followup.comparison import compare_scores, OLD, HEAD, vector, comparison_markdown
from event_sae.research.head_followup.labeling import label_packet
from event_sae.research.output_head.provenance import fingerprint, sha256_file


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def mapping_fixture(tmp_path, monkeypatch):
    source = tmp_path / "source"
    save(source / "config.json", {"text_config": {"vocab_size": 32064}, "pad_to_multiple_of": 64, "n_action_bins": 256})
    save(source / "tokenizer.json", {"model": {"vocab": {str(i): i for i in range(32000)}}})
    (source / "modeling_prismatic.py").write_text("# fixture, not real source\n")
    hashes = {name: sha256_file(source / name) for name in stage1.SOURCE_HASHES}
    monkeypatch.setattr(stage1, "SOURCE_HASHES", hashes)
    head = {"model_revision": stage1.MODEL_REV, "code_revision": stage1.CODE_REV, "vocab_size": 32064,
            "weights_sha256": "a" * 64, "source_hashes": {"config.json": hashes["config.json"]},
            "processor_identity_evidence": {"files": {"tokenizer.json": hashes["tokenizer.json"]}}}
    return source, head


def test_mapping_endpoints_and_head_binding(tmp_path, monkeypatch):
    source, head = mapping_fixture(tmp_path, monkeypatch)
    mapping = stage1.build_action_mapping(source, head)
    assert mapping["action_token_ids"] == list(range(31744, 32000))
    stage1.validate_mapping(mapping, head)
    # Full LM size 32064 is NOT the original tokenizer size 32000.
    bad = deepcopy(mapping)
    bad["action_token_ids"] = list(range(31808, 32064))
    bad["mapping_hash"] = fingerprint({k: v for k, v in bad.items() if k != "mapping_hash"})
    with pytest.raises(ValueError, match="mapping"):
        stage1.validate_mapping(bad, head)
    with pytest.raises(ValueError, match="mapping"):
        stage1.validate_mapping(mapping, {**head, "weights_sha256": "b" * 64})


def test_mapping_refuses_changed_sources_or_tokenizer(tmp_path, monkeypatch):
    source, head = mapping_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="provenance"):
        stage1.build_action_mapping(source, {**head, "model_revision": "other"})
    (source / "modeling_prismatic.py").write_text("changed")
    with pytest.raises(ValueError, match="pinned"):
        stage1.build_action_mapping(source, head)


def comparison_fixture():
    scope = {"sae_sha256": "sae", "split_manifest_hash": "split", "discovery_manifest_hash": "discovery",
             "sample_manifest_hash": "sample"}
    old = {"synthetic": False, "scope": scope, "identity": {"model_revision": "model", "code_revision": "code"}, "feature_scores": [
        {"feature_id": i, "num_active_readouts": 1 if i < 4 else 0,
         **{key: float(5 - i) for key in OLD}} for i in range(5)]}
    plan = {"synthetic": False, "scope": scope, "feature_ids": [0, 1, 2], "eval_cases": [{"task_id": 0, "trial": 1}],
            "score_vectors": {"event_aligned": {str(i): float(5 - i) for i in range(5)}}}
    plan["plan_hash"] = fingerprint(plan)
    paired = {"synthetic": False, "protocol_id": "p", "protocol": {"sae_sha256": "sae", "model_revision": "model",
              "model_code_revision": "code", "eval_cases": plan["eval_cases"], "alpha": 0., "hook_start_step": 0}, "feature_effects": [
        {"feature_id": i, "drop": .1 * i, "protocol_id": "p"} for i in range(3)]}
    action = {"synthetic": False, "schema_version": "head_action_scores_v1", "scope": scope,
              "feature_scores": [{"feature_id": i, "action_conditional_kl": float(i),
                "action_mass_abs_change": float(i), "conditional_action_argmax_flip": .1 * i} for i in range(5)]}
    return old, plan, paired, action


def test_comparison_keeps_missing_outcomes_unknown_and_common_population():
    old, plan, paired, action = comparison_fixture()
    report = compare_scores(old, plan, paired, action=action)
    row = report["comparators"]["action_conditional_kl"]
    assert row["top3"] == [4, 3, 2]
    assert row["unmeasured_top3"] == [4, 3]
    assert row["top3_mean_drop"] is None
    assert row["spearman_with_head_common_active"] == -1
    assert report["common_active_feature_ids"] == [0, 1, 2, 3]
    assert report["common_measured_feature_ids"] == [0, 1, 2]
    assert "Action scores available: True" in comparison_markdown(report)


@pytest.mark.parametrize("field", ["sae_sha256", "split_manifest_hash", "discovery_manifest_hash", "sample_manifest_hash"])
def test_comparison_rejects_different_binding(field):
    old, plan, paired, action = comparison_fixture()
    action = deepcopy(action)
    action["scope"][field] = "wrong"
    with pytest.raises(ValueError):
        compare_scores(old, plan, paired, action=action)


def test_comparison_rejects_stale_plan_duplicate_partial_and_fake_scores():
    old, plan, paired, action = comparison_fixture()
    plan["feature_ids"].append(3)
    with pytest.raises(ValueError, match="plan"):
        compare_scores(old, plan, paired)
    old, plan, paired, action = comparison_fixture()
    action["feature_scores"].pop()
    with pytest.raises(ValueError, match="complete"):
        compare_scores(old, plan, paired, action=action)
    action["synthetic"] = True
    with pytest.raises(ValueError):
        compare_scores(old, plan, paired, action=action)
    with pytest.raises(ValueError, match="duplicate"):
        vector([{"feature_id": 1, "x": 1}, {"feature_id": 1, "x": 2}], "x")
    with pytest.raises(ValueError, match="nonfinite"):
        vector([{"feature_id": 1, "x": float("nan")}], "x")
    old, plan, paired, _ = comparison_fixture()
    paired["protocol"]["sae_sha256"] = "wrong"
    with pytest.raises(ValueError, match="outcome protocol"):
        compare_scores(old, plan, paired)


def test_partial_drvla_has_common_panel_not_silent_zero_imputation():
    old, plan, paired, _ = comparison_fixture()
    dr = {"schema_version": "drvla_recalibrated_scores_v1", "synthetic": False, "identity": old["scope"],
          "feature_scores": [{"feature_id": 1, "generality_decision": .2}, {"feature_id": 3, "generality_decision": .8}]}
    report = compare_scores(old, plan, paired, drvla=dr)
    assert report["common_measured_feature_ids"] == [1]
    assert report["comparators"]["generality_decision"]["top3_mean_drop"] is None
    assert report["comparators"][HEAD]["spearman_with_drop_common_measured"] is None


def test_label_packet_blinds_outcomes_and_never_invents_labels():
    statistics = {"synthetic": False, "identity": {"manifest_hash": "m"}, "feature_statistics": [
        {"feature_id": i, "status": "ok", "episode_coverage": 1, "mean_onset_count": 1,
         "mean_episode_max_activation": .7, "relative_run_length": .5} for i in range(40)]}
    cache = {"manifest": {"synthetic": False, "manifest_hash": "m", "episode_lengths": {"a": 2}},
             "records": [{"episode_uid": "a", "step_in_episode": t, "task_id": 0, "task_episode_idx": 5,
                          "source_run_id": "source", "action_dim_index": d} for t in range(2) for d in range(7)],
             "feature_ids": [torch.arange(40)] * 14, "feature_values": [torch.ones(40)] * 14}
    packet = label_packet(statistics, cache, excluded_feature_ids=[0, 1, 2], statistics_sha256="sha", count=30)
    assert len(packet["cards"]) == 30
    assert not set(packet["candidate_order"]) & {0, 1, 2}
    assert packet["labels"] == "not_provided"
    assert packet["cards"][0]["discovery_traces"][0]["values"] == pytest.approx([1, 1])
    assert "head_rank" not in json.dumps(packet)
    assert "success_rate" not in json.dumps(packet)
    assert packet == label_packet(statistics, cache, excluded_feature_ids=[0, 1, 2], statistics_sha256="sha", count=30)
    cache["manifest"]["manifest_hash"] = "wrong"
    with pytest.raises(ValueError, match="matching"):
        label_packet(statistics, cache, excluded_feature_ids=[], statistics_sha256="sha")


def test_restore_pinned_regular_members_only_and_no_clobber(tmp_path, monkeypatch):
    archive = tmp_path / "experiment.tar.gz"
    names = [stage1.PREFIX + name for name in ("head/output_head_manifest.json", "readouts/manifest.json",
        "scores/scores.json", "analysis/paired_effects.json", "plan/rollout_plan.json")]
    import hashlib
    payload = b"{}"
    entries = {name: {"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()} for name in names}
    with tarfile.open(archive, "w:gz") as tf:
        for name in names + ["../../never-extract"]:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            tf.addfile(member, io.BytesIO(payload))
    digest = sha256_file(archive)
    monkeypatch.setattr(stage1, "ARCHIVE_SHA", digest)
    manifest = tmp_path / "manifest.json"
    save(manifest, {"archive": {"sha256": digest, "size_bytes": archive.stat().st_size}, "files": entries})
    output = tmp_path / "restored"
    result = stage1.restore_stage1(archive, manifest, output)
    assert result["files"] == len(names)
    assert not (tmp_path / "never-extract").exists()
    assert (output / "stage1_restore.json").is_file()
    with pytest.raises(ValueError, match="new output"):
        stage1.restore_stage1(archive, manifest, output)


def test_action_cli_dry_run_never_requires_inputs():
    from scripts.openvla.headfollowup import main
    assert main(["action-score", "--config", "absent", "--reference-result", "absent",
                 "--mapping", "absent", "--output", "absent"]) == 0


def test_setup_separates_runtime_from_frozen_results_without_dense(tmp_path, monkeypatch):
    import yaml
    from event_sae.research.output_head.config import DEFAULTS, load_config
    root = tmp_path / "restored"
    (root / "config").mkdir(parents=True)
    (root / "config/head-full10.yaml").write_text(yaml.safe_dump(DEFAULTS))
    sae = tmp_path / "sae/ae.pt"
    sae.parent.mkdir()
    sae.write_bytes(b"fixture")
    save(sae.parent / "config.json", {})
    model = tmp_path / "model"
    save(model / "config.json", {})
    save(model / "model.safetensors.index.json", {})
    (model / "model-00001-of-00001.safetensors").write_bytes(b"fixture")
    monkeypatch.setattr(stage1, "stage1_audit", lambda *a: {"sae_checkpoint_present": True})
    monkeypatch.setattr(stage1, "sha256_file", lambda p: "18443083d320d2ad3c607431f307697639966ad92075bdfe031398557f3ba9d6"
                        if Path(p) == sae else stage1.SOURCE_HASHES["config.json"])
    output = tmp_path / "settings"
    result = stage1.setup_stage1(root, output, sae=sae, snapshot=model)
    cfg = load_config(result["action_config"])
    assert result["new_rollouts"] == 0
    assert result["temporal_config"] is None
    assert cfg["inputs"]["dense_dir"] is None
    assert Path(cfg["rollout"]["base_eval_config"]).is_file()
    assert cfg["output"]["root_dir"] == str(output / "runtime")
    assert cfg["rollout"]["execute"] is False
    assert str(root) not in cfg["output"]["root_dir"]
    with pytest.raises(ValueError, match="new output"):
        stage1.setup_stage1(root, output, sae=sae, snapshot=model)
    # A requested but missing dense index must fail before any config is written.
    with pytest.raises(ValueError, match="index"):
        stage1.setup_stage1(root, tmp_path / "bad-settings", sae=sae, snapshot=model, dense=tmp_path / "missing")
    assert not (tmp_path / "bad-settings").exists()
