"""Synthetic CPU artifacts through real public research APIs; no simulator."""

import hashlib
import json

import pytest
import torch

from event_sae.research.output_head.candidates import build_evaluation_plan
from event_sae.research.output_head.head import OutputHeadBundle
from event_sae.research.output_head.readouts import build_readout_manifest, prepare_readout_cache
from event_sae.research.output_head.results import analyze_prediction, assemble_paired_effects
from event_sae.research.output_head.sensitivity import PRIMARY_METRIC, score_features


class EndToEndSAE(torch.nn.Module):
    def encode(self, values):
        return torch.cat([values.relu(), torch.zeros((len(values), 1), device=values.device)], dim=-1)

    def decode(self, values):
        directions = values.new_tensor([[.6, -.1], [.2, .8], [0., .3]])
        return values @ directions + values.new_tensor([.1, -.2])


def test_synthetic_index_to_lossless_cache_scores_frozen_plan_and_paired_analysis(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    records, values = [], []
    for episode in range(2):
        for step in range(2):
            for dimension in range(2):
                start = len(values)
                values.extend([[.4 + episode * .1, .1], [1., .3]] if dimension == 0 else [[.2, .8]])
                records.append({"layer_idx": 1, "shard_path": "dense.pt", "row_start": start,
                                "row_end": len(values), "episode_num": episode, "task_id": 0,
                                "task_episode_idx": episode, "step_in_episode": step,
                                "global_forward_idx": 5 * len(records)})
    torch.save(torch.tensor(values), source / "dense.pt")
    (source / "activation_index.jsonl").write_text("\n".join(json.dumps(row) for row in records))
    manifest = build_readout_manifest(source / "activation_index.jsonl", {"synthetic": True},
                                      {"batch_size": 1, "padding": "none", "use_cache": True,
                                       "action_dim": 2, "layer_idx": 1, "source_run_id": "synthetic-run",
                                       "suite": "toy", "evidence": "synthetic generation"})
    sae = EndToEndSAE().eval()
    cache = prepare_readout_cache(manifest, sae, tmp_path / "cache", sae_id="synthetic-sae-v1",
                                   hidden_dtype="float32")
    head = OutputHeadBundle(torch.tensor([.9, 1.1]),
                            torch.tensor([[1., -.2], [-.3, .7], [.2, -.9]]),
                            {"norm": {"implementation": "llama_rms_norm_v1", "eps": 1e-5},
                             "target_layer": 1, "num_layers": 2, "hidden_dtype": "float32"})
    scores = score_features(cache, head, sae, {"pair_batch_size": 2, "max_scored_pairs": 32})
    assert scores["scope"]["discovery_manifest_hash"] == manifest["discovery_manifest_hash"]
    assert scores["feature_scores"][2][PRIMARY_METRIC] == 0
    cases = [{"task_id": 0, "task_episode_idx": trial,
              "initial_state_sha256": hashlib.sha256(f"eval:{trial}".encode()).hexdigest()}
             for trial in range(10, 13)]
    evaluation = {"suite": "toy", "mode": "pilot", "cases": cases,
                  "discovery_eval_overlap": False, "evaluation_labels_previously_used": False}
    plan = build_evaluation_plan(scores, None, evaluation, {"max_total_rollouts": 20},
                                 selection={"methods": [PRIMARY_METRIC, "mean_readout_activation"],
                                            "top_k": 1, "random_audit_features": 2, "max_unique_features": 3})
    assert plan["execute"] is False and plan["synthetic"] is True
    assert plan["feature_ids"] == [0, 1]
    assert plan["total_rollouts"] == 12  # raw + two independent edits + identity, three cases
    raw = [{**case, "success": success, "caught_exception": None, "num_actions": 1}
           for case, success in zip(cases, [True, False, True])]
    edited = {0: [{**row, "success": success} for row, success in zip(raw, [False, False, True])],
              1: [{**row, "success": success} for row, success in zip(raw, [True, True, True])]}
    protocol = {"suite": "toy", "seed": 0, "synthetic": True, "protocol_id": "synthetic-paired",
                "selection_eval_overlap": False, "evaluation_labels_previously_used": False,
                "eval_cases": cases}
    paired = assemble_paired_effects(raw, edited, protocol)
    effects = {row["feature_id"]: row for row in paired["feature_effects"]}
    assert effects[0]["drop"] == pytest.approx(1 / 3)
    assert effects[1]["drop"] == pytest.approx(-1 / 3)
    report = analyze_prediction(scores, paired, {"plan": plan, "bootstrap_replicates": 30, "bootstrap_seed": 5})
    assert report["synthetic"] is True
    assert len(report["feature_effects"]) == 2
    assert {row["population_scope"] for row in report["predictor_comparison"]} == {"topk_union", "audit_sample"}
    # Whole research outputs must serialize without model/tensor placeholders.
    json.dumps({"plan": plan, "scores": scores, "analysis": report}, allow_nan=False)
