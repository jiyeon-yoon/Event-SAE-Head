"""Toy CPU scoring fixtures; all outputs are explicitly synthetic."""

import io
from types import SimpleNamespace

import pytest
import torch

from event_sae.research.output_head.head import OutputHeadBundle
from event_sae.research.output_head.sensitivity import (
    PRIMARY_METRIC, aggregation_weights, edit_feature_reference, full_vocab_kl, score_features,
)


class ToySAE:
    def __init__(self, bias=True):
        self.direction = torch.tensor([[0.7, 0.2], [-0.1, 0.6], [0.4, -0.3]])
        self.bias = torch.tensor([0.11, -0.09]) if bias else torch.zeros(2)
        self.encode_calls = 0
        self.max_decode_rows = 0

    def encode(self, hidden):
        self.encode_calls += 1
        return torch.cat((hidden.clamp_min(0), torch.zeros(hidden.shape[0], 1)), dim=1)

    def decode(self, latents):
        self.max_decode_rows = max(self.max_decode_rows, len(latents))
        return latents @ self.direction.to(latents.device) + self.bias.to(latents.device)


def _head(dtype=torch.float32):
    return OutputHeadBundle(torch.tensor([.9, 1.1], dtype=dtype),
                            torch.tensor([[1., -.2], [-.3, .7], [.2, -.9]], dtype=dtype),
                            {"norm": {"implementation": "llama_rms_norm_v1", "eps": 1e-5},
                             "target_layer": 1, "num_layers": 2, "hidden_dtype": str(dtype)})


def _cache(sae=None):
    sae = sae or ToySAE()
    hidden = torch.tensor([[1., -.2], [-1., .7], [.4, .8], [-.3, -.2]])
    encoded = sae.encode(hidden)
    records = [{"task_id": 0, "episode_uid": "a", "step_in_episode": 0, "action_dim_index": 0},
               {"task_id": 0, "episode_uid": "a", "step_in_episode": 1, "action_dim_index": 0},
               {"task_id": 0, "episode_uid": "b", "step_in_episode": 0, "action_dim_index": 0},
               {"task_id": 1, "episode_uid": "c", "step_in_episode": 0, "action_dim_index": 0}]
    ids = [torch.nonzero(row, as_tuple=True)[0] for row in encoded]
    return {"hidden": hidden, "records": records, "dict_size": 3,
            "feature_ids": ids, "feature_values": [encoded[i, idx] for i, idx in enumerate(ids)],
            "encoder_grouping": "original_forward", "synthetic": True,
            "sample_manifest_hash": "toy-sample", "discovery_manifest_hash": "toy-discovery"}


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_reference_matches_legacy_arithmetic_with_bias_and_cast(bias, dtype):
    sae = ToySAE(bias=bias)
    hidden = torch.tensor([[[1.13, -.28], [-.31, .99]]], dtype=dtype)
    flat = hidden.reshape(-1, 2).float()
    z = sae.encode(flat)
    edited = z.clone()
    edited[:, 0] = 0
    expected = (flat + (sae.decode(edited) - sae.decode(z))).to(dtype).reshape_as(hidden)
    torch.testing.assert_close(edit_feature_reference(hidden, sae, 0, 0), expected, rtol=0, atol=0)
    torch.testing.assert_close(edit_feature_reference(hidden, sae, 0, 1), hidden, rtol=0, atol=0)
    torch.testing.assert_close(edit_feature_reference(hidden, sae, 2, 0), hidden, rtol=0, atol=0)


def test_reference_matches_actual_existing_hook(monkeypatch, tmp_path):
    import event_sae.openvla.intervene as original

    class Layer(torch.nn.Module):
        def forward(self, hidden):
            return (hidden,)

    sae = ToySAE()
    layer = Layer()
    model = SimpleNamespace(language_model=SimpleNamespace(
        model=SimpleNamespace(layers=[layer]), lm_head=torch.nn.Linear(2, 3)),
        _sae_hook_context={"episode_num": 1, "step_in_episode": 0})
    monkeypatch.setattr(original, "load_batch_topk_sae", lambda *args, **kwargs: (sae, {"trainer": {
        "submodule_name": "post_mlp_residual", "layer": 0, "activation_dim": 2, "dict_size": 3}}))
    hook = original.apply_resid_post_feature_perturb_hook(
        model=model, layer_idx=0, sae_checkpoint_path="synthetic", feature_idx=0,
        alpha=0, hook_start_step=0, run_dir=str(tmp_path), log_file=io.StringIO())
    hidden = torch.tensor([[[1., .1], [-.1, .3]]], dtype=torch.bfloat16)
    try:
        actual = layer(hidden)[0]
    finally:
        hook.remove()
    torch.testing.assert_close(actual, edit_feature_reference(hidden, sae, 0, 0), rtol=0, atol=0)


def test_original_group_latents_are_used_without_reencoding():
    class BatchDependent(ToySAE):
        def encode(self, hidden):
            return super().encode(hidden) * len(hidden)

    sae = BatchDependent()
    all_rows = torch.tensor([[10., 0.], [1., 2.]])
    original_z = sae.encode(all_rows)[-1:]
    before = sae.encode_calls
    edited = edit_feature_reference(all_rows[-1:], sae, 0, 0, encoded=original_z)
    assert sae.encode_calls == before
    assert not torch.allclose(edited, edit_feature_reference(all_rows[-1:], sae, 0, 0))


def test_full_vocab_kl_direction_identity_and_invalid_inputs():
    base, edit = torch.tensor([[2., 0., -1.]]), torch.tensor([[0., 1., 0.]])
    expected = (base.softmax(-1) * (base.log_softmax(-1) - edit.log_softmax(-1))).sum(-1)
    torch.testing.assert_close(full_vocab_kl(base, edit), expected)
    assert not torch.allclose(full_vocab_kl(base, edit), full_vocab_kl(edit, base))
    assert full_vocab_kl(base, base).item() == 0
    with pytest.raises(ValueError, match="NaN/Inf"):
        full_vocab_kl(base, edit * float("nan"))
    with pytest.raises(ValueError, match="FP32 or FP64"):
        full_vocab_kl(base, edit, reduction_dtype="bfloat16")
    # A full-vocab change can vanish when conditioning on a selected subset.
    original, changed = torch.tensor([[10., 2., 1.]]), torch.tensor([[0., 2., 1.]])
    assert full_vocab_kl(original, changed).item() > 1
    assert full_vocab_kl(original[:, 1:], changed[:, 1:]).item() == 0


def test_equal_hierarchical_weights_not_global_row_mean():
    weights, tasks = aggregation_weights(_cache()["records"])
    assert tasks == [0, 1]
    torch.testing.assert_close(weights, torch.tensor([.125, .125, .25, .5], dtype=torch.float64))
    assert weights.sum() == 1


def test_duplicate_or_incomplete_action_dimensions_fail():
    records = _cache()["records"]
    with pytest.raises(ValueError, match="Duplicate/incomplete"):
        aggregation_weights([records[0], records[0]])
    with pytest.raises(ValueError, match="Duplicate/incomplete"):
        aggregation_weights([{**records[0], "action_dim_index": 1}])


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_sparse_scoring_matches_dense_reference_and_keeps_inactive_denominator(batch_size):
    sae, head = ToySAE(), _head()
    cache = _cache(sae)
    before = sae.encode_calls
    result = score_features(cache, head, sae, {"pair_batch_size": batch_size})
    assert sae.encode_calls == before  # never silently re-encode readout subsets
    assert sae.max_decode_rows <= batch_size
    weights, _ = aggregation_weights(cache["records"])
    z = sae.encode(cache["hidden"])
    baseline = head(cache["hidden"])
    for fid, row in enumerate(result["feature_scores"]):
        edited = edit_feature_reference(cache["hidden"], sae, fid, 0, encoded=z)
        expected = (weights * full_vocab_kl(baseline, head(edited)).double()).sum()
        assert row[PRIMARY_METRIC] == pytest.approx(float(expected), abs=1e-6)
        assert row["mean_readout_activation"] == pytest.approx(float((weights * z[:, fid]).sum()))
        assert row["num_readouts"] == 4
    assert result["feature_scores"][0]["readout_activation_frequency"] == pytest.approx(.375)
    assert result["feature_scores"][2][PRIMARY_METRIC] == 0
    assert result["feature_scores"][2]["score_status"] == "inactive_in_sample"
    assert result["feature_scores"][2]["decoded_bin_change_rate"] is None
    assert result["synthetic"] is True
    assert result["diagnostics"]["executed_pairs"] == 4


def test_alpha_one_scoring_is_identity_but_retains_activation_statistics():
    result = score_features(_cache(), _head(), ToySAE(), {"alpha": 1.0})
    for row in result["feature_scores"]:
        assert row[PRIMARY_METRIC] == 0
        assert row["full_vocab_argmax_flip_rate"] == 0
        assert row["mean_edit_norm"] == 0
    assert result["feature_scores"][0]["mean_readout_activation"] > 0


def test_action_dimension_scores_average_back_to_suite_score():
    cache = _cache()
    # Two complete two-dimensional action queries, with distinct activations.
    cache["records"] = [{"task_id": index // 2, "episode_uid": f"episode-{index // 2}",
                         "step_in_episode": 0, "action_dim_index": index % 2} for index in range(4)]
    result = score_features(cache, _head(), ToySAE(), {})
    for row in result["feature_scores"]:
        dimensions = [part for part in result["feature_dimension_scores"] if part["feature_id"] == row["feature_id"]]
        assert len(dimensions) == 2
        assert sum(part[PRIMARY_METRIC] for part in dimensions) / 2 == pytest.approx(row[PRIMARY_METRIC])


def test_pair_budget_precedes_head_or_decode_work_and_subset_is_labeled():
    sae = ToySAE()
    cache = _cache(sae)
    with pytest.raises(ValueError, match="Pair budget exceeded"):
        score_features(cache, _head(), sae, {"max_scored_pairs": 1})
    assert sae.max_decode_rows == 0
    result = score_features(cache, _head(), sae, {"feature_universe": [2]})
    assert result["diagnostics"]["feature_universe"] == "screened_subset"
    assert result["diagnostics"]["executed_pairs"] == 0
    with pytest.raises(ValueError, match="working-memory budget"):
        score_features(cache, _head(), sae, {"max_working_memory_bytes": 1})


def test_real_cache_requires_runtime_gates_and_truncation_is_rejected():
    with pytest.raises(ValueError, match="Parity report has not passed"):
        score_features({**_cache(), "synthetic": False}, _head(), ToySAE(), {})
    with pytest.raises(ValueError, match="truncated"):
        score_features({**_cache(), "truncated": True}, _head(), ToySAE(), {})
    with pytest.raises(ValueError, match="encoder grouping"):
        score_features({**_cache(), "encoder_grouping": "unknown"}, _head(), ToySAE(), {})
