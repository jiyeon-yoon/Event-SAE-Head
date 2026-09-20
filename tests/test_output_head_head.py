"""Synthetic CPU checks; none of these establish real OpenVLA parity."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from event_sae.research.output_head.head import (
    EDIT_PARITY_CHECKS, ExactRMSNorm, OutputHeadBundle, compare_logits,
    decode_action_bins, export_local_output_head, load_output_head,
    validate_parity_report,
)
from event_sae.research.output_head.provenance import fingerprint


def _metadata(**updates):
    return {"model_revision": "toy-model", "code_revision": "toy-code", "synthetic": True,
            "target_layer": 1, "num_layers": 2, "hidden_dtype": "float32",
            "norm": {"implementation": "llama_rms_norm_v1", "eps": 1e-5}, **updates}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_exact_rmsnorm_preserves_cast_and_weight_multiply_order(dtype):
    weight = torch.tensor([0.9, 1.1, 0.7], dtype=dtype)
    hidden = torch.tensor([[0.31, -2.09, 1.03], [0, 0, 0]], dtype=dtype)
    head_weight = torch.tensor([[1, 0, -1], [-0.3, 0.2, 0.1]], dtype=dtype)
    bundle = OutputHeadBundle(weight, head_weight, _metadata(hidden_dtype=str(dtype)))
    normalized = hidden.float() * torch.rsqrt(hidden.float().square().mean(-1, keepdim=True) + 1e-5)
    expected = torch.nn.functional.linear(weight * normalized.to(dtype), head_weight)
    torch.testing.assert_close(bundle(hidden.float()), expected, rtol=0, atol=0)
    assert bundle(hidden).dtype == dtype


def test_intermediate_layer_and_invalid_dimensions_are_rejected():
    with pytest.raises(ValueError, match="final decoder"):
        OutputHeadBundle(torch.ones(3), torch.ones(4, 3), _metadata(target_layer=0))
    with pytest.raises(ValueError, match="widths"):
        OutputHeadBundle(torch.ones(2), torch.ones(4, 3), _metadata())
    with pytest.raises(ValueError, match="norm implementation"):
        OutputHeadBundle(torch.ones(3), torch.ones(4, 3), _metadata(norm={"implementation": "unknown"}))


def test_head_does_not_silently_recast_norm_output():
    bundle = OutputHeadBundle(torch.ones(3), torch.ones(4, 3, dtype=torch.bfloat16),
                              _metadata(hidden_dtype="bfloat16"))
    with pytest.raises(ValueError, match="dtype mismatch"):
        bundle(torch.ones(2, 3))


def _snapshot(tmp_path, *, omit_head=False):
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"text_config": {
        "num_hidden_layers": 2, "rms_norm_eps": 1e-5, "hidden_size": 3, "vocab_size": 5}}))
    tensors = {"language_model.model.norm.weight": torch.tensor([1., 2., 3.])}
    if not omit_head:
        tensors["language_model.lm_head.weight"] = torch.arange(15).reshape(5, 3).float()
    save_file(tensors, root / "model-part-01.safetensors")
    # Unrelated shard should not be opened (it deliberately isn't a tensor file).
    (root / "model-part-02.safetensors").write_bytes(b"unrelated")
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        **{key: "model-part-01.safetensors" for key in tensors},
        "vision_backbone.weight": "model-part-02.safetensors"}}))
    return root


def test_local_export_only_reads_named_shards_and_hashes_bundle(tmp_path):
    source = _snapshot(tmp_path)
    out = tmp_path / "head"
    manifest = export_local_output_head(source, out, metadata=_metadata(),
                                       norm_spec=_metadata()["norm"])
    assert manifest["runtime_parity_status"] == "not_run"
    assert manifest["synthetic"] is True
    assert "model-part-02.safetensors" not in manifest["source_hashes"]
    bundle = load_output_head(out)
    hidden = torch.tensor([[0.5, 1.2, -0.4]])
    expected = torch.nn.functional.linear(ExactRMSNorm(torch.tensor([1., 2., 3.]), 1e-5)(hidden),
                                          torch.arange(15).reshape(5, 3).float())
    torch.testing.assert_close(bundle(hidden), expected)
    with pytest.raises(FileExistsError):
        export_local_output_head(source, out, metadata=_metadata(), norm_spec=_metadata()["norm"])
    with (out / "output_head.safetensors").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="weights hash"):
        load_output_head(out)


def test_unknown_tied_weights_or_unverified_norm_are_not_guessed(tmp_path):
    source = _snapshot(tmp_path, omit_head=True)
    with pytest.raises(ValueError, match="norm_spec"):
        export_local_output_head(source, tmp_path / "out", metadata=_metadata())
    with pytest.raises(KeyError, match="missing"):
        export_local_output_head(source, tmp_path / "out", metadata=_metadata(), norm_spec=_metadata()["norm"])


def test_snapshot_casts_only_explicitly_declared_runtime_weight_dtypes(tmp_path):
    source = _snapshot(tmp_path)
    metadata = _metadata(hidden_dtype="bfloat16", norm_dtype="bfloat16", head_dtype="bfloat16")
    manifest = export_local_output_head(source, tmp_path / "head", metadata=metadata,
                                       norm_spec=_metadata()["norm"])
    assert manifest["source_tensor_dtypes"]["head_weight"] == "float32"
    bundle = load_output_head(tmp_path / "head")
    assert bundle.norm.weight.dtype == bundle.head_weight.dtype == torch.bfloat16
    assert bundle(torch.ones(1, 3)).dtype == torch.bfloat16


def test_snapshot_config_mismatch_and_input_output_overlap(tmp_path):
    source = _snapshot(tmp_path)
    with pytest.raises(ValueError, match="epsilon disagrees"):
        export_local_output_head(source, tmp_path / "out", metadata=_metadata(),
                                 norm_spec={"implementation": "llama_rms_norm_v1", "eps": 1e-3})
    with pytest.raises(ValueError, match="outside the input"):
        export_local_output_head(source, source / "output", metadata=_metadata(), norm_spec=_metadata()["norm"])


def test_export_from_existing_runtime_object(tmp_path):
    norm = ExactRMSNorm(torch.ones(3), 1e-5)
    head = torch.nn.Linear(3, 5)
    model = SimpleNamespace(language_model=SimpleNamespace(
        model=SimpleNamespace(norm=norm, layers=[object(), object()]), lm_head=head))
    manifest = export_local_output_head(model, tmp_path / "bundle", metadata=_metadata())
    assert manifest["source_kind"] == "runtime_object"
    loaded = load_output_head(tmp_path / "bundle")
    h = torch.tensor([[1., -2., 3.]])
    torch.testing.assert_close(loaded(h), head(norm(h)))


def test_action_decoding_effective_vocab_padded_ids_and_clipping():
    metadata = {"verified": True, "rule": "effective_vocab_minus_token_minus_one_clip",
                "effective_vocab_size": 8, "model_vocab_size": 10,
                "bin_centers": [-0.75, -0.25, 0.25, 0.75]}
    actual = decode_action_bins(torch.tensor([0, 4, 5, 6, 7, 8, 9]), metadata)
    torch.testing.assert_close(actual, torch.tensor([.75, .75, .25, -.25, -.75, -.75, -.75]))
    with pytest.raises(ValueError, match="not been verified"):
        decode_action_bins(torch.tensor([1]), {**metadata, "verified": False})
    with pytest.raises(ValueError, match="outside model"):
        decode_action_bins(torch.tensor([10]), metadata)


def test_parity_comparison_keeps_fixed_tolerances_and_rejects_nonfinite():
    base = torch.tensor([[0., 1.]])
    assert compare_logits(base, base.clone(), atol=0, rtol=0)["status"] == "passed"
    report = compare_logits(base, base.flip(-1), atol=2, rtol=0)
    assert report["status"] == "failed"  # numerical allowance doesn't erase token flips
    assert report["argmax_mismatch_rate"] == 1
    assert compare_logits(base, base * float("nan"), atol=0, rtol=0)["finite"] is False


def test_parity_certificate_binds_identity_tolerances_and_required_checks():
    identity = {"head_hash": "head", "sae_hash": "sae", "atol": 1e-5,
                "mapping_version": "mapping", "edit_backend": "reference"}
    report = {"status": "passed", "synthetic": False, "identity": identity,
              "identity_fingerprint": fingerprint(identity),
              "checks": {name: {"status": "passed"} for name in EDIT_PARITY_CHECKS}}
    validate_parity_report(report, identity, EDIT_PARITY_CHECKS)
    for key in ("head_hash", "sae_hash", "mapping_version", "edit_backend", "atol"):
        with pytest.raises(ValueError, match="Stale parity"):
            validate_parity_report(report, {**identity, key: "changed"}, EDIT_PARITY_CHECKS)
    incomplete = copy.deepcopy(report)
    incomplete["checks"].pop("cached")
    with pytest.raises(ValueError, match="Missing or failed"):
        validate_parity_report(incomplete, identity, EDIT_PARITY_CHECKS)
    with pytest.raises(ValueError, match="Synthetic"):
        validate_parity_report({**report, "synthetic": True}, identity, EDIT_PARITY_CHECKS)
