"""Local, dtype-preserving final RMSNorm/output-head bundles.

No Hugging Face, simulator, model download, or CUDA initialization occurs on
import. Exporting a snapshot reads only explicitly identified safetensors.
An exported bundle is *not* evidence that runtime parity has passed.
"""

from __future__ import annotations

import inspect
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .provenance import atomic_write_json, fingerprint, read_json, sha256_file

SCHEMA = "output_head_bundle_v1"
NORM_IMPLEMENTATION = "llama_rms_norm_v1"
HEAD_PARITY_CHECKS = ("baseline_logits", "head_batch_shape")
EDIT_PARITY_CHECKS = ("alpha1", "active_alpha0", "inactive", "prefill", "cached", "all_rows", "pair_batch_edit")
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16,
          "float64": torch.float64}


def resolve_dtype(value: str | torch.dtype) -> torch.dtype:
    """Resolve an explicit floating dtype without choosing a device."""
    if isinstance(value, torch.dtype):
        if value not in DTYPES.values():
            raise ValueError(f"Unsupported floating dtype: {value}")
        return value
    name = str(value).removeprefix("torch.")
    if name not in DTYPES:
        raise ValueError(f"Unsupported floating dtype: {value}")
    return DTYPES[name]


def dtype_name(value: torch.dtype) -> str:
    return str(resolve_dtype(value)).removeprefix("torch.")


class ExactRMSNorm(nn.Module):
    """Llama RMSNorm: FP32 variance, cast normalized hidden, then weight.

    Keep this multiplication/cast order; moving the weight multiplication
    into FP32 changes the BF16 reference computation.
    """

    def __init__(self, weight: torch.Tensor, eps: float):
        super().__init__()
        if weight.ndim != 1 or not weight.is_floating_point() or not torch.isfinite(weight).all():
            raise ValueError("RMSNorm weight must be a finite floating vector")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("RMSNorm epsilon must be finite and positive")
        self.register_buffer("weight", weight.detach().clone())
        self.variance_epsilon = float(eps)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        original_dtype = hidden.dtype
        values = hidden.to(torch.float32)
        variance = values.pow(2).mean(-1, keepdim=True)
        values = values * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * values.to(original_dtype)


class OutputHeadBundle(nn.Module):
    """Only the final norm and unembedding, plus their declared provenance."""

    def __init__(self, norm_weight: torch.Tensor, head_weight: torch.Tensor,
                 manifest: Mapping[str, Any], head_bias: torch.Tensor | None = None):
        super().__init__()
        self.manifest = dict(manifest)
        norm = self.manifest.get("norm", {})
        if norm.get("implementation") != NORM_IMPLEMENTATION:
            raise ValueError("Unsupported/unverified norm implementation")
        if head_weight.ndim != 2 or not head_weight.is_floating_point():
            raise ValueError("Head weight must have shape [vocabulary, hidden]")
        if not torch.isfinite(head_weight).all():
            raise ValueError("Head weight contains non-finite values")
        if norm_weight.numel() != head_weight.shape[1]:
            raise ValueError("Norm and head hidden widths differ")
        if head_bias is not None and (head_bias.shape != head_weight.shape[:1]
                                     or head_bias.dtype != head_weight.dtype
                                     or not torch.isfinite(head_bias).all()):
            raise ValueError("Head bias must match vocabulary, dtype, and be finite")
        target = self.manifest.get("target_layer")
        layers = self.manifest.get("num_layers")
        if isinstance(target, bool) or not isinstance(target, int) or not isinstance(layers, int):
            raise ValueError("Explicit target_layer and num_layers are required")
        if layers < 1 or target != layers - 1:
            raise ValueError("Output-head scoring requires the final decoder layer")
        self.hidden_dtype = resolve_dtype(self.manifest["hidden_dtype"])
        self.norm = ExactRMSNorm(norm_weight, float(norm["eps"]))
        self.register_buffer("head_weight", head_weight.detach().clone())
        self.register_buffer("head_bias", None if head_bias is None else head_bias.detach().clone())
        for name, actual in (("hidden_dim", head_weight.shape[1]), ("vocab_size", head_weight.shape[0])):
            if name in self.manifest and self.manifest[name] != actual:
                raise ValueError(f"Manifest {name} does not match tensors")
            self.manifest[name] = int(actual)
        for name, actual in (("norm_dtype", norm_weight.dtype), ("head_dtype", head_weight.dtype)):
            if name in self.manifest and resolve_dtype(self.manifest[name]) != actual:
                raise ValueError(f"Manifest {name} does not match tensor dtype")
            self.manifest[name] = dtype_name(actual)

    @torch.inference_mode()
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim < 2 or hidden.shape[-1] != self.head_weight.shape[1]:
            raise ValueError("Hidden tensor has incompatible shape")
        if not hidden.is_floating_point() or not torch.isfinite(hidden).all():
            raise ValueError("Hidden tensor must contain finite floating values")
        hidden = hidden.to(device=self.head_weight.device, dtype=self.hidden_dtype)
        normalized = self.norm(hidden)
        if normalized.dtype != self.head_weight.dtype:
            raise ValueError("Norm output/head dtype mismatch; do not silently change runtime arithmetic")
        return F.linear(normalized, self.head_weight, self.head_bias)


def _snapshot_tensors(snapshot: Path, tensor_keys: Mapping[str, str]) -> tuple[dict, dict]:
    """Read selected tensors; never pickle-load an unknown checkpoint."""
    from safetensors import safe_open

    if not snapshot.is_dir():
        raise FileNotFoundError(f"Local snapshot directory not found: {snapshot}")
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        index = read_json(index_path)
        mapping = index.get("weight_map", {})
    elif (snapshot / "model.safetensors").is_file():
        with safe_open(snapshot / "model.safetensors", framework="pt", device="cpu") as handle:
            mapping = {key: "model.safetensors" for key in handle.keys()}
    else:
        raise ValueError("Safe local safetensors snapshot unavailable; export from the existing runtime model")
    result, hashes = {}, {}
    for role, key in tensor_keys.items():
        if key not in mapping:
            raise KeyError(f"Tensor {key!r} missing; tied/renamed weights require an explicit verified key")
        relative = Path(mapping[key])
        path = snapshot / relative
        # HF cache paths may themselves be symlinks into local blobs. Validate
        # the manifest pathname, rather than rejecting normal cache symlinks.
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe tensor shard path")
        with safe_open(path, framework="pt", device="cpu") as handle:
            result[role] = handle.get_tensor(key)
        if str(relative) not in hashes:
            hashes[str(relative)] = sha256_file(path)
    return result, hashes


def export_local_output_head(source: str | Path | object, output_dir: str | Path, *,
                             metadata: Mapping[str, Any], norm_spec: Mapping[str, Any] | None = None,
                             tensor_keys: Mapping[str, str] | None = None) -> dict:
    """Export from local safetensors or an already-loaded runtime model.

    For a snapshot, callers must supply the verified norm implementation and
    model identities; keys are checked against the local file, never guessed
    through suffix matching. No downloads or model loading are performed.
    """
    from safetensors.torch import save_file

    required = ("model_revision", "code_revision", "target_layer", "hidden_dtype")
    missing = [key for key in required if metadata.get(key) is None or metadata.get(key) == ""]
    if missing:
        raise ValueError(f"Missing explicit head metadata: {missing}")
    meta = dict(metadata)
    source_hashes = {}
    if isinstance(source, (str, Path)):
        snapshot = Path(source).expanduser().resolve()
        if norm_spec is None or norm_spec.get("implementation") != NORM_IMPLEMENTATION:
            raise ValueError("Snapshot export requires an explicit verified norm_spec")
        config_path = snapshot / "config.json"
        config = read_json(config_path)
        text_config = config.get("text_config", config)
        layers = text_config.get("num_hidden_layers")
        if not isinstance(layers, int):
            raise ValueError("Local text config lacks num_hidden_layers")
        meta["num_layers"] = layers
        if float(text_config.get("rms_norm_eps", float("nan"))) != float(norm_spec["eps"]):
            raise ValueError("Norm epsilon disagrees with local text config")
        keys = dict(tensor_keys or {"norm_weight": "language_model.model.norm.weight",
                                   "head_weight": "language_model.lm_head.weight"})
        if not {"norm_weight", "head_weight"}.issubset(keys) or set(keys) - {"norm_weight", "head_weight", "head_bias"}:
            raise ValueError("Explicit tensor keys must identify norm_weight and head_weight")
        tensors, source_hashes = _snapshot_tensors(snapshot, keys)
        # Loading a checkpoint can cast FP32 disk weights to BF16. Reproduce
        # that cast only when the caller has explicitly identified each
        # runtime weight dtype; never infer it solely from hidden dtype.
        meta["source_tensor_dtypes"] = {key: dtype_name(value.dtype) for key, value in tensors.items()}
        for role, declaration in (("norm_weight", "norm_dtype"), ("head_weight", "head_dtype"),
                                  ("head_bias", "head_dtype")):
            if role in tensors and declaration in meta:
                tensors[role] = tensors[role].to(dtype=resolve_dtype(meta[declaration]))
        source_hashes["config.json"] = sha256_file(config_path)
        meta["source_kind"] = "local_safetensors"
        meta["tensor_keys"] = keys
        if (snapshot / "generation_config.json").is_file():
            meta["generation_config"] = read_json(snapshot / "generation_config.json")
            source_hashes["generation_config.json"] = sha256_file(snapshot / "generation_config.json")
        norm = dict(norm_spec)
    else:
        language = source.language_model
        live_norm, live_head = language.model.norm, language.lm_head
        if not isinstance(live_head, nn.Linear):
            raise ValueError("Only a plain linear runtime lm_head is supported")
        if type(live_norm).__name__ not in {"LlamaRMSNorm", "ExactRMSNorm"}:
            raise ValueError("Unknown runtime norm class; cannot certify its forward definition")
        norm = {"implementation": NORM_IMPLEMENTATION,
                "eps": float(live_norm.variance_epsilon),
                "runtime_class": f"{type(live_norm).__module__}.{type(live_norm).__qualname__}"}
        try:
            norm["runtime_forward_source_hash"] = fingerprint(inspect.getsource(type(live_norm).forward))
        except (OSError, TypeError):
            raise ValueError("Runtime norm forward source is unavailable") from None
        meta["num_layers"] = len(language.model.layers)
        meta["source_kind"] = "runtime_object"
        tensors = {"norm_weight": live_norm.weight.detach().cpu(),
                   "head_weight": live_head.weight.detach().cpu()}
        if live_head.bias is not None:
            tensors["head_bias"] = live_head.bias.detach().cpu()
    meta.update(schema_version=SCHEMA, norm=norm, source_hashes=source_hashes,
                runtime_parity_status="not_run")
    bundle = OutputHeadBundle(tensors["norm_weight"], tensors["head_weight"], meta, tensors.get("head_bias"))
    output = Path(output_dir).expanduser().resolve()
    if isinstance(source, (str, Path)) and (output == snapshot or snapshot in output.parents):
        raise ValueError("Output bundle must be outside the input snapshot")
    output.mkdir(parents=True, exist_ok=True)
    weights_path = output / "output_head.safetensors"
    manifest_path = output / "output_head_manifest.json"
    if weights_path.exists() or manifest_path.exists():
        raise FileExistsError("Output-head bundle already exists; use a new output directory")
    fd, temporary = tempfile.mkstemp(prefix=".head-", suffix=".safetensors", dir=output)
    os.close(fd)
    try:
        save_file({key: value.detach().cpu().contiguous().clone() for key, value in tensors.items()}, temporary)
        os.link(temporary, weights_path)  # atomic no-clobber publication
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    manifest = dict(bundle.manifest)
    manifest.update(weights_file=weights_path.name, weights_sha256=sha256_file(weights_path))
    manifest["bundle_fingerprint"] = fingerprint(manifest)
    atomic_write_json(manifest_path, manifest)
    return manifest


def load_output_head(path: str | Path, *, device: str | torch.device = "cpu") -> OutputHeadBundle:
    """Load and verify an immutable exported bundle on an explicit device."""
    from safetensors.torch import load_file

    root = Path(path).expanduser().resolve()
    manifest_path = root / "output_head_manifest.json" if root.is_dir() else root
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported output-head bundle schema")
    signed = dict(manifest)
    signature = signed.pop("bundle_fingerprint", None)
    if signature != fingerprint(signed):
        raise ValueError("Head manifest fingerprint mismatch")
    filename = Path(manifest["weights_file"])
    if filename.is_absolute() or len(filename.parts) != 1:
        raise ValueError("Unsafe head weights path")
    weights = manifest_path.parent / filename
    if sha256_file(weights) != manifest["weights_sha256"]:
        raise ValueError("Head weights hash mismatch")
    tensors = load_file(weights, device=str(device))
    if set(tensors) - {"norm_weight", "head_weight", "head_bias"}:
        raise ValueError("Unexpected tensors in head bundle")
    return OutputHeadBundle(tensors["norm_weight"], tensors["head_weight"], manifest, tensors.get("head_bias"))


def decode_action_bins(token_ids: torch.Tensor, metadata: Mapping[str, Any]) -> torch.Tensor:
    """Decode raw top-1 IDs using an explicitly verified OpenVLA rule.

    This produces normalized bins, not executed actions or a counterfactual
    autoregressive action sequence. Gripper/environment postprocessing is not
    included.
    """
    if metadata.get("verified") is not True or metadata.get("rule") != "effective_vocab_minus_token_minus_one_clip":
        raise ValueError("Action decoding metadata has not been verified")
    effective = metadata.get("effective_vocab_size")
    vocab = metadata.get("model_vocab_size")
    if not isinstance(effective, int) or not isinstance(vocab, int) or not 0 < effective <= vocab:
        raise ValueError("Invalid effective/model vocabulary sizes")
    if token_ids.is_floating_point() or token_ids.dtype == torch.bool:
        raise ValueError("Token IDs must be integers")
    if ((token_ids < 0) | (token_ids >= vocab)).any():
        raise ValueError("Token ID is outside model vocabulary")
    centers = torch.as_tensor(metadata.get("bin_centers", []), dtype=torch.float32, device=token_ids.device)
    if centers.ndim != 1 or centers.numel() < 1 or not torch.isfinite(centers).all():
        raise ValueError("Valid bin centers are required")
    indices = (effective - token_ids.to(torch.int64) - 1).clamp(0, centers.numel() - 1)
    return centers[indices]


def compare_logits(reference: torch.Tensor, actual: torch.Tensor, *, atol: float,
                   rtol: float, max_argmax_mismatch: float = 0.0) -> dict:
    """Pure comparison with caller-declared, never automatically widened tolerances."""
    if not (math.isfinite(atol) and math.isfinite(rtol) and atol >= 0 and rtol >= 0
            and 0 <= max_argmax_mismatch <= 1):
        raise ValueError("Invalid parity tolerances")
    if reference.shape != actual.shape or reference.ndim < 2 or reference.numel() == 0:
        raise ValueError("Parity logits must have identical nonempty shapes")
    ref, value = reference.float(), actual.to(reference.device).float()
    finite = bool(torch.isfinite(ref).all() and torch.isfinite(value).all())
    error = (ref - value).abs()
    mismatch = float((ref.argmax(-1) != value.argmax(-1)).float().mean()) if finite else None
    close = finite and bool(torch.allclose(ref, value, atol=atol, rtol=rtol))
    passed = close and mismatch <= max_argmax_mismatch
    return {"status": "passed" if passed else "failed", "finite": finite,
            "max_abs_error": float(error.max()) if finite else None,
            "argmax_mismatch_rate": mismatch, "shape": list(ref.shape),
            "atol": atol, "rtol": rtol, "max_argmax_mismatch": max_argmax_mismatch}


def validate_parity_report(report: Mapping[str, Any], current_identity: Mapping[str, Any],
                           required_checks: Sequence[str], *, allow_synthetic: bool = False) -> None:
    """Reject missing, stale, failed, incomplete, or synthetic runtime certificates."""
    if report.get("status") != "passed":
        raise ValueError("Parity report has not passed")
    if report.get("synthetic") is not False and not allow_synthetic:
        raise ValueError("Synthetic or undeclared reports cannot satisfy runtime parity")
    if not current_identity or any(value is None or value == "" for value in current_identity.values()):
        raise ValueError("Current parity identity is incomplete")
    if report.get("identity") != dict(current_identity):
        raise ValueError("Stale parity: current numerical identity differs")
    if report.get("identity_fingerprint") != fingerprint(dict(current_identity)):
        raise ValueError("Parity identity fingerprint mismatch")
    if not required_checks or len(set(required_checks)) != len(required_checks):
        raise ValueError("Explicit unique required parity checks are required")
    checks = report.get("checks", {})
    missing = [name for name in required_checks if not isinstance(checks.get(name), dict)
               or checks[name].get("status") != "passed"]
    if missing:
        raise ValueError(f"Missing or failed parity checks: {missing}")
