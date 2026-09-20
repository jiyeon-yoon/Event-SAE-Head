"""Opt-in M4/M5 adapters. Never imported by help, audit, split, or plan.

These adapters are covered with guards/fixtures, not certified against a real
OpenVLA installation by the CPU implementation tests.
"""

from __future__ import annotations

import copy
import hashlib
import io
import os
import time
from collections import defaultdict
from pathlib import Path

from .provenance import fingerprint, git_identity, read_json, read_jsonl, sha256_file
from .workflow import (REPO_ROOT, _head_path, _sae_paths, load_parity, numerical_identity,
                       output_root, required_input, write_once_json)


def _offline_environment() -> None:
    # Existing trusted remote code must already be cached. A missing dependency
    # is an error; it is never a reason to fetch an unpinned replacement.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def verify_approved_plan(cfg: dict, approved_hash: str, expected_revision: str) -> dict:
    """All cheap authorization/provenance/budget checks precede model imports."""
    if not approved_hash or not expected_revision:
        raise ValueError("Explicit plan hash and code revision are required")
    root = output_root(cfg)
    plan = read_json(root / "plan" / "rollout_plan.json")
    unsigned = {key: value for key, value in plan.items() if key != "plan_hash"}
    if plan.get("plan_hash") != approved_hash or fingerprint(unsigned) != approved_hash:
        raise ValueError("Approved plan hash is empty, stale, or tampered")
    if plan.get("synthetic") is not False or plan.get("status") != "planned":
        raise ValueError("Only a real, within-budget planned experiment may run")
    if plan.get("experiment_config_hash") != fingerprint(cfg):
        raise ValueError("Config changed after plan approval")
    if plan.get("implementation_fingerprint") != numerical_implementation_fingerprint():
        raise ValueError("Implementation changed after plan approval")
    if plan.get("score_artifact_hash") != sha256_file(root / "scores" / "scores.json"):
        raise ValueError("Scores changed after plan approval")
    if plan.get("base_eval_config_hash") != sha256_file(cfg["rollout"]["base_eval_config"]):
        raise ValueError("Base eval configuration changed after plan approval")
    if plan.get("eval_manifest_hash") != sha256_file(cfg["rollout"]["eval_manifest"]):
        raise ValueError("Evaluation manifest changed after plan approval")
    code = git_identity(REPO_ROOT)
    if code["dirty"] or code["commit"] != expected_revision:
        raise ValueError("Run requires the exact approved clean code revision")
    conditions = plan["conditions"]
    total = sum(item["num_cases"] for item in conditions)
    if total != plan["total_rollouts"] or total > cfg["rollout"]["max_total_rollouts"]:
        raise ValueError("Approved rollout budget is inconsistent or exceeded")
    if not any(item["mode"] == "identity" for item in conditions):
        raise ValueError("An alpha=1 identity check is required")
    return plan


def numerical_implementation_fingerprint() -> str:
    from .provenance import implementation_fingerprint
    return implementation_fingerprint()


def _validate_frozen_split(cfg: dict, cases: list[dict], *, stage: str) -> None:
    from .splits import validate_selection_split, validate_split_manifest
    path = cfg["sampling"]["split_manifest"]
    if cfg["sampling"]["mode"] == "confirmatory" and not path:
        raise ValueError("Confirmatory execution requires the pre-pilot frozen split")
    if path:
        split = read_json(path)
        validate_split_manifest(split, require_confirmatory=True)
        validate_selection_split(split, cases, stage=stage)


def validate_condition_result(payload: dict, condition: dict, cases: list[dict],
                              protocol_id: str, identity: dict, revision: str) -> None:
    """A completed filename is not a resume certificate: validate its contents."""
    unsigned = {key: value for key, value in payload.items() if key != "result_hash"}
    if payload.get("result_hash") != fingerprint(unsigned):
        raise ValueError("Result content fingerprint mismatch")
    expected = {"status": "completed", "synthetic": False, "protocol_id": protocol_id,
                "head_sae_identity": identity, "mode": condition["mode"],
                "feature_id": condition["feature_id"], "alpha": condition["alpha"],
                "code": {"commit": revision, "dirty": False}, "completed_rollouts": len(cases)}
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("Result condition or provenance does not match approved plan")
    expected_cases = {(row["task_id"], row["task_episode_idx"]): row["initial_state_sha256"] for row in cases}
    actual_cases = {}
    rows = payload.get("episodes", [])
    for row in rows:
        key = (row["task_id"], row["task_episode_idx"])
        if key in actual_cases or not isinstance(row.get("success"), bool) or row.get("caught_exception"):
            raise ValueError("Invalid/duplicate/exceptional resumed episode")
        if "caught_exception" not in row:
            raise ValueError("Resume requires explicit episode completion evidence")
        actual_cases[key] = row.get("initial_state_sha256")
    if actual_cases != expected_cases or len(rows) != len(cases):
        raise ValueError("Resumed cases differ from approved evaluation states")
    if payload.get("actions_sha256") != sha256_file(payload["actions_path"]):
        raise ValueError("Resumed action artifact changed")
    if condition["mode"] != "raw":
        metrics = payload.get("hook_metrics", {})
        if metrics.get("num_forwards", 0) <= 0:
            raise ValueError("Resumed condition lacks executed hook evidence")
        if condition["alpha"] == 0 and (metrics.get("post_intervention_active_feature_values") != 0 or
                                        metrics.get("post_intervention_max_feature_activation") != 0):
            raise ValueError("Resumed condition lacks target suppression evidence")
    if condition["mode"] == "identity" and payload.get("identity_check") != "passed":
        raise ValueError("Resumed alpha=1 identity check has not passed")


def verify_identity_actions(raw: dict, edited: dict) -> None:
    if read_json(raw["actions_path"]) != read_json(edited["actions_path"]):
        raise ValueError("Alpha=1 identity action sequences differ from raw baseline")
    raw_outcomes = {(x["task_id"], x["task_episode_idx"]): x["success"] for x in raw["episodes"]}
    edit_outcomes = {(x["task_id"], x["task_episode_idx"]): x["success"] for x in edited["episodes"]}
    if raw_outcomes != edit_outcomes:
        raise ValueError("Alpha=1 identity outcomes differ from baseline")


def validate_runtime(cfg: dict) -> dict:
    """Compare the original all-row hook against offline fixed-context readouts.

Input is a small local torch file with ``calls`` (preprocessed predict_action
kwargs), ``episodes`` (source state IDs/hashes), ``processor_identity`` and
model/code revisions. It is not generated from approximate JPEG recovery.
No environment rollout is used for this numerical check.
"""
    import torch
    from event_sae.openvla.activations import load_batch_topk_sae
    from event_sae.openvla.intervene import apply_resid_post_feature_perturb_hook
    from .head import (HEAD_PARITY_CHECKS, EDIT_PARITY_CHECKS, compare_logits,
                       load_output_head, max_bfloat16_ulp_error, resolve_dtype)
    from .sensitivity import edit_feature_reference

    root = output_root(cfg)
    validation_dir = root / "validation"
    if (validation_dir / "runtime_parity.json").exists() or (validation_dir / "edit_parity.json").exists():
        identity = numerical_identity(cfg)
        load_parity(cfg, identity)
        return {"status": "passed", "resumed": True, "validation": str(validation_dir)}
    fixture_path = required_input(cfg, "runtime_inputs")
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
    if fixture.get("synthetic") is not False:
        raise ValueError("Real parity requires explicitly non-synthetic runtime inputs")
    calls = fixture.get("calls", [])
    if not calls or len(calls) * 4 > cfg["validation"]["max_model_calls"]:
        raise ValueError("Runtime input call count is empty or exceeds validation budget")
    _validate_frozen_split(cfg, fixture.get("episodes", []), stage="pilot")
    identity = numerical_identity(cfg)
    if not identity.get("processor_identity") or fixture.get("processor_identity") != identity["processor_identity"]:
        raise ValueError("Runtime input processor provenance differs from head export")
    for key in ("model_revision", "code_revision"):
        if fixture.get(key) != identity[key]:
            raise ValueError(f"Runtime input provenance mismatch: {key}")
    _offline_environment()
    from transformers import AutoModelForVision2Seq
    device = cfg["scoring"]["device"]
    model = AutoModelForVision2Seq.from_pretrained(
        str(required_input(cfg, "local_model_snapshot")), local_files_only=True,
        trust_remote_code=True, code_revision=cfg["model"]["code_revision"],
        torch_dtype=resolve_dtype(cfg["model"]["expected_live_hidden_dtype"]),
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).to(device).eval()
    # Match the native runner's local statistics override exactly.
    model.norm_stats = read_json(required_input(cfg, "local_model_snapshot") / "dataset_statistics.json")
    if cfg["model"]["unnorm_key"] not in model.norm_stats:
        raise ValueError("Runtime validation unnormalization key is missing from local statistics")
    layers = model.language_model.model.layers
    if cfg["scope"]["layer_idx"] != len(layers) - 1:
        raise ValueError("Loaded runtime target is not its final decoder layer")
    head = load_output_head(_head_path(cfg), device=device)
    weights, _ = _sae_paths(cfg)
    sae, _ = load_batch_topk_sae(weights, device=device)
    # Check actual loaded tensors, not just user-declared revision strings.
    live = model.language_model
    if (live.model.norm.weight.dtype != head.norm.weight.dtype or
            live.lm_head.weight.dtype != head.head_weight.dtype or
            not torch.equal(live.model.norm.weight, head.norm.weight) or
            not torch.equal(live.lm_head.weight, head.head_weight) or
            float(live.model.norm.variance_epsilon) != head.norm.variance_epsilon):
        raise ValueError("Exported head is not identical to loaded runtime norm/head")
    if ((live.lm_head.bias is None) != (head.head_bias is None) or
            (head.head_bias is not None and (live.lm_head.bias.dtype != head.head_bias.dtype or
                                             not torch.equal(live.lm_head.bias, head.head_bias)))):
        raise ValueError("Exported head bias differs from loaded runtime")
    tolerances = {key: cfg["validation"][key] for key in ("atol", "rtol", "max_argmax_mismatch")}
    generation = read_json(required_input(cfg, "generation_manifest"))
    observations, isolation_checks = [], []
    active_ids, inactive_ids = set(), None
    dictionary_size = None
    batch_checks, edit_batch_checks = [], []
    validation_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def run_condition(feature: int | None, alpha: float, name: str) -> None:
        nonlocal inactive_ids, dictionary_size
        current = {}
        condition_dir = validation_dir / name
        condition_dir.mkdir(exist_ok=True)

        def before_original_hook(_module, _inputs, output):
            nonlocal inactive_ids, dictionary_size
            h = output[0].detach()
            flat = h.reshape(-1, h.shape[-1])
            z = sae.encode(flat.float())
            dictionary_size = z.shape[1]
            alive = set(torch.nonzero(z[-1] > 0).flatten().tolist())
            if feature is None:
                active_ids.update(alive)
                zero = set(torch.nonzero(torch.count_nonzero(z, dim=0) == 0).flatten().tolist())
                inactive_ids = zero if inactive_ids is None else inactive_ids & zero
                isolated_edit = flat[-1:]
                # Compare the exported bundle with the live norm/head at the
                # exact scoring shape. Comparing M=1 with M=N is invalid for
                # BF16 CUDA GEMM even when every row is identical.
                batch_size = cfg["scoring"]["pair_batch_size"]
                rows = flat[-1:].expand(batch_size, -1).contiguous().to(h.dtype)
                batch_checks.append(compare_logits(
                    head(rows),
                    torch.nn.functional.linear(live.model.norm(rows), live.lm_head.weight,
                                               live.lm_head.bias),
                    **tolerances))
                choices = sorted(alive) or [0]
                features = torch.tensor([choices[i % len(choices)] for i in range(batch_size)], device=z.device)
                encoded = z[-1:].expand(batch_size, -1).contiguous()
                changed = encoded.clone()
                changed[torch.arange(batch_size, device=z.device), features] = 0
                vector_edit = (rows.float() + (sae.decode(changed) - sae.decode(encoded))).to(rows.dtype)
                scalar_edit = torch.cat([edit_feature_reference(rows[i:i+1], sae, int(features[i]), 0.0,
                                                                hidden_dtype=rows.dtype, encoded=encoded[i:i+1])
                                         for i in range(batch_size)])
                edit_batch_checks.append(compare_logits(head(scalar_edit), head(vector_edit), **tolerances))
            else:
                # Only selected z row goes into the offline readout reference;
                # it came from encode of the original entire forward.
                isolated_edit = edit_feature_reference(flat[-1:], sae, feature, alpha,
                                                       hidden_dtype=h.dtype, encoded=z[-1:])

            # The live norm/head sees the complete prefill tensor. Core hook
            # parity therefore uses the full-group edit and full forward shape.
            if feature is None:
                full_edit = flat.to(h.dtype)
            else:
                full_edit = edit_feature_reference(flat, sae, feature, alpha,
                                                   hidden_dtype=h.dtype, encoded=z)
            full_logits = head(full_edit.reshape_as(h)).reshape(-1, head.head_weight.shape[0])[-1:]

            # Separately measure the isolated-row arithmetic used by offline
            # scoring. It is a declared predictor rather than an exact copy of
            # the all-row runtime hook, so divergence is reported but cannot
            # satisfy or invalidate hook parity.
            isolated_full = flat.to(h.dtype).clone()
            isolated_full[-1:] = isolated_edit
            isolated_full_logits = head(isolated_full.reshape_as(h)).reshape(
                -1, head.head_weight.shape[0])[-1:]
            isolated_scalar_logits = head(isolated_edit)
            full_hidden = full_edit[-1:]
            hidden_actual = isolated_edit.to(full_hidden.device)
            hidden_finite = bool(torch.isfinite(full_hidden).all()
                                 and torch.isfinite(hidden_actual).all())
            hidden_error = (full_hidden.float() - hidden_actual.float()).abs()
            hidden_close = hidden_finite and bool(torch.allclose(
                full_hidden.float(), hidden_actual.float(),
                atol=tolerances["atol"], rtol=tolerances["rtol"]))
            differing = int(torch.count_nonzero(full_hidden != hidden_actual).item())
            decoder_grouping = compare_logits(full_logits, isolated_full_logits, **tolerances)
            projection_grouping = compare_logits(
                isolated_full_logits, isolated_scalar_logits, **tolerances)
            runtime_vs_score = compare_logits(full_logits, isolated_scalar_logits, **tolerances)
            all_diagnostic_logits_pass = all(
                item["status"] == "passed"
                for item in (decoder_grouping, projection_grouping, runtime_vs_score))
            isolation_checks.append({
                "condition": name, "alpha": alpha, "feature_id": feature,
                "row_count": h.shape[-2],
                **runtime_vs_score,
                "status": "passed" if hidden_close and all_diagnostic_logits_pass else "failed",
                "logit_status": runtime_vs_score["status"],
                "decoder_grouping": decoder_grouping,
                "projection_grouping": projection_grouping,
                "runtime_vs_score": runtime_vs_score,
                "hidden_status": "passed" if hidden_close else "failed",
                "hidden_finite": hidden_finite,
                "hidden_exact_equal": differing == 0,
                "hidden_differing_elements": differing,
                "hidden_differing_fraction": differing / full_hidden.numel(),
                "hidden_max_abs_error": float(hidden_error.max()) if hidden_finite else None,
                "hidden_max_bfloat16_ulp_error": (
                    max_bfloat16_ulp_error(full_hidden, hidden_actual) if hidden_finite else None),
            })
            current.update(expected=full_logits, rows=h.shape[-2],
                           active=feature is not None and bool(z[-1, feature] != 0))

        def after_head(_module, _inputs, output):
            if not current:
                raise ValueError("Missing block capture for runtime logits")
            actual = output.reshape(-1, output.shape[-1])[-1:]
            result = compare_logits(current["expected"], actual, **tolerances)
            observations.append({"condition": name, "alpha": alpha, "feature_id": feature,
                                 "row_count": current["rows"], "active": current["active"], **result})
            current.clear()

        capture = layers[-1].register_forward_hook(before_original_hook)
        head_hook = live.lm_head.register_forward_hook(after_head)
        intervention = None
        log_stream = io.StringIO()
        start_count = len(observations)
        try:
            if feature is not None:
                intervention = apply_resid_post_feature_perturb_hook(
                    model=model, layer_idx=cfg["scope"]["layer_idx"],
                    sae_checkpoint_path=str(weights), feature_idx=feature, alpha=alpha,
                    hook_start_step=0, run_dir=condition_dir, log_file=log_stream)
            with torch.inference_mode():
                for index, call in enumerate(calls):
                    if not isinstance(call, dict) or any(not torch.is_tensor(value) for value in call.values()):
                        raise ValueError("Runtime calls must contain preprocessed tensor kwargs only")
                    model._sae_hook_context = {"episode_num": index, "step_in_episode": 0}
                    kwargs = {key: value.to(device=device, dtype=head.hidden_dtype if value.is_floating_point() else value.dtype)
                              for key, value in call.items()}
                    if (kwargs.get("input_ids") is None or kwargs["input_ids"].ndim != 2 or
                            kwargs["input_ids"].shape[0] != 1 or
                            ("attention_mask" in kwargs and not bool(kwargs["attention_mask"].all()))):
                        raise ValueError("Runtime validation requires verified unpadded batch-size-1 inputs")
                    before_query = len(observations)
                    model.predict_action(**kwargs, unnorm_key=cfg["model"]["unnorm_key"], do_sample=False)
                    query_records = observations[before_query:]
                    if len(query_records) != generation["action_dim"]:
                        raise ValueError("Runtime generation forward count disagrees with readout mapping for one query")
                    if query_records[0]["row_count"] <= 1 or any(row["row_count"] != 1 for row in query_records[1:]):
                        raise ValueError("Runtime prefill/cached forward structure disagrees with mapping")
                    for row in query_records:
                        row["call_index"] = index
        finally:
            capture.remove()
            head_hook.remove()
            if intervention is not None:
                intervention.remove()
            log_stream.close()
        if len(observations) - start_count != len(calls) * generation["action_dim"]:
            raise ValueError("Runtime generation forward count disagrees with readout mapping")

    run_condition(None, 1.0, "baseline")
    if not active_ids or not inactive_ids:
        raise ValueError("Validation inputs must expose both active and inactive SAE features")
    feature, inactive = min(active_ids), min(inactive_ids)
    run_condition(feature, 1.0, "alpha1")
    run_condition(feature, 0.0, "active_alpha0")
    run_condition(inactive, 0.0, "inactive")

    def check(rows: list[dict]) -> dict:
        return {"status": "passed" if rows and all(row["status"] == "passed" for row in rows) else "failed",
                "num_forwards": len(rows), "comparisons": rows}

    baseline = [x for x in observations if x["condition"] == "baseline"]
    edited = [x for x in observations if x["condition"] != "baseline"]
    checks = {
        "alpha1": check([x for x in edited if x["condition"] == "alpha1"]),
        "active_alpha0": check([x for x in edited if x["condition"] == "active_alpha0" and x["active"]]),
        "inactive": check([x for x in edited if x["condition"] == "inactive" and not x["active"]]),
        "prefill": check([x for x in edited if x["row_count"] > 1]),
        "cached": check([x for x in edited if x["row_count"] == 1]),
        "all_rows": check(edited),
        "pair_batch_edit": check(edit_batch_checks),
    }
    base = {"synthetic": False, "identity": identity, "identity_fingerprint": fingerprint(identity),
            "sample_identity": sha256_file(fixture_path), "observations": len(observations),
            "model_action_queries": len(calls) * 4, "elapsed_seconds": time.perf_counter() - started}
    head_report = {**base, "checks": {"baseline_logits": check(baseline), "head_batch_shape": check(batch_checks)}}
    edit_report = {
        **base,
        "checks": checks,
        "diagnostics": {
            "isolated_score_vs_runtime": {
                **check(isolation_checks),
                "required": False,
                "arithmetic_mode": cfg["scoring"]["arithmetic_mode"],
                "interpretation": (
                    "Separately reports SAE decoder grouping drift, output-head projection grouping "
                    "drift, and their combined runtime-versus-score drift; this predictor diagnostic "
                    "is not a runtime parity gate."
                ),
            }
        },
    }
    for report, names, filename in ((head_report, HEAD_PARITY_CHECKS, "runtime_parity.json"),
                                    (edit_report, EDIT_PARITY_CHECKS, "edit_parity.json")):
        report["status"] = "passed" if all(report["checks"][key]["status"] == "passed" for key in names) else "failed"
        write_once_json(validation_dir / filename, report)
    if head_report["status"] != "passed" or edit_report["status"] != "passed":
        raise ValueError("Runtime numerical parity failed; no scoring/rollout is authorized")
    return {"status": "passed", "validation": str(validation_dir), "elapsed_seconds": base["elapsed_seconds"]}


def run_plan(cfg: dict, approved_hash: str, expected_revision: str) -> dict:
    """Execute frozen, bounded cases through the original all-row eval/hook."""
    plan = verify_approved_plan(cfg, approved_hash, expected_revision)
    cases = plan["eval_cases"]
    _validate_frozen_split(cfg, cases, stage="evaluation" if cfg["sampling"]["mode"] == "confirmatory" else "pilot")
    identity = numerical_identity(cfg)
    load_parity(cfg, identity)
    snapshot = required_input(cfg, "local_model_snapshot")
    weights, _ = _sae_paths(cfg)
    _offline_environment()
    from event_sae.openvla.eval.config import load_config
    from event_sae.openvla.eval.runner import eval_libero
    from event_sae.openvla.eval.utils import DEVICE
    from event_sae.openvla.intervene import apply_resid_post_feature_perturb_hook
    if str(DEVICE) != cfg["scoring"]["device"]:
        raise ValueError(f"Native runner device {DEVICE} differs from parity device {cfg['scoring']['device']}")
    root = output_root(cfg)
    base = load_config(cfg["rollout"]["base_eval_config"])
    if base.model.revision != cfg["model"]["revision"] or base.model.code_revision != cfg["model"]["code_revision"]:
        raise ValueError("Base eval model revisions do not match sensitivity experiment")
    base.model.checkpoint = str(snapshot)
    base.model.attn_implementation = "sdpa"
    base.model.local_files_only = True
    base.env.task_suite_name = cfg["scope"]["suite"]
    base.env.seed = cfg["rollout"]["env_seed"]
    base.env.per_episode_seed = True
    base.sae_collect.enabled = False
    base.logging.save_video = cfg["rollout"]["save_video"]
    base.logging.save_trajectory_records = False
    base.logging.save_prompt_records = True
    base.logging.save_actions = True  # Required for raw vs alpha=1 comparison.
    if base.model.load_in_8bit or base.model.load_in_4bit or cfg["model"]["expected_live_hidden_dtype"] != "bfloat16":
        raise ValueError("Native research rollout requires unquantized BF16")
    statistics = read_json(snapshot / "dataset_statistics.json")
    effective_unnorm = (base.env.task_suite_name if base.env.task_suite_name in statistics
                        else f"{base.env.task_suite_name}_no_noops")
    if effective_unnorm != cfg["model"]["unnorm_key"] or effective_unnorm not in statistics:
        raise ValueError("Native rollout unnorm key differs from sensitivity protocol")
    from dataclasses import asdict
    effective_policy = asdict(base)
    effective_policy.pop("logging")
    effective_policy.pop("sae_collect")
    protocol_id = fingerprint({"plan_hash": approved_hash, "identity": identity,
                               "code_revision": expected_revision, "effective_policy": effective_policy,
                               "per_episode_seed": "sha256_seed_task_trial_v1"})
    results = {}
    conditions = sorted(plan["conditions"], key=lambda c: {"raw": 0, "identity": 1, "intervention": 2}[c["mode"]])
    for condition in conditions:
        name = condition["condition_id"]
        if condition["num_cases"] != len(cases):
            raise ValueError("Native MVP identity and feature conditions must cover the same approved cases")
        target = root / "runs" / name
        summary = target / "result.json"
        if summary.exists():
            previous = read_json(summary)
            validate_condition_result(previous, condition, cases, protocol_id, identity, expected_revision)
            if condition["mode"] == "identity":
                verify_identity_actions(results["raw"], previous)
            results[name] = previous
            continue
        if target.exists() and any(target.iterdir()):
            raise ValueError(f"Incomplete condition {name}: inspect before restarting; budget is not silently retried")
        selected = cases[:condition["num_cases"]]
        by_task = defaultdict(list)
        expected = {}
        for case in selected:
            by_task[case["task_id"]].append(case["task_episode_idx"])
            expected[(case["task_id"], case["task_episode_idx"])] = case["initial_state_sha256"]
        if len({len(values) for values in by_task.values()}) != 1:
            raise ValueError("Native MVP runner requires equal evaluation counts per task")
        run_cfg = copy.deepcopy(base)
        run_cfg.env.task_ids = sorted(by_task)
        run_cfg.env.num_trials_per_task = len(next(iter(by_task.values())))
        run_cfg.logging.root_dir = str(target)
        handles = []

        def applier(*, model, cfg, run_dir, log_file):
            handle = apply_resid_post_feature_perturb_hook(
                model=model, layer_idx=identity["layer_idx"], sae_checkpoint_path=str(weights),
                feature_idx=condition["feature_id"], alpha=condition["alpha"],
                hook_start_step=0, run_dir=run_dir, log_file=log_file)
            handles.append(handle)
            return [handle]

        started = time.perf_counter()
        try:
            outcome = eval_libero(run_cfg, extra_hook_applier=None if condition["mode"] == "raw" else applier,
                                  trial_indices_by_task=dict(by_task), expected_initial_states=expected)
        finally:
            for handle in handles:
                handle.remove()
        rows = read_jsonl(outcome.episode_results_path)
        if len(rows) != len(selected) or any(row.get("caught_exception") for row in rows):
            raise ValueError("Incomplete/exceptional rollout condition cannot be counted as failure")
        hook_metrics = handles[0].summary() if handles else None
        if hook_metrics is not None:
            if hook_metrics["num_forwards"] <= 0:
                raise ValueError("Intervention hook did not execute")
            if condition["alpha"] == 0 and (hook_metrics.get("post_intervention_active_feature_values") != 0 or
                                              hook_metrics.get("post_intervention_max_feature_activation") != 0):
                raise ValueError("Target latent suppression did not validate")
        payload = {"status": "completed", "synthetic": False, "mode": condition["mode"],
                   "feature_id": condition["feature_id"], "alpha": condition["alpha"],
                   "episodes": rows, "completed_rollouts": len(rows), "suite": cfg["scope"]["suite"],
                   "seed": base.env.seed, "protocol_id": protocol_id,
                   "model_checkpoint": cfg["model"]["checkpoint"], "model_revision": cfg["model"]["revision"],
                   "model_code_revision": cfg["model"]["code_revision"],
                   "code": {"commit": expected_revision, "dirty": False}, "head_sae_identity": identity,
                   "run_config": effective_policy,
                   "sae_sha256": identity["sae_checkpoint_hash"], "layer_idx": identity["layer_idx"],
                   "hook_start_step": 0, "hook_metrics": hook_metrics,
                   "actions_path": str(Path(outcome.run_dir) / "actions.json"),
                   "elapsed_seconds": time.perf_counter() - started}
        payload["actions_sha256"] = sha256_file(payload["actions_path"])
        if condition["mode"] == "identity":
            verify_identity_actions(results["raw"], payload)
            payload["identity_check"] = "passed"
        payload["result_hash"] = fingerprint(payload)
        validate_condition_result(payload, condition, cases, protocol_id, identity, expected_revision)
        write_once_json(summary, payload)
        results[name] = payload
    return {"status": "completed", "total_rollouts": plan["total_rollouts"],
            "conditions": list(results), "protocol_id": protocol_id}
