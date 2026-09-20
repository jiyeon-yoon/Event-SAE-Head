"""Strict CPU-only experiment configuration. Missing local inputs block execution."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import yaml

from .provenance import assert_safe_output


DEFAULTS: dict[str, Any] = {
    "schema_version": "output_head_sensitivity_v1",
    "experiment_name": "openvla_l31_fixed_prefix_head_sensitivity",
    "scope": {"model_family": "openvla", "suite": "libero_spatial", "layer_idx": 31,
              "capture_target": "post_mlp_residual", "require_final_decoder_layer": True,
              "retrain_sae": False, "recollect_discovery": False, "collect_extended_env_state": False},
    "inputs": {"dense_dir": None, "source_episode_manifest": None, "sae_checkpoint": None,
               "event_scores_path": None, "existing_candidates_path": None, "existing_result_dirs": [],
               "local_model_snapshot": None, "output_head_bundle": None, "generation_manifest": None,
               "head_export_metadata": None, "runtime_inputs": None},
    "model": {"checkpoint": "openvla/openvla-7b-finetuned-libero-spatial",
              "revision": "962318cec55ac10993ff0f5f43eda9a270b4c873",
              "code_revision": "47a0ec7fc4ec123775a391911046cf33cf9ed83f",
              "unnorm_key": "libero_spatial", "expected_live_hidden_dtype": "bfloat16",
              "allow_download": False},
    "splits": {"split_seed": 2026, "discovery_per_task": 35, "validation_per_task": 5,
               "evaluation_per_task": 10, "require_initial_state_disjoint": True,
               "evaluation_labels_previously_used": None},
    "sampling": {"mode": "pilot", "split_manifest": None, "sample_seed": 2026,
                 "task_ids": [0, 1], "max_episodes_per_task": 2, "max_steps_per_episode": 8,
                 "step_selection": "evenly_spaced", "require_complete_action_query": True,
                 "require_frozen_eval_before_pilot": True},
    "scoring": {"alpha": 0.0, "primary_metric": "head_full_vocab_kl_fixed_prefix",
                "logits_stage": "raw_head", "edit_backend": "reference",
                "arithmetic_mode": "runtime_matched", "reduction_dtype": "float32",
                "aggregation": "equal_task_episode_step_dimension", "include_inactive_as_zero": True,
                "feature_universe": "all_dictionary_features", "pair_batch_size": 32,
                "max_scored_pairs": 20000, "full_vocab": True, "save_per_row_logits": False,
                "device": "cpu", "negative_tolerance": 1e-6,
                "max_working_memory_bytes": 536870912},
    "selection": {"methods": ["event_aligned", "head_full_vocab_kl_fixed_prefix",
                              "mean_readout_activation", "readout_activation_frequency"],
                  "top_k": 3, "random_audit_features": 6, "random_seed": 2026,
                  "topk_tie_break": "feature_id_ascending", "audit_population": "discovery_readout_alive",
                  "audit_sampling": "uniform_without_replacement",
                  "audit_overlap_policy": "keep_memberships_no_redraw", "max_unique_features": 24,
                  "require_all_method_topk": True},
    "rollout": {"execute": False, "env_seed": 0, "alpha": 0.0, "hook_start_step": 0,
                "eval_manifest": None,
                "base_eval_config": "configs/reproduction/openvla/libero_spatial_intervention_layer31.yaml",
                "max_total_rollouts": 100, "require_clean_git": True, "require_head_parity": True,
                "require_edit_parity": True, "require_current_parity_fingerprint": True,
                "require_identity_check": True, "save_actions_for_identity": True,
                "save_video": False, "save_trajectory_records": False, "collect_activations": False},
    "validation": {"atol": 1e-4, "rtol": 1e-4, "max_argmax_mismatch": 0.0,
                   "max_model_calls": 8},
    "analysis": {"primary_label": "success_rate_drop", "bootstrap_replicates": 2000,
                 "bootstrap_seed": 2026, "confidence_level": 0.95, "allow_missing_pairs": False,
                 "require_declared_selection_eval_overlap": True,
                 "secondary_labels": ["outcome_switch_rate"], "report_outcome_transition_counts": True},
    "output": {"root_dir": None, "overwrite": False, "resume": True},
}


def _merge(base: dict, updates: dict, prefix: str = "") -> dict:
    if not isinstance(updates, dict):
        raise ValueError(f"{prefix or 'config'} must be a mapping")
    for key, value in updates.items():
        path = f"{prefix}.{key}".strip(".")
        if key not in base:
            raise ValueError(f"Unknown config field: {path}")
        default = base[key]
        if isinstance(default, dict):
            _merge(default, value, path)
        else:
            if default is not None:
                if isinstance(default, bool):
                    valid = isinstance(value, bool)
                elif isinstance(default, int):
                    valid = type(value) is int
                elif isinstance(default, float):
                    valid = type(value) in (int, float) and math.isfinite(value)
                else:
                    valid = isinstance(value, type(default))
                if not valid:
                    raise ValueError(f"Invalid type/value for {path}")
            base[key] = value
    return base


def validate_config(updates: dict) -> dict:
    cfg = _merge(copy.deepcopy(DEFAULTS), updates)
    for section, names in {
        "inputs": [key for key in cfg["inputs"] if key != "existing_result_dirs"],
        "output": ["root_dir"], "sampling": ["split_manifest"], "rollout": ["eval_manifest"],
    }.items():
        for key in names:
            value = cfg[section][key]
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{section}.{key} must be a nonempty path string or null")
    if any(not isinstance(value, str) or not value for value in cfg["inputs"]["existing_result_dirs"]):
        raise ValueError("existing_result_dirs must contain path strings")
    fixed = {
        "schema_version": "output_head_sensitivity_v1", "scope.model_family": "openvla",
        "scope.suite": "libero_spatial", "scope.capture_target": "post_mlp_residual",
        "scope.require_final_decoder_layer": True, "scope.retrain_sae": False,
        "scope.recollect_discovery": False, "scope.collect_extended_env_state": False,
        "model.allow_download": False, "scoring.alpha": 0.0, "rollout.alpha": 0.0,
        "scoring.primary_metric": "head_full_vocab_kl_fixed_prefix", "scoring.logits_stage": "raw_head",
        "scoring.aggregation": "equal_task_episode_step_dimension", "scoring.full_vocab": True,
        "scoring.include_inactive_as_zero": True, "scoring.save_per_row_logits": False,
        "scoring.edit_backend": "reference", "scoring.arithmetic_mode": "runtime_matched",
        "scoring.feature_universe": "all_dictionary_features",
        "selection.topk_tie_break": "feature_id_ascending", "selection.audit_population": "discovery_readout_alive",
        "selection.audit_sampling": "uniform_without_replacement",
        "selection.audit_overlap_policy": "keep_memberships_no_redraw",
        "selection.require_all_method_topk": True, "rollout.require_clean_git": True,
        "rollout.require_head_parity": True, "rollout.require_edit_parity": True,
        "rollout.require_current_parity_fingerprint": True, "rollout.require_identity_check": True,
        "rollout.save_actions_for_identity": True, "rollout.collect_activations": False,
        "rollout.hook_start_step": 0,
        "rollout.save_trajectory_records": False, "output.overwrite": False,
        "analysis.primary_label": "success_rate_drop", "analysis.allow_missing_pairs": False,
        "sampling.require_complete_action_query": True, "splits.require_initial_state_disjoint": True,
    }
    for path, value in fixed.items():
        actual = cfg
        for key in path.split("."):
            actual = actual[key]
        if actual != value:
            raise ValueError(f"MVP requires {path}={value!r}")
    if cfg["sampling"]["mode"] not in ("pilot", "confirmatory"):
        raise ValueError("sampling.mode must be pilot or confirmatory")
    if cfg["model"]["expected_live_hidden_dtype"] not in ("bfloat16", "float32", "float16"):
        raise ValueError("Unsupported hidden dtype")
    if cfg["scoring"]["reduction_dtype"] not in ("float32", "float64"):
        raise ValueError("KL reductions require float32 or float64")
    for section, names in {
        "sampling": ("max_episodes_per_task", "max_steps_per_episode"),
        "scoring": ("pair_batch_size", "max_scored_pairs", "max_working_memory_bytes"),
        "selection": ("top_k", "max_unique_features"),
        "rollout": ("max_total_rollouts",), "validation": ("max_model_calls",),
        "splits": ("discovery_per_task", "validation_per_task", "evaluation_per_task"),
        "analysis": ("bootstrap_replicates",),
    }.items():
        for name in names:
            if cfg[section][name] <= 0:
                raise ValueError(f"{section}.{name} must be positive")
    if cfg["selection"]["random_audit_features"] < 0:
        raise ValueError("random_audit_features must be nonnegative")
    methods = cfg["selection"]["methods"]
    allowed_methods = {"event_aligned", "window_mean", "task_mean", "head_full_vocab_kl_fixed_prefix",
                       "mean_readout_activation", "readout_activation_frequency", "mean_edit_norm"}
    if not methods or any(not isinstance(x, str) or x not in allowed_methods for x in methods) or len(set(methods)) != len(methods):
        raise ValueError("selection.methods must contain distinct supported predictors")
    tasks = cfg["sampling"]["task_ids"]
    if not tasks or any(type(x) is not int or x < 0 or x >= 10 for x in tasks) or len(set(tasks)) != len(tasks):
        raise ValueError("sampling.task_ids must be unique Spatial task IDs")
    if not 0 < cfg["analysis"]["confidence_level"] < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    if cfg["scope"]["layer_idx"] < 0 or cfg["rollout"]["hook_start_step"] < 0:
        raise ValueError("layer/hook indices must be nonnegative")
    for key in ("atol", "rtol", "max_argmax_mismatch"):
        if cfg["validation"][key] < 0:
            raise ValueError(f"validation.{key} must be nonnegative")
    if cfg["splits"]["evaluation_labels_previously_used"] not in (None, True, False):
        raise ValueError("evaluation_labels_previously_used must be boolean or null")
    if cfg["scoring"]["negative_tolerance"] < 0 or cfg["validation"]["max_argmax_mismatch"] > 1:
        raise ValueError("Invalid KL/parity tolerance")
    root = cfg["output"]["root_dir"]
    if root:
        # Generated inputs within a dedicated output tree are allowed. Original
        # model/SAE/discovery sources are never allowed to overlap that tree.
        protected = [cfg["inputs"][key] for key in
                     ("dense_dir", "source_episode_manifest", "sae_checkpoint", "event_scores_path",
                      "local_model_snapshot", "runtime_inputs")]
        assert_safe_output(root, protected)
    return cfg


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return validate_config(yaml.safe_load(stream) or {})
