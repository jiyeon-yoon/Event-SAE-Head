"""Post-hoc common-panel comparisons; unmeasured outcomes are never imputed."""
from __future__ import annotations

import math

from event_sae.research.output_head.provenance import fingerprint
from event_sae.research.output_head.results import spearman_correlation

HEAD = "head_full_vocab_kl_fixed_prefix"
OLD = (HEAD, "mean_edit_norm", "mean_readout_activation", "readout_activation_frequency", "full_vocab_argmax_flip_rate")


def vector(rows, key):
    out = {}
    for row in rows:
        fid, value = row["feature_id"], row[key]
        if (isinstance(fid, bool) or not isinstance(fid, int) or fid < 0 or fid in out
                or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError("duplicate/invalid/nonfinite score or feature ID")
        out[fid] = float(value)
    if not out:
        raise ValueError("empty score vector")
    return out


def compare_scores(old, plan, paired, *, action=None, drvla=None, k=3):
    if (plan.get("plan_hash") != fingerprint({key: v for key, v in plan.items() if key != "plan_hash"})
            or plan.get("scope") != old.get("scope")
            or old.get("synthetic") is not False or plan.get("synthetic") is not False
            or paired.get("synthetic") is not False):
        raise ValueError("frozen plan/score scope mismatch or non-real input")
    if k != 3:
        raise ValueError("Stage 1 comparison keeps the frozen Top-3 rule")
    scope = old["scope"]
    protocol, identity = paired.get("protocol", {}), old.get("identity", {})
    if (protocol.get("sae_sha256") != scope.get("sae_sha256")
            or protocol.get("model_revision") != identity.get("model_revision")
            or protocol.get("model_code_revision") != identity.get("code_revision")
            or protocol.get("eval_cases") != plan.get("eval_cases")
            or not protocol.get("eval_cases") or protocol.get("alpha") != 0
            or protocol.get("hook_start_step") != 0):
        raise ValueError("outcome protocol differs from the frozen model/SAE/evaluation cases")
    vectors = {name: vector(old["feature_scores"], name) for name in OLD}
    vectors["event_aligned"] = vector([
        {"feature_id": int(fid), "score": value} for fid, value in plan["score_vectors"]["event_aligned"].items()], "score")
    active = {row["feature_id"] for row in old["feature_scores"] if row["num_active_readouts"] > 0}
    for data, metrics in ((action, ("action_conditional_kl", "action_mass_abs_change", "conditional_action_argmax_flip")),
                          (drvla, ("generality_decision",))):
        if data is None:
            continue
        identity = data.get("scope", data.get("identity", {}))
        if data.get("synthetic") is not False or any(identity.get(key) != scope[key] for key in
                ("sae_sha256", "discovery_manifest_hash", "split_manifest_hash")):
            raise ValueError("new scores do not match frozen SAE/discovery/split")
        if data is action and (identity.get("sample_manifest_hash") != scope["sample_manifest_hash"]
                              or data.get("schema_version") != "head_action_scores_v1"):
            raise ValueError("action scoring sample/schema mismatch")
        if data is drvla and data.get("schema_version") != "drvla_recalibrated_scores_v1":
            raise ValueError("Dr.VLA scores require independent calibration")
        for name in metrics:
            vectors[name] = vector(data["feature_scores"], name)
    universe = set(vectors[HEAD])
    if any(set(scores) - universe for scores in vectors.values()):
        raise ValueError("new scores contain unknown feature IDs")
    if action is not None and set(vectors["action_conditional_kl"]) != universe:
        raise ValueError("action scores must cover the complete frozen dictionary")
    drops = vector(paired["feature_effects"], "drop")
    if set(drops) != set(plan["feature_ids"]):
        raise ValueError("outcome panel differs from original plan")
    if (len({r["protocol_id"] for r in paired["feature_effects"]}) != 1
            or any(r["protocol_id"] != paired["protocol_id"] for r in paired["feature_effects"])):
        raise ValueError("mixed outcome protocols")
    common = sorted(active.intersection(*(set(v) for v in vectors.values())))
    panel = sorted(set(drops).intersection(*(set(v) for v in vectors.values())))
    results = {}
    top_head = sorted(vectors[HEAD], key=lambda i: (-vectors[HEAD][i], i))[:k]
    for name, scores in vectors.items():
        top = sorted(scores, key=lambda i: (-scores[i], i))[:k]
        missing = [fid for fid in top if fid not in drops]
        results[name] = {
            "eligible_features": len(scores), "top3": top,
            "head_top3_intersection": sorted(set(top) & set(top_head)),
            "measured_top3_count": sum(fid in drops for fid in top), "unmeasured_top3": missing,
            "top3_mean_drop": sum(drops[fid] for fid in top) / k if len(top) == k and not missing else None,
            "spearman_with_head_common_active": spearman_correlation([vectors[HEAD][i] for i in common], [scores[i] for i in common]),
            "spearman_with_drop_common_measured": spearman_correlation([scores[i] for i in panel], [drops[i] for i in panel]),
        }
    return {"schema_version": "head_stage1_comparison_v1", "status": "compared",
            "new_rollouts": 0, "action_scores_present": action is not None, "drvla_scores_present": drvla is not None,
            "scope": scope, "frozen_plan_hash": plan["plan_hash"], "outcome_protocol_id": paired["protocol_id"],
            "action_full_vocab_replay_max_abs_error": action.get("replay_full_vocab_max_feature_abs_error") if action else None,
            "common_active_feature_ids": common, "common_measured_feature_ids": panel,
            "comparators": results,
            "limitations": ["retrospective, selected-panel comparison, not confirmatory/global outcome correlation",
                "unmeasured Top-3 outcomes unavailable; no substitution with lower-ranked measured features",
                "Dr.VLA uses full timelines; Head/action metrics use the same frozen eight-step sample",
                "generality is not established by drop or KL; conditional-token KL is not physical action error"]}


def comparison_markdown(result):
    lines = ["# Stage 1 — offline ranking comparison", "",
             f"New rollouts: 0. Common active features: {len(result['common_active_feature_ids'])}; "
             f"common measured panel: {len(result['common_measured_feature_ids'])}.", "",
             f"Action scores available: {result['action_scores_present']}. Dr.VLA scores available: {result['drvla_scores_present']}.", "",
             "| Metric | Top-3 | Measured / 3 | Mean drop if all measured | Head rank correlation* |",
             "|---|---|---:|---:|---:|"]
    for name, row in result["comparators"].items():
        lines.append(f"| {name} | {row['top3']} | {row['measured_top3_count']} | {row['top3_mean_drop']} | {row['spearman_with_head_common_active']} |")
    lines += ["", "*Same named common active population, not all zero-tied dictionary entries.", "",
              *["- " + item for item in result["limitations"]], ""]
    return "\n".join(lines)
