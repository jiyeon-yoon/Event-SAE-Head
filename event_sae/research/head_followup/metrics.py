"""Action-restricted KL and Dr.VLA-statistics readout adaptation.

Dr.VLA reference: https://arxiv.org/html/2603.19183v2, Appendix C.1.
No published coefficients are used and no semantic labels are fabricated.
"""
from __future__ import annotations

import math

import torch

from event_sae.research.output_head.sensitivity import full_vocab_kl


def action_metrics(base, edited, action_token_ids):
    """Conditional KL plus probability mass diagnostics; not physical action error.

Conditioning on the action subset hides probability leaking outside it, so
always return the subset masses and mass change along with the KL.
"""
    if base.ndim != 2 or base.shape != edited.shape or not torch.isfinite(base).all() or not torch.isfinite(edited).all():
        raise ValueError("finite matching [rows, vocabulary] logits required")
    ids = list(action_token_ids)
    if (not ids or len(set(ids)) != len(ids)
            or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < base.shape[-1] for i in ids)):
        raise ValueError("invalid action token mapping")
    original, changed = base.float(), edited.float()
    p_mass = (torch.logsumexp(original[:, ids], -1) - torch.logsumexp(original, -1)).exp()
    q_mass = (torch.logsumexp(changed[:, ids], -1) - torch.logsumexp(changed, -1)).exp()
    return {
        "action_conditional_kl": full_vocab_kl(original[:, ids], changed[:, ids]),
        "full_vocab_kl": full_vocab_kl(original, changed),
        "base_action_mass": p_mass, "edited_action_mass": q_mass,
        "action_mass_abs_change": (p_mass - q_mass).abs(),
        "conditional_action_argmax_flip": (original[:, ids].argmax(-1) != changed[:, ids].argmax(-1)).float(),
    }


class GeneralityAccumulator:
    """One complete episode at a time; equal-episode, active-episode denominators.

Inputs are mean latents AFTER separately encoding all seven original forwards.
Positive-but-never-above-threshold episodes have undefined run length. They
are counted, not silently treated as inactive or imputed as zero-length runs.
"""
    def __init__(self, dictionary_size: int, threshold: float = .1):
        if isinstance(dictionary_size, bool) or not isinstance(dictionary_size, int) or dictionary_size < 1:
            raise ValueError("invalid dictionary size")
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("invalid onset threshold")
        self.width, self.threshold, self.episodes = dictionary_size, threshold, set()
        self.active = torch.zeros(dictionary_size, dtype=torch.int64)
        self.zero_onset = torch.zeros_like(self.active)
        self.onsets = torch.zeros(dictionary_size, dtype=torch.float64)
        self.peaks = torch.zeros_like(self.onsets)
        self.lengths = torch.zeros_like(self.onsets)

    def add_episode(self, episode_uid: str, steps: list[int], values):
        if not isinstance(episode_uid, str) or not episode_uid or episode_uid in self.episodes:
            raise ValueError("episode identity must be nonempty and unique")
        if (not steps or any(isinstance(t, bool) or not isinstance(t, int) for t in steps)
                or steps != list(range(len(steps)))):
            raise ValueError("complete consecutive steps starting at zero required; sampled cache is invalid")
        if (values.ndim != 2 or tuple(values.shape) != (len(steps), self.width)
                or not torch.isfinite(values).all() or (values < 0).any()):
            raise ValueError("invalid nonnegative temporal latent matrix")
        values = values.detach().cpu()
        active = (values > 0).any(0)
        state = torch.zeros(self.width, dtype=torch.bool)
        count, duration = torch.zeros(self.width, dtype=torch.int64), torch.zeros(self.width, dtype=torch.int64)
        for row in values:
            next_state = torch.where(row > self.threshold, True, torch.where(row == 0, False, state))
            count += (~state & next_state).long()
            duration += next_state.long()
            state = next_state
        self.episodes.add(episode_uid)
        self.active += active.long()
        self.zero_onset += (active & (count == 0)).long()
        self.onsets += count
        self.peaks += values.max(0).values.double()
        valid = count > 0
        self.lengths[valid] += duration[valid].double() / count[valid] / len(steps)

    def result(self):
        if not self.episodes:
            raise ValueError("no complete episodes")
        output = []
        for fid in range(self.width):
            n = int(self.active[fid])
            missing = int(self.zero_onset[fid])
            output.append({
                "feature_id": fid, "episode_coverage": n / len(self.episodes),
                "mean_onset_count": float(self.onsets[fid]) / n if n else None,
                "mean_episode_max_activation": float(self.peaks[fid]) / n if n else None,
                "relative_run_length": float(self.lengths[fid]) / n if n and not missing else None,
                "active_episodes": n, "zero_onset_active_episodes": missing,
                "status": "inactive" if not n else "undefined_run_length" if missing else "ok",
            })
        return {"schema_version": "drvla_readout_statistics_v1", "num_episodes": len(self.episodes),
                "feature_statistics": output, "onset_threshold": self.threshold,
                "generality_score": "not_computed_requires_independent_labels",
                "scope": "readout_adaptation_not_original_mean_pooled_SAE"}


METRIC_KEYS = ("episode_coverage", "mean_onset_count", "mean_episode_max_activation", "relative_run_length")


def calibrate_generality(statistics, labels, *, excluded_feature_ids):
    """Fit the proposed comparator, not the SAE, with explicit label provenance."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    if (labels.get("schema_version") != "drvla_discovery_labels_v1"
            or labels.get("used_outcomes") is not False or labels.get("used_head_rank") is not False
            or labels.get("split_role") != "discovery" or not labels.get("evaluator")):
        raise ValueError("discovery-only labels with evaluator and explicit no-outcome/no-rank declaration required")
    rows = labels.get("labels", [])
    if len(rows) != 30 or sum(r.get("label") == "general" for r in rows) != 15 or sum(r.get("label") == "episode_specific" for r in rows) != 15:
        raise ValueError("exactly 30 unambiguous labels, 15 per class required")
    ids = [r.get("feature_id") for r in rows]
    if (any(isinstance(i, bool) or not isinstance(i, int) for i in ids)
            or len(set(ids)) != len(ids) or set(ids) & set(excluded_feature_ids)):
        raise ValueError("duplicate/invalid or previously measured calibration feature")
    by_id = {r["feature_id"]: r for r in statistics["feature_statistics"]}
    if any(i not in by_id or by_id[i]["status"] != "ok" or not r.get("evidence") for i, r in zip(ids, rows)):
        raise ValueError("all labels need supported statistics and independent semantic evidence")
    def vector(fid):
        values = [by_id[fid][key] for key in METRIC_KEYS]
        if any(v is None or not math.isfinite(v) for v in values):
            raise ValueError("nonfinite classifier metric")
        return values
    x = np.asarray([vector(fid) for fid in ids], dtype=float)
    y = np.asarray([int(r["label"] == "general") for r in rows])
    def estimator():
        return make_pipeline(StandardScaler(), LogisticRegression(C=1, solver="lbfgs", max_iter=1000, random_state=20260927))
    correct = []
    for i in range(len(rows)):
        mask = np.arange(len(rows)) != i
        fold = estimator().fit(x[mask], y[mask])
        correct.append(bool(fold.predict(x[i:i + 1])[0] == y[i]))
    model = estimator().fit(x, y)
    valid = sorted(fid for fid, row in by_id.items() if row["status"] == "ok")
    decisions = model.decision_function(np.asarray([vector(fid) for fid in valid]))
    ranked = sorted(zip(valid, decisions), key=lambda pair: (-float(pair[1]), pair[0]))
    scale, regression = model.steps[0][1], model.steps[1][1]
    return {"schema_version": "drvla_recalibrated_scores_v1", "feature_scores": [
        {"feature_id": fid, "generality_decision": float(value)} for fid, value in ranked],
        "leave_one_feature_out_accuracy": sum(correct) / len(correct),
        "validation_scope": "descriptive_small_visually_selected_calibration_set",
        "rank_scope": "includes_calibration_features_not_independent_semantic_validation",
        "calibration_feature_ids": ids, "metric_order": list(METRIC_KEYS),
        "scaler_mean": scale.mean_.tolist(), "scaler_scale": scale.scale_.tolist(),
        "coefficients": regression.coef_[0].tolist(), "intercept": float(regression.intercept_[0]),
        "sae_retrained": False}
