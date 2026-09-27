"""Discovery-only review material, deliberately omitting Head ranks and outcomes."""
from __future__ import annotations

from collections import defaultdict
import random

from .metrics import METRIC_KEYS


def label_packet(statistics, cache, *, excluded_feature_ids, statistics_sha256, count=60, seed=20260927):
    if (statistics.get("synthetic") is not False or cache["manifest"].get("synthetic") is not False
            or statistics["identity"]["manifest_hash"] != cache["manifest"]["manifest_hash"]):
        raise ValueError("label packet needs matching real temporal statistics/cache")
    if not isinstance(count, int) or isinstance(count, bool) or not 30 <= count <= 200:
        raise ValueError("candidate count must be between 30 and 200")
    eligible = sorted(row["feature_id"] for row in statistics["feature_statistics"]
                      if row["status"] == "ok" and row["feature_id"] not in excluded_feature_ids)
    rng = random.Random(seed)
    rng.shuffle(eligible)
    chosen = eligible[:count]
    if len(chosen) < 30:
        raise ValueError("too few independently eligible calibration candidates")
    chosen_set = set(chosen)
    # Seven original prediction-readout latents are averaged after encoding.
    traces = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    identities = {}
    for row, ids, values in zip(cache["records"], cache["feature_ids"], cache["feature_values"]):
        uid, step = row["episode_uid"], row["step_in_episode"]
        identities[uid] = {key: row.get(key) for key in ("suite", "task_id", "task_episode_idx", "source_run_id")}
        for fid, value in zip(ids.tolist(), values.tolist()):
            if fid in chosen_set:
                traces[fid][uid][step] += value / 7
    by_id = {row["feature_id"]: row for row in statistics["feature_statistics"]}
    cards = []
    for fid in chosen:
        # Store all discovery episode traces (including exact zero episodes),
        # not only most-active videos. No outcome or Head order influences selection.
        episodes = [{"episode_uid": uid, **identities[uid],
                     "values": [traces[fid][uid].get(t, 0.) for t in range(length)]}
                    for uid, length in sorted(cache["manifest"]["episode_lengths"].items())]
        cards.append({"feature_id": fid, "statistics": {key: by_id[fid][key] for key in METRIC_KEYS},
                      "discovery_traces": episodes})
    return {"schema_version": "drvla_discovery_review_packet_v1", "statistics_sha256": statistics_sha256,
            "candidate_seed": seed, "candidate_order": chosen, "cards": cards,
            "excluded_previously_measured": sorted(excluded_feature_ids),
            "labels": "not_provided", "video_review": "required; resolve discovery source_run_id/task/trial to original videos",
            "instructions": ["Review discovery-only traces AND corresponding videos; do not consult Head ranks or outcomes.",
                             "Label general or episode_specific only with concrete semantic evidence; uncertain stays unlabeled.",
                             "Do not equate high episode coverage with meaningful/general behavior without video evidence.",
                             "Select exactly 15 unambiguous examples of each class; if unavailable, stop and expand review."]}
