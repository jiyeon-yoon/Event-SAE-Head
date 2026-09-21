"""Event helpers; optional vision/VLM dependencies are loaded only on demand."""

from importlib import import_module

_MODULE_EXPORTS = {
    "annotate": ("annotate_clusters", "call_gemini", "load_api_key", "parse_annotation_response"),
    "build_features": ("EpisodeStateSummary", "VisionEmbedder", "build_episode_state_index",
                       "build_event_features", "state_vector_from_record"),
    "cluster": ("build_task_vectors", "cluster_events", "select_exemplars"),
    "extract_media": ("EpisodeRecords", "extract_keyframe_media", "find_episode_video",
                      "fit_frame_window", "load_episode_records", "load_waypoint_summary"),
    "io": ("load_jsonl", "write_jsonl"),
    "prompts": ("PHASE_DESCRIPTIONS", "PHASE_LABELS", "PROMPT_VERSION", "build_cluster_annotation_prompt"),
}
_EXPORTS = {name: module for module, names in _MODULE_EXPORTS.items() for name in names}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value

__all__ = [
    "EpisodeRecords",
    "EpisodeStateSummary",
    "PHASE_DESCRIPTIONS",
    "PHASE_LABELS",
    "PROMPT_VERSION",
    "VisionEmbedder",
    "annotate_clusters",
    "build_cluster_annotation_prompt",
    "build_episode_state_index",
    "build_event_features",
    "build_task_vectors",
    "call_gemini",
    "cluster_events",
    "extract_keyframe_media",
    "find_episode_video",
    "fit_frame_window",
    "load_api_key",
    "load_episode_records",
    "load_jsonl",
    "load_waypoint_summary",
    "parse_annotation_response",
    "select_exemplars",
    "state_vector_from_record",
    "write_jsonl",
]
