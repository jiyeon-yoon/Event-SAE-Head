"""M4/M5 adapter integration against a tiny fake runtime, entirely on CPU.

The fixtures deliberately mimic the non-synthetic input gate so the guarded
adapter can be exercised. Every emitted parity artifact is explicitly marked
synthetic by the fixture writer; none can certify a real OpenVLA execution.
"""

import copy
import io
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

import event_sae.openvla.activations as activations
import event_sae.openvla.intervene as original_hook
from event_sae.research.output_head import head as head_module
from event_sae.research.output_head import runtime
from event_sae.research.output_head.config import DEFAULTS
from event_sae.research.output_head.head import EDIT_PARITY_CHECKS, HEAD_PARITY_CHECKS, ExactRMSNorm, OutputHeadBundle
from event_sae.research.output_head.provenance import fingerprint, read_json, sha256_file


class FakeSAE:
    def encode(self, hidden):
        return torch.cat((hidden.clamp_min(0), torch.zeros(len(hidden), 1, device=hidden.device)), dim=-1)

    def decode(self, encoded):
        directions = torch.tensor([[.5, .1], [0., .5], [.1, .2]], device=encoded.device)
        return encoded @ directions + .1


class FakeLayer(torch.nn.Module):
    def forward(self, hidden):
        return (hidden,)


class FakeVLA(torch.nn.Module):
    def __init__(self, *, cached=True):
        super().__init__()
        self.language_model = torch.nn.Module()
        self.language_model.model = torch.nn.Module()
        self.language_model.model.layers = torch.nn.ModuleList([FakeLayer()])
        self.language_model.model.norm = ExactRMSNorm(torch.tensor([.9, 1.1]), 1e-5)
        self.language_model.lm_head = torch.nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            self.language_model.lm_head.weight.copy_(torch.tensor([[1., -.2], [-.3, .7], [.2, -.9]]))
        self.cached = cached
        self.query_count = 0
        self.forward_count = 0
        self.forward_counts = None

    def predict_action(self, *, hidden, unnorm_key, do_sample, input_ids=None, attention_mask=None):
        assert unnorm_key == "libero_spatial" and do_sample is False
        self.query_count += 1
        chosen = []
        count = 7 if self.forward_counts is None else self.forward_counts[(self.query_count - 1) % len(self.forward_counts)]
        for index in range(count):
            values = hidden if index == 0 or not self.cached else hidden[:, -1:]
            result = self.language_model.model.layers[-1](values)[0]
            logits = self.language_model.lm_head(self.language_model.model.norm(result))
            chosen.append(int(logits[0, -1].argmax()))
            self.forward_count += 1
        return chosen


@pytest.fixture
def fake_runtime(monkeypatch, tmp_path):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["scope"]["layer_idx"] = 0
    cfg["model"].update(revision="fake-model", code_revision="fake-code", expected_live_hidden_dtype="float32")
    cfg["scoring"].update(device="cpu", pair_batch_size=3)
    cfg["validation"].update(max_model_calls=4)
    cfg["output"]["root_dir"] = str(tmp_path / "synthetic-output")
    source = tmp_path / "fake-snapshot"
    source.mkdir()
    (source / "dataset_statistics.json").write_text(json.dumps({"libero_spatial": {"synthetic": True}}))
    input_path = tmp_path / "synthetic-runtime-inputs.pt"
    generation_path = tmp_path / "synthetic-generation.json"
    generation_path.write_text(json.dumps({"synthetic": True, "action_dim": 7}))
    fixture = {"synthetic": False, "test_only_fake_runtime": True,
               "model_revision": "fake-model", "code_revision": "fake-code",
               "processor_identity": "fake-processor", "episodes": [],
               "calls": [{"hidden": torch.tensor([[[1., .2], [.6, .3], [.8, .4]]]),
                          "input_ids": torch.tensor([[1, 2, 3]]),
                          "attention_mask": torch.ones(1, 3, dtype=torch.int64)}]}
    torch.save(fixture, input_path)
    cfg["inputs"].update(runtime_inputs=str(input_path), local_model_snapshot=str(source),
                         generation_manifest=str(generation_path))
    model, sae = FakeVLA(), FakeSAE()
    live = model.language_model
    head = OutputHeadBundle(live.model.norm.weight, live.lm_head.weight, {
        "norm": {"implementation": "llama_rms_norm_v1", "eps": 1e-5},
        "target_layer": 0, "num_layers": 1, "hidden_dtype": "float32"})
    identity = {"model_revision": "fake-model", "code_revision": "fake-code",
                "processor_identity": "fake-processor", "layer_idx": 0,
                "sae_checkpoint_hash": "fake-sae", "implementation": "fake-adapter-test"}
    monkeypatch.setattr(runtime, "numerical_identity", lambda config: dict(identity))
    monkeypatch.setattr(runtime, "output_root", lambda config: Path(config["output"]["root_dir"]))
    monkeypatch.setattr(runtime, "_sae_paths", lambda config: (tmp_path / "fake-sae.pt", tmp_path / "fake-config.json"))
    monkeypatch.setattr(runtime, "_head_path", lambda config: tmp_path / "fake-head")
    monkeypatch.setattr(runtime, "_offline_environment", lambda: None)
    monkeypatch.setattr(head_module, "load_output_head", lambda *args, **kwargs: head)
    sae_config = {"trainer": {"submodule_name": "post_mlp_residual", "layer": 0,
                              "activation_dim": 2, "dict_size": 3}}
    loader = lambda *args, **kwargs: (sae, sae_config)
    monkeypatch.setattr(activations, "load_batch_topk_sae", loader)
    monkeypatch.setattr(original_hook, "load_batch_topk_sae", loader)
    model_loads = []

    def from_pretrained(path, **kwargs):
        model_loads.append({"path": path, **kwargs})
        assert kwargs["local_files_only"] is True
        assert kwargs["code_revision"] == "fake-code"
        return model

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoModelForVision2Seq = SimpleNamespace(from_pretrained=from_pretrained)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    writer = runtime.write_once_json

    def synthetic_writer(path, payload):
        marked = {**payload, "synthetic": True, "test_only_fake_runtime": True}
        if "result_hash" in marked:
            marked["result_hash"] = fingerprint({key: value for key, value in marked.items() if key != "result_hash"})
        writer(path, marked)

    monkeypatch.setattr(runtime, "write_once_json", synthetic_writer)
    return SimpleNamespace(cfg=cfg, model=model, head=head, sae=sae, fixture=fixture,
                           input_path=input_path, identity=identity, loads=model_loads,
                           output=Path(cfg["output"]["root_dir"]))


def test_validate_runtime_calls_actual_legacy_hook_and_covers_all_checks(fake_runtime):
    f = fake_runtime
    result = runtime.validate_runtime(f.cfg)
    assert result["status"] == "passed"
    assert len(f.loads) == 1
    assert f.model.query_count == 4 and f.model.forward_count == 28
    baseline = read_json(f.output / "validation/runtime_parity.json")
    edited = read_json(f.output / "validation/edit_parity.json")
    assert baseline["synthetic"] is edited["synthetic"] is True
    assert baseline["model_action_queries"] == 4
    assert baseline["observations"] == 28
    assert {item["call_index"] for item in baseline["checks"]["baseline_logits"]["comparisons"]} == {0}
    assert f.model.norm_stats == {"libero_spatial": {"synthetic": True}}
    assert set(baseline["checks"]) == set(HEAD_PARITY_CHECKS)
    assert set(edited["checks"]) == set(EDIT_PARITY_CHECKS)
    assert all(item["status"] == "passed" for item in edited["checks"].values())
    assert edited["checks"]["prefill"]["num_forwards"] == 3
    assert edited["checks"]["cached"]["num_forwards"] == 18
    assert not f.model.language_model.model.layers[0]._forward_hooks
    assert not f.model.language_model.lm_head._forward_hooks
    # The original hook wrote diagnostics, requiring the real log_file argument.
    diagnostics = list((f.output / "validation/active_alpha0").glob("intervene_*_records.jsonl"))
    assert len(diagnostics) == 1 and diagnostics[0].read_text().strip()


def test_validate_call_budget_counts_all_four_conditions_before_loading(fake_runtime):
    f = fake_runtime
    f.cfg["validation"]["max_model_calls"] = 3
    with pytest.raises(ValueError, match="validation budget"):
        runtime.validate_runtime(f.cfg)
    assert f.loads == [] and f.model.query_count == 0


def test_validate_rejects_synthetic_inputs_before_model_execution(fake_runtime):
    f = fake_runtime
    torch.save({**f.fixture, "synthetic": True}, f.input_path)
    with pytest.raises(ValueError, match="non-synthetic runtime"):
        runtime.validate_runtime(f.cfg)
    assert not f.loads


def test_validate_rejects_dtype_even_when_tensor_values_equal(fake_runtime):
    f = fake_runtime
    f.head.head_weight = f.head.head_weight.double()
    assert torch.equal(f.head.head_weight, f.model.language_model.lm_head.weight)
    with pytest.raises(ValueError, match="not identical"):
        runtime.validate_runtime(f.cfg)
    assert f.model.query_count == 0


def test_validate_missing_cached_forward_fails_gate_and_removes_hooks(fake_runtime):
    f = fake_runtime
    f.model.cached = False
    with pytest.raises(ValueError, match="prefill/cached forward structure"):
        runtime.validate_runtime(f.cfg)
    assert not (f.output / "validation/edit_parity.json").exists()
    assert f.model.query_count == 1
    assert not f.model.language_model.model.layers[0]._forward_hooks


def test_validate_resume_rechecks_current_identity(fake_runtime, monkeypatch):
    f = fake_runtime
    runtime.validate_runtime(f.cfg)
    rechecks = []
    monkeypatch.setattr(runtime, "load_parity", lambda cfg, identity: rechecks.append(identity))
    result = runtime.validate_runtime(f.cfg)
    assert result["resumed"] is True
    assert rechecks == [f.identity] and f.model.query_count == 4
    monkeypatch.setattr(runtime, "load_parity", lambda *args: (_ for _ in ()).throw(ValueError("stale parity")))
    with pytest.raises(ValueError, match="stale parity"):
        runtime.validate_runtime(f.cfg)


def test_validate_rejects_generation_count_mismatch_and_cleans_up(fake_runtime):
    f = fake_runtime
    Path(f.cfg["inputs"]["generation_manifest"]).write_text(json.dumps({"synthetic": True, "action_dim": 8}))
    with pytest.raises(ValueError, match="forward count disagrees"):
        runtime.validate_runtime(f.cfg)
    assert f.model.query_count == 1
    assert not f.model.language_model.model.layers[0]._forward_hooks


def test_validate_does_not_allow_six_and_eight_forward_queries_to_cancel(fake_runtime):
    f = fake_runtime
    f.cfg["validation"]["max_model_calls"] = 8
    f.fixture["calls"].append(copy.deepcopy(f.fixture["calls"][0]))
    torch.save(f.fixture, f.input_path)
    f.model.forward_counts = [6, 8]
    # The aggregate would be the nominal 2*7, but the first query is already
    # malformed and must fail before a second query can hide that discrepancy.
    with pytest.raises(ValueError, match="one query"):
        runtime.validate_runtime(f.cfg)
    assert f.model.query_count == 1 and f.model.forward_count == 6
    assert not f.model.language_model.model.layers[0]._forward_hooks
    assert not f.model.language_model.lm_head._forward_hooks


def test_validate_rejects_padding_before_predict_action(fake_runtime):
    f = fake_runtime
    f.fixture["calls"][0]["attention_mask"][0, 0] = 0
    torch.save(f.fixture, f.input_path)
    with pytest.raises(ValueError, match="unpadded batch-size-1"):
        runtime.validate_runtime(f.cfg)
    assert f.model.query_count == 0


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_approved_plan_rejects_missing_auth_tampering_and_dirty_code(tmp_path, monkeypatch):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["output"]["root_dir"] = str(tmp_path)
    cfg["rollout"]["eval_manifest"] = str(tmp_path / "eval.json")
    _write_json(tmp_path / "eval.json", {"synthetic": True, "cases": []})
    _write_json(tmp_path / "scores/scores.json", {"synthetic": True, "feature_scores": []})
    # This synthetic unit fixture claims the runtime gate's expected status;
    # no model or environment is imported or executed by verify_approved_plan.
    plan = {"synthetic": False, "test_only_fake_runtime": True, "status": "planned",
            "experiment_config_hash": fingerprint(cfg), "implementation_fingerprint": "fake-implementation",
            "score_artifact_hash": sha256_file(tmp_path / "scores/scores.json"),
            "base_eval_config_hash": sha256_file(cfg["rollout"]["base_eval_config"]),
            "eval_manifest_hash": sha256_file(tmp_path / "eval.json"), "total_rollouts": 2,
            "conditions": [{"mode": "raw", "num_cases": 1}, {"mode": "identity", "num_cases": 1}]}
    plan["plan_hash"] = fingerprint(plan)
    _write_json(tmp_path / "plan/rollout_plan.json", plan)
    monkeypatch.setattr(runtime, "output_root", lambda cfg: tmp_path)
    monkeypatch.setattr(runtime, "numerical_implementation_fingerprint", lambda: "fake-implementation")
    monkeypatch.setattr(runtime, "git_identity", lambda root: {"dirty": False, "commit": "fake-commit"})
    with pytest.raises(ValueError, match="Explicit plan hash"):
        runtime.verify_approved_plan(cfg, "", "fake-commit")
    assert runtime.verify_approved_plan(cfg, plan["plan_hash"], "fake-commit") == plan
    monkeypatch.setattr(runtime, "git_identity", lambda root: {"dirty": True, "commit": "fake-commit"})
    with pytest.raises(ValueError, match="clean code revision"):
        runtime.verify_approved_plan(cfg, plan["plan_hash"], "fake-commit")
    plan["total_rollouts"] = 0
    _write_json(tmp_path / "plan/rollout_plan.json", plan)
    with pytest.raises(ValueError, match="tampered"):
        runtime.verify_approved_plan(cfg, plan["plan_hash"], "fake-commit")


def _install_fake_evaluator(f, monkeypatch, *, identity_mismatch=False):
    from event_sae.openvla.eval import config as eval_config

    f.cfg["model"]["expected_live_hidden_dtype"] = "bfloat16"
    f.model.to(dtype=torch.bfloat16)

    cases = [{"task_id": 0, "task_episode_idx": index, "initial_state_sha256": f"state-{index}"}
             for index in (8, 17)]
    conditions = [{"condition_id": "feature-0", "mode": "intervention", "feature_id": 0, "alpha": 0., "num_cases": 2},
                  {"condition_id": "identity", "mode": "identity", "feature_id": 0, "alpha": 1., "num_cases": 2},
                  {"condition_id": "raw", "mode": "raw", "feature_id": None, "alpha": None, "num_cases": 2}]
    plan = {"conditions": conditions, "eval_cases": cases, "total_rollouts": 6}
    monkeypatch.setattr(runtime, "verify_approved_plan", lambda *args: plan)
    monkeypatch.setattr(runtime, "load_parity", lambda *args: {})
    native_cfg = eval_config.RunConfig()
    native_cfg.model.revision, native_cfg.model.code_revision = "fake-model", "fake-code"
    monkeypatch.setattr(eval_config, "load_config", lambda path: copy.deepcopy(native_cfg))
    seen = []

    def evaluate(cfg, extra_hook_applier, trial_indices_by_task, expected_initial_states):
        condition = Path(cfg.logging.root_dir).name
        seen.append(condition)
        assert trial_indices_by_task == {0: [8, 17]}
        assert expected_initial_states == {(0, 8): "state-8", (0, 17): "state-17"}
        assert cfg.env.per_episode_seed is True and cfg.sae_collect.enabled is False
        folder = Path(cfg.logging.root_dir) / "synthetic-native-run"
        folder.mkdir(parents=True)
        if extra_hook_applier is not None:
            extra_hook_applier(model=f.model, cfg=cfg, run_dir=folder, log_file=io.StringIO())
        f.model._sae_hook_context = {"episode_num": 1, "step_in_episode": 0}
        f.model.predict_action(hidden=f.fixture["calls"][0]["hidden"].to(torch.bfloat16),
                               unnorm_key="libero_spatial", do_sample=False)
        rows = [{**case, "success": True, "caught_exception": None, "synthetic": True} for case in cases]
        (folder / "episode_results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        actions = {"task-0": [1, 2]}
        if identity_mismatch and condition == "identity":
            actions = {"task-0": [9, 2]}
        _write_json(folder / "actions.json", actions)
        return SimpleNamespace(run_dir=str(folder), episode_results_path=str(folder / "episode_results.jsonl"))

    module = ModuleType("event_sae.openvla.eval.runner")
    module.eval_libero = evaluate
    monkeypatch.setitem(sys.modules, "event_sae.openvla.eval.runner", module)
    return seen


def test_run_plan_orders_identity_before_intervention_and_rejects_synthetic_resume(fake_runtime, monkeypatch):
    f = fake_runtime
    seen = _install_fake_evaluator(f, monkeypatch)
    result = runtime.run_plan(f.cfg, "fake-plan", "fake-revision")
    assert result["status"] == "completed" and result["total_rollouts"] == 6
    assert seen == ["raw", "identity", "feature-0"]
    assert read_json(f.output / "runs/identity/result.json")["identity_check"] == "passed"
    assert read_json(f.output / "runs/feature-0/result.json")["hook_metrics"]["post_intervention_active_feature_values"] == 0
    # The fixture writer labels the mocked outputs synthetic, which must not
    # later satisfy production resume validation.
    with pytest.raises(ValueError, match="provenance does not match"):
        runtime.run_plan(f.cfg, "fake-plan", "fake-revision")
    assert seen == ["raw", "identity", "feature-0"]
    assert not f.model.language_model.model.layers[0]._forward_hooks


def test_run_plan_identity_action_mismatch_stops_before_feature_rollouts(fake_runtime, monkeypatch):
    f = fake_runtime
    seen = _install_fake_evaluator(f, monkeypatch, identity_mismatch=True)
    with pytest.raises(ValueError, match="identity action sequences differ"):
        runtime.run_plan(f.cfg, "fake-plan", "fake-revision")
    assert seen == ["raw", "identity"]
    assert not (f.output / "runs/feature-0").exists()


@pytest.mark.parametrize("corruption, message", [
    ("missing_identity", "identity check has not passed"),
    ("wrong_alpha", "condition or provenance"),
    ("wrong_state", "cases differ"),
    ("missing_completion", "completion evidence"),
    ("missing_hash", "content fingerprint"),
])
def test_result_resume_revalidates_scientific_condition_and_case_evidence(
    fake_runtime, monkeypatch, corruption, message
):
    f = fake_runtime
    _install_fake_evaluator(f, monkeypatch)
    runtime.run_plan(f.cfg, "fake-plan", "fake-revision")
    payload = read_json(f.output / "runs/identity/result.json")
    # Only this in-memory copy emulates a real certificate for checking each
    # guard. The fake result artifact on disk remains explicitly synthetic.
    payload["synthetic"] = False
    condition = {"mode": "identity", "feature_id": 0, "alpha": 1.0}
    cases = [{"task_id": 0, "task_episode_idx": index, "initial_state_sha256": f"state-{index}"}
             for index in (8, 17)]
    payload["result_hash"] = fingerprint({key: value for key, value in payload.items() if key != "result_hash"})
    runtime.validate_condition_result(payload, condition, cases, payload["protocol_id"], f.identity, "fake-revision")
    if corruption == "missing_identity":
        payload.pop("identity_check")
    elif corruption == "wrong_alpha":
        payload["alpha"] = 0.0
    elif corruption == "wrong_state":
        payload["episodes"][0]["initial_state_sha256"] = "another-state"
    elif corruption == "missing_completion":
        payload["episodes"][0].pop("caught_exception")
    if corruption == "missing_hash":
        payload.pop("result_hash")
    else:
        payload["result_hash"] = fingerprint({key: value for key, value in payload.items() if key != "result_hash"})
    with pytest.raises(ValueError, match=message):
        runtime.validate_condition_result(payload, condition, cases, payload["protocol_id"], f.identity, "fake-revision")
