# OpenVLA 출력층 민감도: 구현 및 실행 안내

전체 10-task 후속 실행은 [현재 실행 안내](run_output_head_full10.md)를 따른다.

## 구현 범위와 현재 실행 상태

기존 OpenVLA 마지막 decoder residual에서 feature 하나를 제거한 뒤, 실제 final
norm과 lm_head의 full-vocabulary 분포 변화를 계산한다. 주 점수는
`head_full_vocab_kl_fixed_prefix = KL(p_base || p_edit)`다. 기존 문맥에서의 국소
변화이며, 편집된 7차원 counterfactual action sequence나 성공률 감소 자체가 아니다.

2026-09-20에 M0~M3 코드를 구현했고, 2026-09-21에 RTX 4090에서 M4 pilot을
실행했다. 전체 명령·artifact·결과는
[출력층 민감도 pilot 재현 기록](reproduce_output_head_sensitivity_pilot.md)에 있다.
Pod 종료 후 복구와 다음 실험 준비는
[재시작 안내](resume_output_head_sensitivity.md)를 참고한다.

| 단계 | 현재 상태 | 실제 확인 범위 |
|---|---|---|
| M0 | 완료 | 실제 입력, revision, checksum, provenance, 실행 차단 |
| M1 | 완료 | task 0·1의 5,600 readout cache와 local head export/load |
| M2 | 완료 | 32,768 feature, 308,780 active pair의 reference Head-KL score |
| M3 | 완료 | exact Event-score scope adapter, 16-feature plan, paired 분석 |
| M4 | **pilot 완료** | 실모델 BF16 parity, GPU score, 72 LIBERO rollouts |
| M5 | **미실행** | 독립 held-out task/state의 confirmatory 평가 |

Synthetic CPU 테스트는 수치·연결·안전장치 테스트이며 그 자체는 연구 가설의
실증 결과가 아니다. 실증 결과는 따로 수행한 M4 pilot에서 나왔으며, 평가가
2 tasks·4 cases에 그친 overlapping pilot이므로 확증 결론은 아니다.
기존 `Event-SAE-Baseline`, `Event-SAE-Pipeline` 또는 그 입력을 수정하지 않는다.
대용량 다운로드, 학습, GPU 작업, commit/push는 최초 구현 작업에 포함하지 않는다.

최초 구현 검증 기록(2026-09-20):

- 변경 전 전체 테스트: 26 passed.
- 변경 후 `python -m pytest -q`: **159 passed, 7.80초**. 기존 회귀 테스트 포함.
- Live input capture 추가 후 재검증: **168 passed, 6.83초**.
- `git diff --check`, 신규 연구 코드/CLI `compileall`, CLI `--help`: 통과.
- 공용 설정 `audit`: exit code 2, 경로 미지정과 runtime `not_run`을 정상 보고.
- 후속 M4 pilot: 5,600 readouts, 32,768 features, 308,780 active pairs, GPU scoring 676.60초,
  16-feature panel과 control의 72 rollouts를 완료. 정확한 기록은 위 Runbook 참조.
- 테스트 환경: macOS, Python 환경의 PyTorch 2.2.2; 이 시간은 테스트 실행 시간이지 실험 처리량이 아님.

## 파일별 역할

- `event_sae/research/output_head/config.py`: schema, 연구 범위·예산, 입력/output 경계.
- `readouts.py`, `splits.py`: 원본 forward/episode 확인, 사전 split, sparse cache.
- `head.py`, `sensitivity.py`: local tensor export, reference edit, full-vocab KL와 보조 지표.
- `candidates.py`, `results.py`: 원래 ranking 공식, 고정 panel, paired 성공률 drop/CI.
- `provenance.py`, `workflow.py`, `runtime.py`: artifact hash, 단계 연결, opt-in runtime.
- `scripts/openvla/output_head_sensitivity.py`: 단일 CLI.
- `scripts/openvla/headsetup.py`: 로컬 snapshot 검증, head metadata 생성·export.
- `scripts/openvla/headinputs.py`: 정책 가중치 없이 live LIBERO 입력 2개 캡처.
- `configs/research/openvla/output_head_sensitivity.yaml`: 경로가 비어 있는 공용 설정.

기존 eval 코드는 lazy import와 선택적 원본 `trial_indices`, 연구용 episode별 seed,
명시적 SDPA/local-only loading만 확장했다. 기본값에서는 기존 first-N trial 순서,
자동 attention backend 선택과 protocol serialization을 유지한다. 기존 full-sweep
제약은 완화하지 않았다. 새 연구에서는 SDPA·BF16·비양자화·all-row hook을 고정한다.

## 먼저 실행할 CPU 확인

저장소 root에서 기존 torch/pytest/PyYAML 환경으로 실행한다. 모델·LIBERO·TensorFlow는
필요 없다. Head export 테스트에는 safetensors가 필요하다. 테스트가 실제 모델을
다운로드하거나 simulator를 실행하지 않는다.

```bash
python -m pytest -q
python scripts/openvla/output_head_sensitivity.py --help
python scripts/openvla/output_head_sensitivity.py audit \
  --config configs/research/openvla/output_head_sensitivity.yaml
```

공용 설정은 경로가 `null`이므로 audit의 `status: blocked`, exit code 2가 정상이다.
`--help`, metadata audit/split/빈 plan은 torch 없이도 동작한다. 실제 event matrix를
읽는 plan은 CPU torch만 사용하며 모델·CUDA·LIBERO를 초기화하지 않는다.

## 로컬 설정과 입력

공용 YAML을 `configs/local/head.yaml`로 복사하고 경로를 채운다. 이 폴더는 Git에서
제외된다. 경로는 YAML 파일 위치가 아니라 **실행한 현재 디렉터리** 기준이므로
실데이터에는 절대경로를 권장한다. 입력과 output은 별도 디렉터리여야 한다.

| 설정 | 실제로 필요한 것 |
|---|---|
| `inputs.dense_dir` | `activation_index.jsonl`과 기존 dense `.pt` shards |
| `inputs.sae_checkpoint` | 기존 `ae.pt`와 인접 `config.json`; BatchTopKSAE·layer·capture target 일치 |
| `inputs.source_episode_manifest` | 원본 task/trial/초기 상태 hash와 수집 episode/run 대응표 |
| `inputs.generation_manifest` | batch/padding/KV-cache/action-dim의 검증 근거 JSON |
| `inputs.local_model_snapshot` | 이미 존재하는 pinned OpenVLA snapshot; runtime에는 processor/remote code도 로컬에 필요 |
| `inputs.head_export_metadata` | 확인한 norm 정의·dtype·model/code revision·processor identity JSON |
| `inputs.output_head_bundle` | 기존 파생 bundle이 있으면 경로; 없으면 `<output>/head` 사용 |
| `inputs.event_scores_path` | 선택한 비교 방법의 전체 feature matrix와 실제 scope provenance |
| `inputs.runtime_inputs` | 소수 실제 preprocessed `predict_action` 입력 tensor와 provenance `.pt` |
| `rollout.eval_manifest` | 평가할 원본 task/trial/hash 목록 및 split/overlap/label-use 이력 |
| `output.root_dir` | 기존 입력과 겹치지 않는 별도 연구 output |

단지 빈 경로를 채웠다는 이유로 검증 근거가 생기지는 않는다. 다음 예시의 placeholder는
실제 수집/runtime 정보를 확인해 바꿔야 한다.

Generation manifest 예시:

```json
{
  "source_run_id": "REPLACE_WITH_VERIFIED_RUN_ID",
  "dataset_id": "REPLACE_WITH_MERGED_DATASET_ID",
  "suite": "libero_spatial",
  "layer_idx": 31,
  "action_dim": 7,
  "batch_size": 1,
  "padding": "none",
  "use_cache": true,
  "index_scope": "complete",
  "evidence": "REPLACE_WITH_COLLECTION_CONFIG_AND_CODE_IDENTITIES"
}
```

Merged index의 `source_run_idx`는 dataset namespace와 결합한다. 원본 prompt 길이를
281로 가정하거나 마지막 7개 row를 잘라내지 않는다. 한 step의 forward마다 마지막
유효 row를 선택하며, action dimension 수·range·중복 검사를 통과해야 한다.

Episode manifest는 `{"suite":"libero_spatial", "initial_state_hash_provenance":"...",
"episodes":[...]}` 형식이다. 각 episode에는 `episode_num`, `task_id`,
`task_episode_idx`, `initial_state_sha256` 및 index와 같은 `source_run_id` 또는
`source_run_idx`가 필요하다. `task_episode_idx`는 run-local 순번이 아닌 원래 LIBERO
초기 상태 index다. Initial state hash는 기존 runner의 dtype/shape/bytes hashing 결과다.
SHA-256 형태를 갖춘 임의 문자열은 유효한 provenance가 아니다.

Head metadata 예시:

```json
{
  "model_revision": "962318cec55ac10993ff0f5f43eda9a270b4c873",
  "code_revision": "47a0ec7fc4ec123775a391911046cf33cf9ed83f",
  "target_layer": 31,
  "hidden_dtype": "bfloat16",
  "norm_dtype": "bfloat16",
  "head_dtype": "bfloat16",
  "processor_identity": "REPLACE_WITH_VERIFIED_PROCESSOR_IDENTITY",
  "norm_spec": {"implementation":"llama_rms_norm_v1", "eps": 0.000001}
}
```

`eps`와 dtype는 예시를 맹목적으로 쓰지 말고 실제 pinned config/loader와 대조한다.
기본 tensor key는 `language_model.model.norm.weight`와
`language_model.lm_head.weight`다. 다른 저장 형식은 `tensor_keys`를 명시하거나 Python
API의 runtime-object export를 사용한다. 없는 key를 비슷한 이름으로 추측하지 않는다.
지원하지 않는 norm/양자화/불명확한 tied-weight 포맷은 오류다.

Action decoding metadata는 확인된 경우만 exporter에 `action_decoding`으로 제공한다.
없어도 full-vocab KL은 가능하며 bin 변화 지표는 null과 사유를 기록한다. Padded vocab,
effective vocab, bin centers를 혼동하지 않는다.

## RunPod 실행 순서

```bash
python scripts/openvla/output_head_sensitivity.py audit --config configs/local/head.yaml

# Confirmatory 실험이면 반드시 pilot 전에 생성·보존한다.
python scripts/openvla/output_head_sensitivity.py split --config configs/local/head.yaml

python scripts/openvla/headsetup.py --config configs/local/head.yaml \
  --snapshot /workspace/head-inputs/openvla-spatial --export
python scripts/openvla/output_head_sensitivity.py prepare --config configs/local/head.yaml
python scripts/openvla/headinputs.py --config configs/local/head.yaml --allow-simulator
python scripts/openvla/output_head_sensitivity.py validate --config configs/local/head.yaml \
  --allow-model-execution
python scripts/openvla/output_head_sensitivity.py score --config configs/local/head.yaml
python scripts/openvla/output_head_sensitivity.py plan --config configs/local/head.yaml
```

`split`이 만든 manifest를 `sampling.split_manifest`에 연결하고, 그 evaluation subset으로
`rollout.eval_manifest`를 만든다. Pilot 시에는 discovery/validation만 사용한다.
`splits.frozen_before_pilot`은 기본 `null`이며, 당시 실제 동결 이력이 확인될 때만
명시한다. split 파일을 지금 만드는 것이 과거 pilot 이전 동결의 증거는 아니다.
평가 manifest 형식은 다음과 같다.

```json
{
  "mode": "pilot",
  "discovery_eval_overlap": true,
  "evaluation_labels_previously_used": null,
  "episodes": [
    {"suite":"libero_spatial", "task_id":0, "task_episode_idx":0,
     "initial_state_sha256":"REPLACE_WITH_ACTUAL_SHA256"}
  ]
}
```

Confirmatory에는 `mode: confirmatory`, `frozen_before_pilot: true`, 실제
`split_manifest_hash`, `discovery_eval_overlap: false`,
`evaluation_labels_previously_used: false`가 필요하며 frozen split의 실제 membership과
대조한다. 기존 label을 보고 방법을 정했으면 confirmatory로 전환할 수 없다.

`headinputs.py`는 LIBERO simulator와 pinned processor만 불러오고 7B policy 가중치는
불러오지 않는다. task 0/1의 fresh observation을 기존 `get_action()` 전처리에 통과시킨 뒤
`predict_action` 경계에서 tensor를 복사하고 중단하므로 policy action은 생성·실행하지
않는다. 기본 출력은 head metadata 옆의 `runtime.pt`이며 로컬 YAML의
`inputs.runtime_inputs`가 자동으로 갱신된다. 기존 파일은 덮어쓰지 않는다.

`validate`의 runtime `.pt`는 `synthetic: false`, `model_revision`, `code_revision`,
`processor_identity`, `episodes`, `calls`를 포함한다. `calls`는 processor가 이미 만든
`input_ids`·`attention_mask`·`pixel_values` 등의 tensor dict 목록이다. Batch size 1,
padding 없음이 필요하다. JPEG에서 재구성한 입력을 원본과 동일하다고 주장하지 않는다.
최대 모델 호출 수 8은 raw·alpha1·active alpha0·inactive의 **총 호출 수**다. 즉 기본값에서
최대 2개의 실제 입력을 검증한다. 이 input fixture 생성 자체는 기존 runtime에서 소수
입력을 확보하는 M4 준비 작업이며 새 전체 discovery 수집을 의미하지 않는다.

Head/edit parity는 baseline logits, 실제 all-row hook, 첫/cached forward와 실제 score
batch shape를 비교한다. 모델·head·SAE·mapping·dtype·backend·허용오차·관련 코드가 바뀌면
오래된 passed를 사용하지 않는다. 수치 허용오차는 실패를 감추려고 자동 확대하지 않는다.

실데이터 pilot의 scoring은 `isolated_row_bf16_reference_v1`과
`pair_batch_size: 1`을 고정한다. 이는 cached action forward와 같은 한-row BF16
norm/head 연산이다. prefill 전체-row hook과 isolated-row SAE decode의 차이는
`diagnostics.isolated_score_vs_runtime`에 nonblocking 진단으로 기록한다. 이 진단은 SAE
decoder grouping, output-head projection grouping, 두 차이를 합친 runtime-vs-score 오차를
각각 분리한다. 따라서 offline score를
prefill runtime logits와 수치적으로 동일하다고 주장하지 않고, **isolated-row score가 실제
all-row intervention의 효과를 예측하는지**를 후속 실험에서 검증한다. 서로 다른 BF16
연산 grouping이 같아야 한다고 가정하거나 그 차이를 전역 허용오차 확대로 숨기지 않는다.

`plan`은 top-K union을 자르지 않고 deduplicate 후 실제 rollout 수를 계산한다.
현재 native runner는 raw 1개 + unique features + identity 1개를 같은 전체 평가 case에서
수행한다. 예산 초과는 `blocked-<hash>.json`으로 남기고 실행하지 않는다. 이때 예산을
수정해 다시 plan할 수 있다. 통과한 최종 panel은 immutable하다.

검토·commit 후 승인된 **현재 새 저장소 revision**을 사용한다. 개발 과정에서 agent가
commit/push를 대신 실행하지 않는다.

```bash
python scripts/openvla/output_head_sensitivity.py run --config configs/local/head.yaml \
  --approved-plan-hash ACTUAL_PLAN_HASH \
  --expected-code-revision ACTUAL_CLEAN_COMMIT \
  --execute
python scripts/openvla/output_head_sensitivity.py analyze --config configs/local/head.yaml
```

단독 `--execute`나 YAML의 `execute` 값은 승인 hash를 대체하지 않는다. Dirty worktree,
변경된 plan/config/input, parity 미실행, 예산 초과, 초기 상태 hash 불일치는 거부한다.
Alpha1 action/outcome identity를 확인한 뒤 suppression을 실행한다. 중단된 condition은
자동 재시도해 예산을 넘기지 않는다. 완료 결과 resume도 hash·case·hook·identity를 재검사한다.

## 비교 scope와 분석 해석

Head score는 task → episode → step → dimension에 동일 가중치를 준다. 비활성 row의
0도 분모에 포함하며 feature별 sample을 바꾸지 않는다. Cached sparse encoding은
전체 nonzero를 보존한다. 기존 저장 Top-K의 누락을 dead feature로 보지 않는다.

`discovery_manifest_hash`는 같은 원본 episode 모집단의 hash이며, step sampling을
포함하는 `sample_manifest_hash`와 다르다. Event/window/task matrix의 `scope`는
`task_ids`, `dictionary_size`, `sae_sha256`, `discovery_manifest_hash` 및 confirmatory일 때
`split_manifest_hash`를 실제 계산 provenance에 맞게 제공해야 한다.

이미 전체 500 episodes로 집계한 matrix를 임의로 2-task/discovery 전용이라고 표시하면
안 된다. 현재 CLI는 scope가 검증되지 않거나 다른 matrix를 **비교 불가로 차단**한다.
Discovery-filtered event score 자동 재계산은 이 확장에 포함하지 않았다. 기존 scoring
파이프라인에서 맞는 범위의 artifact를 준비해야 한다. 선택하지 않은 window/task
matrix가 없다는 이유로 event/head 비교를 막지는 않는다. 원래 random-alive ranking과
새 audit 무작위 표본은 다른 표본이며, 현재 CLI의 random 비교는 후자다.

Audit은 같은 readout discovery에서 alive인 feature를 균일 비복원 표집한다. Top-K와
겹쳐도 다시 뽑지 않고 membership을 보존한다. 기본 6개는 탐색용이지 높은 검정력의
증거가 아니다. 기존 event 상위 feature가 readout에서 inactive여도 삭제하지 않는다.

Primary outcome은 task별 `SR_raw - SR_edit`의 평균이다. 음수(개선), 상수/undefined
상관, zero-width bootstrap, 누락 결과를 숨기지 않는다. 모든 feature가 같은 초기 상태
bootstrap draw를 공유한다. 결과는 고정 panel·고정 tasks에 조건부이며 SAE 학습까지
held-out이라는 주장이나 전체 dictionary의 global recall을 하지 않는다.

## 산출물과 제한

`<output>/` 아래에 split/sample manifest, head, readouts, validation, scores, plan,
runs, analysis를 저장한다. Scores에는 suite/task/dimension CSV, 분석에는 paired effects,
predictor 비교 CSV와 uncertainty JSON 및 report.md가 생긴다. Per-row edited logits나
전체 `[rows, dictionary]` latent matrix를 보관하지 않는다.

- 현재 최적화는 exact active-pair batching이다. Algebraic fast edit와 vocab-block projection은 구현하지 않았고 암묵적으로 사용하지 않는다.
- Working-memory 상한은 중간 tensor 추정량이다. 모델/SAE 가중치·allocator overhead를 포함한 총 GPU RAM 보장은 아니며 실제 peak 값은 score 때 별도 기록한다.
- 원본 대형 dense shard identity는 size/mtime와 index hash를 사용하며 그 검증 수준을 명시한다. 새 cache/head는 내용 SHA256을 검사한다. 전체 수백 GB source hash를 재계산했다고 주장하지 않는다.
- Native runner는 task별 동일 case 수 및 전체 case identity 검사를 지원한다. 연구용 episode seed는 seed/task/original-trial hash에서 유도하며 기존 run의 RNG 방식과 구별한다.
- M4 pilot에서 dependency/runtime/model compatibility, 처리량, CUDA peak allocation, 성공률을
  실측했다. 다만 이는 overlapping pilot이며 M5 held-out confirmatory 결과를 대신하지 않는다.
