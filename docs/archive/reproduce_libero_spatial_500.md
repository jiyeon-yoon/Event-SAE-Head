# LIBERO-Spatial 500 재현

> 보관 문서 — 당시 Baseline 안내다. 현재 Head 연구는 [연구·결과](../experiment.md)와 [실행·복구](../runbook.md)를 따른다.

이 문서는 **기존 500 rollouts와 학습된 Layer 31 SAE를 내려받아 Baseline을
검증하는 경로**다. 처음 실행하는 팀원은 이 경로부터 사용한다. Activation 수집과
SAE 학습부터 다시 수행하려면 [OpenVLA 가이드](openvla.md)의 Phase 1을 따른다.

이미 수집·학습한 아래 산출물을 재사용한다.

- 10 tasks × 50 rollouts = 500 episodes
- Layer 31 dense activation 369 shards
- Layer 31 BatchTopK SAE (`4096 → 32768`, `k=64`, 4,000 steps)

새 activation 수집과 SAE 재학습은 하지 않는다.

## 실행 범위

```text
5개 dataset 검증·병합
→ offline FVE/MSE/alive/L0 + sparse Top-K (dense 데이터 1회 순회)
→ AWE keyframes
→ 5-frame event bundles + SigLIP descriptors
→ task-local clustering
→ event/window/task/random feature ranking
```

Gemini 라벨은 설명용이므로 ranking 계산에 필요하지 않다. Hooked SR과
intervention은 새로운 closed-loop rollout을 생성하므로 아래 discovery 결과를 먼저
검증한 뒤 별도로 실행한다.

## RunPod

- Container image:
  `ghcr.io/jiyeon-yoon/event-sae-runtime@sha256:ef895731ffd73985e1183c964b479bdcc7e97dfeabea6c8777b903c54ade1405`
- RTX 4090 한 장으로 실행 가능
- Container disk: 400 GB
- Volume disk: 0 GB 가능. 단 Pod stop·restart·terminate 전에 결과를 외부에
  업로드한다. `/workspace` 보존을 전제로 하지 않는다.
- GHCR image는 의존성 환경을 제공한다. Baseline 코드는 아래에서 별도로 clone하고
  검증된 commit으로 고정한다.

Pod가 시작되면 Web terminal을 열고 환경만 확인한다.

```bash
event-sae-init
event-sae-verify --require-gpu
```

장시간 작업은 tmux 안에서 실행한다.

```bash
tmux new -s event-sae-spatial
```

이제 열린 tmux 화면 안에 아래 블록을 붙여 넣는다.

```bash
set -euo pipefail

export EVENT_SAE_COMMIT=fd3bc485668b8fb32b3948a9b642859f41b16277

cd /workspace
test -d Event-SAE-Baseline/.git || \
  git clone https://github.com/jiyeon-yoon/Event-SAE-Baseline.git Event-SAE-Baseline
test -z "$(git -C Event-SAE-Baseline status --porcelain)" || \
  { echo "DIRTY_SOURCE"; exit 1; }
git -C Event-SAE-Baseline fetch origin main
git -C Event-SAE-Baseline checkout --detach "$EVENT_SAE_COMMIT"
test "$(git -C Event-SAE-Baseline rev-parse HEAD)" = "$EVENT_SAE_COMMIT" || exit 1

cd /workspace/Event-SAE-Baseline
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
git rev-parse HEAD
python -m pytest -q
```

테스트가 실패하면 GPU 작업을 시작하지 않는다. tmux 재접속 명령은
`tmux attach -t event-sae-spatial`이다.

## 1. 입력 다운로드·검증

입력 저장소는 공개지만 281 GB를 비로그인 상태로 받으면 rate limit이 발생할 수
있다. **Read token으로 로그인**한 뒤 다운로드한다.

```bash
hf auth whoami || hf auth login

cd /workspace/Event-SAE-Baseline
python scripts/openvla/download_libero_spatial_reproduction_inputs.py \
  --output-root /workspace/event-sae-spatial-inputs
```

정상이면 마지막에 다음 수량과 `INPUTS_OK`가 출력된다.

```text
episodes: 500
videos: 500
activation_shards: 369
```

## 2. Discovery pipeline 실행

```bash
set -euo pipefail
export EVENT_SAE_COMMIT=fd3bc485668b8fb32b3948a9b642859f41b16277
cd /workspace/Event-SAE-Baseline
python scripts/openvla/reproduce_libero_spatial_500.py \
  --input-root /workspace/event-sae-spatial-inputs \
  --work-dir /workspace/event-sae-spatial-repro \
  --expected-code-revision "$EVENT_SAE_COMMIT" \
  --device cuda:0 \
  --batch-size 1024
```

다운로드와 Top-K 변환은 재실행할 수 있다. Top-K는 이미 끝난 shard를 GPU로 다시
계산하지 않지만 369개 결과를 다시 읽어 검증한다. 다른 단계의 최종 파일이
불완전하면 중단하므로 오류에 표시된 해당 파일만 확인 후 삭제해 재실행한다.
병합 결과는 원본 pair 디렉터리를 가리키는 symlink이므로 pipeline 완료 전 입력
디렉터리를 삭제하면 안 된다. Offline fidelity는 학습에 사용한 activation을 다시
평가한 **in-sample 진단값**이다.

완료 기준:

```text
DISCOVERY_PIPELINE_OK: /workspace/event-sae-spatial-repro/pipeline_summary.json
```

핵심 결과:

| 결과 | 경로 |
|---|---|
| 전체 단계 검증 | `/workspace/event-sae-spatial-repro/pipeline_summary.json` |
| offline fidelity | `/workspace/event-sae-spatial-repro/pipeline/offline_fidelity.json` |
| AWE 결과 | `/workspace/event-sae-spatial-repro/pipeline/keyframes/waypoint_summary.json` |
| cluster 통계 | `/workspace/event-sae-spatial-repro/pipeline/clusters/summary.json` |
| feature score | `/workspace/event-sae-spatial-repro/pipeline/scores/event_feature_scores.pt` |
| intervention 후보 | `/workspace/event-sae-spatial-repro/pipeline/rankings/candidates.jsonl` |

논문 Table 2의 Spatial 값인 `4.15 keyframes/rollout`, `48 clusters`,
`36 recurring clusters`는 비교 기준이다. 새로 수집한 rollout이므로 완전히 동일한
수치를 강제하지 않는다.

## 3. Discovery 결과 업로드

Volume disk가 0 GB이면 아래 원격 검증 전 Pod를 stop·restart·terminate하지 않는다.
팀원은 자신의 Hugging Face namespace에 결과 저장소를 만든다.

```bash
set -euo pipefail
hf auth whoami || hf auth login
HF_USER=$(python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])")
export HF_RESULTS_REPO="$HF_USER/event-sae-libero-spatial-reproduction"

python scripts/openvla/upload_reproduction_results.py \
  --work-dir /workspace/event-sae-spatial-repro \
  --repo-id "$HF_RESULTS_REPO" \
  --section discovery
```

마지막 출력이 `REMOTE_DISCOVERY_OK`인지 확인한다. 병합된 281 GB 원본은 이미
5개 HF dataset에 있으므로 중복 업로드하지 않고, 새로 만든 Top-K·fidelity·AWE·
cluster·ranking 결과만 올린다.

## 4. Hooked SR

정확한 비교를 위해 같은 고정 OpenVLA revision과 seed로 Raw와 SAE reconstruction을
각각 10 rollouts/task, 총 100개씩 새로 실행한다. 기존 500개에서 처음 10개를
재사용하는 방법은 빠른 탐색용일 뿐 정확한 비교로 취급하지 않는다.

```bash
set -euo pipefail
export EVENT_SAE_COMMIT=fd3bc485668b8fb32b3948a9b642859f41b16277
cd /workspace/Event-SAE-Baseline
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p /workspace/event-sae-spatial-repro/hooked-sr/runs

python scripts/openvla/evaluate_policy.py \
  --config configs/reproduction/openvla/libero_spatial_hooked_sr_layer31.yaml \
  --mode raw \
  --expected-rollouts 100 \
  --expected-code-revision "$EVENT_SAE_COMMIT" \
  --result-output /workspace/event-sae-spatial-repro/hooked-sr/raw.json \
  --override logging.root_dir=/workspace/event-sae-spatial-repro/hooked-sr/runs/raw

python scripts/openvla/evaluate_policy.py \
  --config configs/reproduction/openvla/libero_spatial_hooked_sr_layer31.yaml \
  --mode reconstruction \
  --sae-checkpoint /workspace/event-sae-spatial-inputs/checkpoint/trainer_0/ae.pt \
  --layer-idx 31 \
  --expected-rollouts 100 \
  --expected-code-revision "$EVENT_SAE_COMMIT" \
  --result-output /workspace/event-sae-spatial-repro/hooked-sr/reconstruction.json \
  --override logging.root_dir=/workspace/event-sae-spatial-repro/hooked-sr/runs/reconstruction

python scripts/openvla/compare_hooked_sr.py \
  --raw-result /workspace/event-sae-spatial-repro/hooked-sr/raw.json \
  --reconstruction-result /workspace/event-sae-spatial-repro/hooked-sr/reconstruction.json \
  --expected-code-revision "$EVENT_SAE_COMMIT" \
  --output-path /workspace/event-sae-spatial-repro/hooked-sr/comparison.json

test -n "${HF_RESULTS_REPO:-}" || {
  HF_USER=$(python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])")
  export HF_RESULTS_REPO="$HF_USER/event-sae-libero-spatial-reproduction"
}
python scripts/openvla/upload_reproduction_results.py \
  --work-dir /workspace/event-sae-spatial-repro \
  --repo-id "$HF_RESULTS_REPO" \
  --section hooked-sr
```

`REMOTE_HOOKED_SR_OK`가 완료 기준이다.

## 5. Intervention 개발 검증

논문 규모의 전체 sweep 전에 구현만 검증하려면 아래 축소 실험을 실행한다. 기존
전체 Intervention 코드와 결과 경로는 사용하거나 삭제하지 않는다.

- 조건: Raw, event rank-1 `alpha=1`, 같은 event feature `alpha=0`, random rank-1 `alpha=0`
- 조건별: 10 tasks × 5 trials = 50 rollouts
- 전체: 200 rollouts
- 동일 task/trial의 LIBERO initial-state 배열 SHA256을 네 조건에서 직접 비교
- Raw와 `alpha=1`의 action·성공 결과가 같아야 함
- `alpha=0`은 target feature의 개입 후 nonzero 수가 0이어야 함
- Raw 대비 action 변화와 조건별·task별 SR은 결과 JSON에 기록

이 실험은 구현 검증이며 논문 Table 3의 통계적 재현 결과로 사용하지 않는다. 현재
실측 속도 기준 약 2~4시간을 예상한다.

```bash
set -euo pipefail
export EVENT_SAE_COMMIT=fd3bc485668b8fb32b3948a9b642859f41b16277
cd /workspace/Event-SAE-Baseline
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python scripts/openvla/run_intervention_development_validation.py \
  --config configs/reproduction/openvla/libero_spatial_intervention_development_validation_layer31.yaml \
  --candidates /workspace/event-sae-spatial-repro/pipeline/rankings/candidates.jsonl \
  --sae-checkpoint /workspace/event-sae-spatial-inputs/checkpoint/trainer_0/ae.pt \
  --work-dir /workspace/event-sae-spatial-repro \
  --expected-code-revision "$EVENT_SAE_COMMIT"
```

완료 기준:

```text
INTERVENTION_DEVELOPMENT_VALIDATION_OK
```

핵심 결과는 아래 파일 한 개에서 확인한다.

```text
/workspace/event-sae-spatial-repro/intervention-development-validation/development_validation_summary.json
```

`passed=true`, `identical_initial_states=true`,
`alpha1_identity_actions_and_outcomes=true`, 두 `feature_zeroed=true`를 확인한다.
`action_change_observed`는 실제 행동 변화 관측값이며 구현 통과 조건과 분리된다.

```bash
test -n "${HF_RESULTS_REPO:-}" || {
  HF_USER=$(python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])")
  export HF_RESULTS_REPO="$HF_USER/event-sae-libero-spatial-reproduction"
}
python scripts/openvla/upload_reproduction_results.py \
  --work-dir /workspace/event-sae-spatial-repro \
  --repo-id "$HF_RESULTS_REPO" \
  --section intervention-development-validation
```

업로드 완료 기준은 `REMOTE_INTERVENTION_DEVELOPMENT_VALIDATION_OK`다.

## 6. 논문 규모의 전체 Intervention

이 단계는 Baseline 코드 실행에 필수인 smoke test가 아니라, 논문 규모의 통계적
intervention 결과가 필요할 때만 실행한다.

`candidates.jsonl`은 ranking 4종 × 5개로 20행이지만 feature ID가 서로 겹칠 수
있다. 동일 feature는 한 번만 실행하고 결과를 ranking 간 공유해야 한다.

논문의 메인 intervention은 후보 하나당 50 rollouts/task, 총 500 rollout이다.
아래 명령은 동일 feature ID를 중복 제거하고 raw baseline과 전체 feature
intervention을 순서대로 실행한다. 완료된 결과는 재실행하지 않으므로 같은 명령으로
이어서 실행할 수 있다.

```bash
set -euo pipefail
export EVENT_SAE_COMMIT=fd3bc485668b8fb32b3948a9b642859f41b16277
cd /workspace/Event-SAE-Baseline

python scripts/openvla/run_intervention_sweep.py \
  --config configs/reproduction/openvla/libero_spatial_intervention_layer31.yaml \
  --candidates /workspace/event-sae-spatial-repro/pipeline/rankings/candidates.jsonl \
  --sae-checkpoint /workspace/event-sae-spatial-inputs/checkpoint/trainer_0/ae.pt \
  --work-dir /workspace/event-sae-spatial-repro \
  --expected-code-revision "$EVENT_SAE_COMMIT"

test -n "${HF_RESULTS_REPO:-}" || {
  HF_USER=$(python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])")
  export HF_RESULTS_REPO="$HF_USER/event-sae-libero-spatial-reproduction"
}
python scripts/openvla/upload_reproduction_results.py \
  --work-dir /workspace/event-sae-spatial-repro \
  --repo-id "$HF_RESULTS_REPO" \
  --section intervention
```

완료 기준은 `INTERVENTION_SWEEP_OK`와 `REMOTE_INTERVENTION_OK`다.
