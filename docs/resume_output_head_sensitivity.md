# 출력층 민감도 실험 재시작 — 2026-09-21

이전 RunPod은 종료됐다. 기존 2-task/4-case/72-rollout 결과는 보존하고,
새 평가를 시작하기 전에 백업을 복구하고 discovery/evaluation 분리 경로를 준비한다.
현재 로컬에서는 GPU나 Hugging Face 다운로드를 실행하지 않았다.

후속 요청으로 실험 목표를 전체 10개 task로 확대했다.
아래 2-task 후속 제안 대신 [전체 실험 설계](output_head_full10_study.md)와
[현재 전체 실행 안내](run_output_head_full10.md)를 따른다.
이 문서 4절의 미구현 목록은 이전 시점 기록이다. 현재 followup 경로에 구현됐으며
과거 초기상태를 복원했다는 의미는 아니다.

## 1. 이번에 개발한 것

- Plan의 feature ID를 JSON 저장 전에 문자열로 고정했다. `plan` 저장 후
  `Approved plan hash is empty, stale, or tampered`가 나던 원인을 수정했다.
  기존 fingerprint 함수와 과거 결과의 bytes는 유지한다.
- `headrestore.py`를 추가했다. 내려받은 **압축파일 전체**의 SHA-256과 필수
  artifact 목록을 확인하고 새 폴더에 복구한다. 모델/torch/LIBERO는 필요 없다.
- `splits.frozen_before_pilot`의 기본값을 `null`로 바꿨다. 새 manifest를
  생성했다는 이유로 과거 pilot 이전에 split을 고정했다고 기록하지 않는다.

`headrestore.py`의 `verified`는 백업 무결성과 파일 목록 확인이다.
실모델 parity, 72회 완료 여부, 연구 결론을 재검증했다는 뜻은 아니다.
원래 경로 문자열/해시를 그대로 보존하므로 복구본을 다른 경로에 놓았다고
과거 config가 바로 다시 실행되지는 않는다.

## 2. 새 Pod을 켜기 전

아래 복구는 GPU 없이도 가능하다. Mac에 `hf`가 있으면 로컬에서 먼저 수행해도 된다.
토큰은 `hf auth login`의 숨겨진 입력창에만 입력한다.

새 Pod은 Head 저장소의 최신 배포 commit을 사용한다. Baseline/Pipeline 저장소는
수정하지 않는다. 배포 상태와 현재 명령은 전체 실행 안내를 따른다.

Mac에서 현재 Head 저장소 안에서:

```bash
hf auth login
BACKUP="$PWD/artifacts/head-pilot-backup"
HF_HUB_OFFLINE=0 hf download jiyeony/event-sae-head-pilot-results --repo-type dataset --local-dir "$BACKUP"
ARCHIVE="$BACKUP/matched-task01-v1-results.tar.gz"
python3 scripts/openvla/headrestore.py --archive "$ARCHIVE"
```

기본 checksum은 이전 실험에서 따로 기록한 값이다:

```text
057e99bb74673c0ee28e599192875af4b338cfd274d8267d035988057067f1bf
```

정상 출력: `status: verified`, `recorded_plan.total_rollouts: 72`,
`recorded_plan.num_eval_cases: 4`.

복구할 때만 다음 명령을 추가한다. 목적지는 아직 없는 폴더여야 한다.

```bash
python3 scripts/openvla/headrestore.py --archive "$ARCHIVE" --destination artifacts/restored-pilot
```

리포트: `artifacts/restored-pilot/matched-task01-v1/analysis/report.md`.
이미 복구했으면 `--destination` 명령을 다시 실행하지 않는다.

## 3. 새 RunPod에서 복구할 경우

기존 pilot에 쓴 runtime image와 고정 revision은
[기존 실행 기록](reproduce_output_head_sensitivity_pilot.md#3-실행-규모와-고정-식별자)에 있다.
이 문서는 현재 RunPod 상품/가격이나 image registry 가용성을 재확인한 문서는 아니다.

코드가 GitHub에 반영된 다음, 새 Pod terminal에서 각 줄을 순서대로 실행한다.

```bash
event-sae-init
event-sae-verify --require-gpu
cd /workspace
git clone https://github.com/jiyeon-yoon/Event-SAE-Head.git
cd Event-SAE-Head
git rev-parse --short HEAD
python scripts/openvla/headrestore.py --help
hf auth login
BACKUP=/workspace/head-backup
HF_HUB_OFFLINE=0 hf download jiyeony/event-sae-head-pilot-results --repo-type dataset --local-dir "$BACKUP"
ARCHIVE="$BACKUP/matched-task01-v1-results.tar.gz"
python scripts/openvla/headrestore.py --archive "$ARCHIVE"
python scripts/openvla/headrestore.py --archive "$ARCHIVE" --destination /workspace/restored-head-pilot
```

Private 저장소에서 함께 받은 `config/head-matched.yaml`과
`manifests/eval_manifest_matched.json`은 이전 설정의 기록이다.
15 GB 모델, SAE, dense activation 등이 새 Pod에 복구된 것은 아니다.
필요한 새 실험 입력을 확정한 뒤 기존 실행 기록의 pinned download를 사용한다.

기존 결과 폴더에 새 `plan`, `validate`, `score`를 덮어쓰지 않는다.
소스 변경으로 implementation fingerprint가 달라졌으므로 후속 실험은
새 output root에서 새 parity와 plan을 생성해야 한다.

## 4. 다음 실험 제안과 아직 필요한 개발

목적: task 0·1의 작은 탐색 결과를 본 뒤, 별도의 task 2·3에서
Head-KL와 Event score가 고른 feature의 성공률 하락을 비교한다.
이 실험은 task 0·1의 순위를 다른 task에 그대로 적용하는 transfer 실험과 다르다.

| 항목 | 후속 실험 제안 |
|---|---|
| task | 2, 3 |
| task당 50개 기존 episode | discovery 35 / validation 5 / evaluation 10 |
| 점수 계산 | 두 방법 모두 동일 discovery 35개로 재계산 |
| 후보 | Head-KL top-3 ∪ Event top-3, 최대 6개 |
| 실제 평가 | task당 10개, 총 20개의 고정 초기상태 |
| control | raw + alpha=1 identity |
| 비용 상한 | (최대 6 features + 2 controls) × 20 = 최대 160 rollouts |
| 주 비교 | 두 방법 top-3의 평균 paired 성공률 하락 차이와 불확실성 |

160회는 실행 상한 제안이지 충분한 통계 검정력을 보장하는 숫자가 아니다.
실제 후보 중복을 제거한 `plan`에서 횟수를 확정한다. 평가 결과를 보고 feature,
split, 방법 설정을 고치면 이후 결과도 exploratory로 구분한다.
이번 저비용 패널만으로 전체 dictionary의 순위 상관 우월성을 주장하지 않는다.

**아직 바로 실행할 수 있는 M5 config는 아니다.** 다음 두 준비가 필요하다.

1. **Discovery-only Event score 재계산.** 현재 `subset_event_scores.py`는
   50개 episode로 합산된 matrix에서 task 행만 고른다. 이를 35개 episode의
   점수로 바꿀 수 없다. 원본 event/cluster membership/TopK artifact를 확인하고
   discovery episode만 포함해 기존 `score_matrix.py` 공식을 재계산하는 경로가 필요하다.
   재계산 시 선택 episode의 활성값, event window, task mean의 범위도 함께 제한한다.
2. **수집 episode와 초기상태의 대응 근거.** 원본 index와 prompt/episode records에서
   `(source_run_id, episode_num)` → `(task_id, task_episode_idx, state hash)`를 검증해야 한다.
   현재 LIBERO에서 hash를 계산한 사실만으로 과거 수집 당시 같은 상태였다고 단정하지 않는다.

기존 2-task pilot은 discovery와 평가가 겹쳤다. 이제 파일을 만들고
`frozen_before_pilot: true`를 넣어 그 이력을 바꿀 수는 없다. 별도 follow-up protocol의
동결 시점과 미사용 평가 label의 근거를 먼저 정의해야 한다. 이력이 불명확하면
`null`을 유지한다. 기존 SAE를 재사용하므로 향후 disjoint split도
"SAE 학습에 사용하지 않은 데이터"라고 주장하지 않는다.

## 5. 개발 검증

```bash
python -m pytest -q
git diff --check
python scripts/openvla/headrestore.py --help
```

CPU tests는 복구 bytes 보존/잘못된 archive 차단, plan 저장·재로드·승인 검증,
split 이력 미확인 차단을 검사한다. 실제 private backup 다운로드 및 새 GPU
실험 검증은 별도로 수행해야 한다.

2026-09-21 로컬 macOS/PyTorch 2.2.2 검증: 전체 **207 passed (6.81초)**,
`git diff --check` 및 복구 CLI `--help` 통과.
