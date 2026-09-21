# Event-SAE 출력층 민감도: 전체 10-task 실행 안내

작성: 2026-09-21. 대상 저장소는 **jiyeon-yoon/Event-SAE-Head**뿐이다.
Baseline 및 다른 팀원 저장소는 변경하지 않는다.

## 1. 이번에 무엇을 완성했나

남았던 것은 새로운 점수 공식을 만드는 일이 아니라 **데이터 연결 검증**이었다.

1. 원본 episode 번호를 task·trial에 연결하고, 현재 LIBERO 초기상태 hash를 기록한다.
2. task당 50 episode를 선정용 25 / 구현 검증용 5 / 행동 평가용 20으로 동결한다.
3. **동일한 선정용 episode만** 써서 Head-KL와 Event score를 계산했는지 검사한다.
4. 평가용 episode가 선정용에 섞이거나, 다른 SAE·index·split의 점수가 들어오면 중단한다.
5. 고정된 후보로 실제 rollout한 뒤 전체 및 task 0·1 / task 2–9를 나눠 분석한다.

새 도구:

| 도구 | 역할 |
|---|---|
| `headfull.py download` | pinned 원본 데이터·SAE·모델·Event 입력 다운로드 |
| `headfull.py merge` | 5개 source를 일관된 global episode 번호로 병합; dense는 symlink |
| `headfull.py setup` | 원본 연결 검증, 현재 상태 registry, split/eval manifest, local YAML 생성 |
| `headevents.py` | discovery-only Event score 재계산 및 Head/split 연결 검증 |
| `headgroups.py` | 원래 후보를 바꾸지 않고 subgroup 분석 |
| `headrestore.py` | 기존 pilot archive 실제 bytes checksum 확인 및 안전한 복구 |

### 해결한 것과 해결할 수 없는 것

**원수집 당시 초기상태 hash는 공개 데이터에 없다.** 현재 LIBERO 상태를 확인하는
코드를 만들었다고 그 과거 기록이 생기지는 않는다.

- `initial_state_sha256` / `current_runtime_state_sha256`: 현재 상태, 새 rollout에서 재확인.
- `historical_initial_state_sha256`: 원본 증거가 없으면 `null`.
- `historical_state_identity_verified`: 이번 공개 입력에서는 `false`.
- 과거 batch/padding/cache 설정도 구조에 근거한 해석이며 `historical_runtime_flags_verified: false`.
- `followup` 모드에서만 이 한계를 명시하고 진행한다. strict confirmatory 검증을 우회하지 않는다.
- split은 **이번 실행 전 현재 시각**으로 동결한다. 과거 pilot 이전이라고 소급하지 않는다.

따라서 "trial-ID 및 현재 runtime 상태가 분리된 후속 실험"으로 보고한다.
과거 상태까지 확인된 확증 실험, SAE/cluster 학습까지 완전한 held-out 실험이라고 쓰면 안 된다.
과거 정보의 부재는 남아 있는 구현 누락이 아니라 연구 범위의 한계다.

## 2. 실험 규모와 기대 결과

연구 질문: **feature를 끌 때 출력 확률분포가 많이 바뀌는 feature가 실제 성공률도
많이 떨어뜨리는가? Event score보다 잘 예측하는가?**

- OpenVLA / layer 31 / 기존 SAE 고정. 모델·SAE 재학습 없음.
- 10 task × discovery 25 = 250 episode, 최대 14,000 readout.
- Event / Head-KL / 평균 활성 / 활성 빈도, 각각 top-3.
- 선정용 데이터에서 활성화된 feature 중 무작위 6개 추가.
- 합집합 최대 18 feature + raw / identity control = 최대 20조건.
- 같은 200 평가 case를 모든 조건에 적용 → **최대 4,000 rollout**.
- 실제 중복 제거 후 횟수는 `plan`에서 확인한다.

주 비교는 Head-KL top-3 평균 성공률 하락과 Event top-3 평균 하락의 차이 및 CI다.
Spearman은 측정한 후보 panel 안의 보조 지표다. 전체 32,768 feature의 행동 효과를
모두 측정한 것은 아니다. Head-KL가 더 좋지 않게 나와도 정상적인 연구 결과다.
이 규모만으로 통계적 유의성·논문 게재가 보장되지는 않는다.

기존 task 0·1 pilot은 이미 보았으므로 탐색 subgroup으로 따로 표시한다.
task 2–9도 이력 미확인 label은 미확인으로 유지하고 확증 결과라고 자동 표시하지 않는다.
자세한 사전 비교·범위는 [전체 설계](output_head_full10_study.md)에 있다.

## 3. 새 Pod 준비

기존 pilot에서 사용한 이미지 기록:

```text
ghcr.io/jiyeon-yoon/event-sae-runtime@sha256:ef895731ffd73985e1183c964b479bdcc7e97dfeabea6c8777b903c54ade1405
```

RTX 4090 한 대 기준 경로다. 원본 5개 데이터만 약 281 GiB이고 모델·SAE·Event TopK·
결과 공간도 필요하다. 기존 **400 GB 디스크 설정**을 참고하되 실제 free space를 확인한다.
현재 상품 가용성/요금 견적은 이 문서에서 검증하지 않았다.

새 Pod 터미널에서 **명령 하나가 성공한 뒤 다음 명령**을 실행한다.

```bash
event-sae-init
event-sae-verify --require-gpu
cd /workspace
git clone https://github.com/jiyeon-yoon/Event-SAE-Head.git
cd /workspace/Event-SAE-Head
git status --short
git rev-parse HEAD
```

이미 clone했다면 clone 대신:

```bash
cd /workspace/Event-SAE-Head
git switch main
git pull --ff-only origin main
```

새 터미널/재접속 때마다 아래 변수를 다시 설정한다.

```bash
cd /workspace/Event-SAE-Head
export HF_HOME=/workspace/cache/head-full10-hf
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
CFG=configs/local/head-full10.yaml
OUT=/workspace/event-sae-head-results/full10-followup-v1
META=/workspace/head-inputs/full10
MODEL=/workspace/head-inputs/openvla-spatial
EVENTS=/workspace/head-inputs/event-public
CLI=scripts/openvla/output_head_sensitivity.py
```

전용 HF cache는 nested processor 코드의 unrevisioned lookup도 검증된 code revision으로
연결하기 위한 것이다. 다른 cached `main`을 몰래 덮어쓰지 않는다.

## 4. 입력 다운로드 → 병합 → split 생성

GPU rollout 없이 실행할 계획만 보기:

```bash
python scripts/openvla/headfull.py download
```

실제 다운로드(대용량, 네트워크/디스크 사용):

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python scripts/openvla/headfull.py download --execute
python scripts/openvla/headfull.py merge
python scripts/openvla/headfull.py setup --allow-libero
```

정상: `downloaded` → `merged_and_verified` → `configured`.
setup 출력은 discovery 250 / validation 50 / evaluation 200, `mode: followup`이다.
현재 상태 파일을 읽지만 OpenVLA 추론이나 평가 rollout은 하지 않는다.

출력 구조:

```text
/workspace/head-inputs/full10/
  source_episodes.json       # 원본 task/trial ↔ 현재 상태, 과거 상태는 별도
  generation_manifest.json  # 구조 검사 및 과거 설정 해석의 한계
  split.json                # 선정/검증/평가 분리와 최초 동결 시각
  eval_manifest.json        # 평가 200 case
  head.json                 # 다음 단계에서 생성
  runtime.pt                # 다음 단계에서 생성
  discovery_event_scores.pt # Event 재계산 후 생성
/workspace/Event-SAE-Head/configs/local/head-full10.yaml
/workspace/event-sae-head-results/full10-followup-v1/
```

동일 입력 재실행은 최초 split 시각을 유지한다. 입력·state·split 조건이 달라졌으면 중단한다.
이전 2-task 결과 폴더나 기존 50-episode aggregate Event score를 여기에 붙이지 않는다.

## 5. 출력층 추출 → 선정 데이터 준비 → 현재 실행 검증

```bash
python scripts/openvla/headsetup.py --config "$CFG" --snapshot "$MODEL" --metadata-output "$META/head.json" --export
python "$CLI" prepare --config "$CFG"
python scripts/openvla/headinputs.py --config "$CFG" --output "$META/runtime.pt" --allow-simulator
python "$CLI" validate --config "$CFG" --allow-model-execution
```

- export: 로컬 모델에서 norm/head 추출, 모델 전체 추론 아님.
- prepare: discovery readout 캐시 생성, 최대 14,000 rows.
- headinputs: discovery/validation 상태에서 입력 관찰만 생성; 평가 결과를 보지 않는다.
- validate: GPU 모델 추론으로 head/edit parity 검증. **`status: passed`가 필요하다.**
- 기존 `runtime.pt`가 있으면 headinputs를 무작정 재실행하지 않는다. 이미 완료한 단계는 건너뛴다.
- `readout_only_exact: false`는 원본 forward 묶음을 보존해야 한다는 의미일 수 있다.
  단독으로 실패 판정하지 않되 실제 validate 결과는 반드시 통과해야 한다.

TensorRT/Gym 등의 경고 문구보다 최종 JSON status와 traceback을 확인한다.
실패하면 tolerance를 임의로 늘리거나 manifest의 검증 여부를 바꾸지 않는다.

## 6. 두 방법을 같은 discovery로 계산

```bash
python "$CLI" score --config "$CFG"
python scripts/openvla/headevents.py --config "$CFG" --event-root "$EVENTS"
```

Head는 전체 32,768 feature를 점수화하되 비활성 pair는 0으로 포함한다.
Event는 같은 250 episode의 event/window/task mean/coverage를 다시 계산한다.
단순히 기존 점수 파일에 새 split hash를 붙이는 작업이 아니다.

`headevents`는 성공한 뒤 `inputs.event_scores_path`를 연결한다.
기존 cluster membership은 고정 재사용하며 discovery만으로 cluster를 다시 학습한 것은 아니다.
Head와 Event의 episode 범위는 같지만 Head는 episode당 최대 8개 step, Event는 원래 event/window
정의에 필요한 discovery step을 사용한다. 완전히 같은 token 집합을 평균했다는 뜻은 아니다.

## 7. 계획 확인 후에만 실제 rollout 실행

```bash
python "$CLI" plan --config "$CFG"
PLAN="$OUT/plan/rollout_plan.json"
python -c 'import json,sys;p=json.load(open(sys.argv[1]));print("rollouts",p["total_rollouts"],"features",len(p["feature_ids"]),"mode",p["mode"])' "$PLAN"
```

`status: planned`, `selection_eval_overlap: false`, `total_rollouts <= 4000`이어야 한다.
계획의 feature/조건/case와 예산을 확인한다. 아래 run부터 장시간 GPU 비용이 발생한다.
score 소요 시간으로 4,000회 rollout 시간을 추정할 수 없다. Pod의 실제 시간당 요금과
초기 실행 속도로 예산을 판단하며, 필요하면 결과를 보지 않고 중단·백업한다.

해시는 직접 입력하지 않고 **현재 파일**에서 읽는다:

```bash
PLAN_HASH=$(python -c 'import json,sys;from event_sae.research.output_head.provenance import fingerprint;p=json.load(open(sys.argv[1]));h=p.pop("plan_hash");assert h==fingerprint(p);print(h)' "$PLAN")
CODE_REV=$(git rev-parse HEAD)
git status --short
LOG="$OUT/full10-run.log"
```

`git status --short`가 비어 있고 hash 검사가 성공했을 때 실행한다.

```bash
nohup env PYTHONUNBUFFERED=1 python "$CLI" run --config "$CFG" --execute --approved-plan-hash "$PLAN_HASH" --expected-code-revision "$CODE_REV" >> "$LOG" 2>&1 &
echo $!
tail -f "$LOG"
```

`tail -f`는 Ctrl+C로 닫아도 백그라운드 실행은 계속된다. Pod 자체를 terminate하면 종료된다.
완료 기준은 JSON `status: completed`와 `total_rollouts`가 계획과 일치하는 것이다.
실행 중 config, split, score, code를 수정하거나 git pull하지 않는다.
코드 수정이 필요하면 기존 결과를 보존하고 새 protocol을 준비한다.

중단 후 같은 Pod/입력을 유지했다면 변수를 다시 설정한다. 완료된 조건은 같은 승인
run 명령에서 검증 후 재사용한다. **미완료 조건 폴더에 파일이 남았다면 자동 재개가
차단된다.** 이때는 해당 로그/artifact를 보존하고 원인을 확인한 뒤 복구해야 한다.
episode별 무손실 checkpoint는 아니다. plan/조건 폴더를 무작정 지우지 않는다.

## 8. 분석과 결과 확인

```bash
python "$CLI" analyze --config "$CFG"
python scripts/openvla/headgroups.py --config "$CFG"
sed -n '1,180p' "$OUT/analysis/report.md"
sed -n '1,180p' "$OUT/analysis/task_groups_v1.md"
```

핵심 파일:

- `analysis/report.md`: 전체 요약, feature별 성공률 하락, 순위 비교.
- `analysis/analysis.json`: `topk_contrasts`의 방법 간 차이와 CI.
- `analysis/paired_effects.json`: 같은 case에서 raw/feature 성공 여부를 짝지은 원자료.
- `analysis/task_groups_v1.md`: task 0·1 / 2–9 별 결과; feature를 재선정하지 않는다.
- `validation/`: 현재 head/edit parity 검증 기록.

작은 pilot의 0.688 vs 0.402 상관계수가 그대로 재현돼야 하는 것은 아니다.
raw 성공률이 낮은 task도 결과를 보고 제외하지 않는다. CI가 0을 포함하면 방법 간 우열은
불확실하다고 쓰고, 후보 panel을 전체 dictionary로 일반화하지 않는다.

## 9. 종료 전 백업

GitHub에는 코드/문서만, 원시 데이터·로컬 설정·결과는 private 저장소에 보관한다.
먼저 run이 완료되거나 완전히 중단됐는지 확인한다. 쓰는 중인 파일을 archive하지 않는다.

```bash
ARCHIVE=/workspace/full10-followup-v1-backup.tar.gz
tar -C /workspace -czf "$ARCHIVE" event-sae-head-results/full10-followup-v1 head-inputs/full10 Event-SAE-Head/configs/local/head-full10.yaml
sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
HF_HUB_OFFLINE=0 hf auth login
```

토큰은 터미널의 숨겨진 입력창에만 입력한다. 채팅·Git·로그에 쓰지 않는다.
공개 입력의 pinned revision은 코드에 있으므로 281 GiB 원본은 다시 업로드할 필요 없다.

```bash
export BACKUP_REPO=jiyeony/event-sae-head-full10-results
HF_HUB_OFFLINE=0 python -c 'import os;from huggingface_hub import HfApi;HfApi().create_repo(os.environ["BACKUP_REPO"],repo_type="dataset",private=True,exist_ok=True)'
HF_HUB_OFFLINE=0 python -c 'import os;from huggingface_hub import HfApi;assert HfApi().repo_info(os.environ["BACKUP_REPO"],repo_type="dataset").private,"STOP: backup repository is public"'
HF_HUB_OFFLINE=0 hf upload "$BACKUP_REPO" "$ARCHIVE" full10-followup-v1-backup.tar.gz --repo-type dataset
HF_HUB_OFFLINE=0 hf upload "$BACKUP_REPO" "${ARCHIVE}.sha256" full10-followup-v1-backup.tar.gz.sha256 --repo-type dataset
HF_HUB_OFFLINE=0 hf download "$BACKUP_REPO" full10-followup-v1-backup.tar.gz --repo-type dataset --local-dir /workspace/head-backup-verify
sha256sum "$ARCHIVE" /workspace/head-backup-verify/full10-followup-v1-backup.tar.gz
```

private 검사가 실패하면 업로드하지 않는다. **실제 원격 archive를 다시 받은 두 hash가
일치한 뒤** Pod 종료를 판단한다.
checksum 파일만 다운로드해서 비교하는 것보다 이 검증이 더 강하다.
기존 pilot용 `headrestore.py`의 기본 checksum은 이 새 archive용이 아니므로 그대로 적용하지 않는다.

## 10. Git 배포 및 검증 범위

개발 코드는 Head 저장소의 main에 배포한다. 실행 전 확인 명령:

```bash
git remote -v
git log -1 --oneline
git status --short
python -m pytest -q
git diff --check
```

`origin`은 Event-SAE-Head이어야 한다. Baseline 쪽 upstream에는 push하지 않는다.
local YAML과 실험 입력/결과는 Git ignore 대상이다. 커밋에는 코드·테스트·문서만 포함한다.

로컬 검증은 synthetic/CPU 데이터로 연결·분리·해시·재실행·회귀를 검사한다.
전체 공개 입력 다운로드, native LIBERO 초기상태 검증, RTX 4090 parity, full10 score/rollout은
**새 Pod에서 실제 실행해야 검증 완료**다. 소스 구현 완료와 GPU 실험 완료를 구분한다.

2026-09-21 최종 로컬 검증: **312 tests passed (5.56초)**, `git diff --check` 통과.
macOS sandbox에서는 자식 PyTorch 프로세스의 OpenMP 공유 메모리가 차단되어,
동일한 전체 테스트를 승인된 sandbox 외부 환경에서 실행해 통과를 확인했다.
공개 pinned TopK manifest 및 source prompt/index metadata도 검사해 실제 legacy schema를
테스트에 반영했다. 대용량 dense tensor 전체나 실제 GPU 결과를 검증한 것은 아니다.
