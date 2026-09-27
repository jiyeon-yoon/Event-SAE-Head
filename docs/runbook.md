# 출력층 민감도: 실행·복구 안내

연구 내용·결과·데이터별 역할은 [experiment.md](experiment.md)를 참고한다.
**이미 완료된 결과만 확인하려면 A만 실행한다. GPU Pod을 새로 만들 필요가 없다.**

## A. 공개 결과 다운로드·검증·복구 — GPU 불필요

Python 3.10 이상과 `hf` CLI를 사용한다. 공개 결과는 로그인 없이 다운로드할 수 있다.
CLI가 없으면 별도 환경에 `python -m pip install huggingface_hub`로 설치한다.

```bash
REPO=jiyeony/event-sae-head-full10-results
HF_REV=69c33d6a943c253dc97cf9a3968095276f7e9540
BUNDLE=./head-full10-release
export HF_HUB_OFFLINE=0

hf download "$REPO" --repo-type dataset --revision "$HF_REV" --local-dir "$BUNDLE"
python "$BUNDLE/tools/headbundle.py" verify --bundle "$BUNDLE"
```

정상: `status: verified`, `total_rollouts: 3600`, `conditions: 18`.
archive 약 1.53 GB와 임시 압축 해제 공간이 필요하다. 검증은 tensor/pickle을 실행하거나
모델을 재추론하지 않는다. 저장된 파일 해시·case·행동 기록·성공률 하락을 대조한다.
bootstrap 신뢰구간 전체 재계산과 simulator 재실행을 뜻하지 않는다.

보고서 바로 읽기:

```bash
sed -n '1,180p' "$BUNDLE/reports/report.md"
sed -n '1,180p' "$BUNDLE/reports/task_groups_v1.md"
```

`reports/episodes.csv`는 3,600회 평가 목록, `reports/summary.json`은 방법별 수치·비교 CI·시간이다.
전체 원자료까지 열려면 **아직 없는 새 목적지**에 복구한다.

```bash
RESTORED=./head-full10-restored
python "$BUNDLE/tools/headbundle.py" restore --bundle "$BUNDLE" --destination "$RESTORED"
python "$BUNDLE/tools/headbundle.py" inspect --root "$RESTORED/results/full10-followup-v1"
```

복구 구조:

```text
head-full10-restored/
  results/full10-followup-v1/  # 18개 run, 점수, head/readout, parity, plan, 분석, 로그
  inputs/full10/              # split, source/generation/eval, runtime, discovery Event score
  config/                    # 원본 YAML와 plan hash에 연결된 normalized JSON
  source/experiment-code.tar.gz
  sources/                   # 원본 다운로드 receipt와 pinned 입력 참조
  environment/               # 백업 시점 package/GPU 기록
```

원래 JSON/YAML의 `/workspace/...` 경로는 증거 보존을 위해 바꾸지 않는다. 다른 경로에서
원래 `analyze`까지 실행하려면 당시 코드·의존성·입력을 원래 경로로 복원해야 한다.
단순 열람/검증은 위 portable 도구만으로 된다. **pilot용 `headrestore.py`와 혼동하지 않는다.**

## B. 점수 계산부터 실제 rollout까지 새로 실행 — GPU·대용량 입력 필요

이번 40.88시간짜리 실험을 다시 계산하려는 경우에만 진행한다. 원본 5개 약 281 GiB,
OpenVLA 약 15 GB, SAE, Event 입력 및 결과 공간이 추가로 필요하다. 아래는 **기존 산출물이
없는 새 `/workspace`**를 전제로 한다. 복구한 결과 위에 새로운 실험을 덮어쓰지 않는다.
다른 모델/설정의 새 연구는 별도 protocol/output으로 설계해야 한다.

### B1. 실행 환경과 코드

기존 기록: RTX 4090 1장, 400 GB 디스크 설정, 아래 runtime 이미지.
현재 가용성·가격이나 새 장비에서의 수치 일치를 보장하는 목록은 아니다.

```text
ghcr.io/jiyeon-yoon/event-sae-runtime@sha256:ef895731ffd73985e1183c964b479bdcc7e97dfeabea6c8777b903c54ade1405
```

```bash
event-sae-init
event-sae-verify --require-gpu
cd /workspace
git clone https://github.com/jiyeon-yoon/Event-SAE-Head.git
cd /workspace/Event-SAE-Head
git checkout --detach 8924a8d9b62edb16d261785cbc2124729017f2b1
git status --short
```

마지막 출력이 비어 있어야 한다. 위 commit은 실제 full10 실행 코드다.
최신 main의 문서/백업 도구 revision과 구분한다. 이미 clone된 저장소라면 위 clone을
반복하거나 로컬 수정을 덮어쓰지 말고 상태부터 확인한다.

```bash
tmux new -s head-full10
```

tmux 안에서 다음 변수를 설정한다. 재접속/새 shell에서는 다시 설정한다.

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

### B2. 입력·분리·검증

다운로드 전 필요하면 `HF_HUB_OFFLINE=0 hf auth login`으로 로그인한다. 토큰은 로그인
입력창에만 넣는다. 전용 HF cache를 사용하며, 기존 다른 모델의 `main` 참조를 덮어쓰지 않는다.
**한 명령이 성공한 뒤 다음 명령으로 넘어간다.**

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python scripts/openvla/headfull.py download --execute
python scripts/openvla/headfull.py merge
python scripts/openvla/headfull.py setup --allow-libero
```

정상: `downloaded` → `merged_and_verified` → `configured`.
369 dense shards, 500 episodes, discovery 250 / validation 50 / evaluation 200, `mode: followup`.
입력 revision은 실행 commit의 downloader에 고정되어 있다. 원본 상태의 과거 hash가
없다는 한계는 유지되며, 현재 상태 hash를 과거 수집의 증거로 바꾸지 않는다.

```bash
python scripts/openvla/headsetup.py --config "$CFG" --snapshot "$MODEL" \
  --metadata-output "$META/head.json" --export
python "$CLI" prepare --config "$CFG"
python scripts/openvla/headinputs.py --config "$CFG" \
  --output "$META/runtime.pt" --allow-simulator
python "$CLI" validate --config "$CFG" --allow-model-execution
```

`validate`의 `status: passed`가 필요하다. 실패하면 tolerance/manifest를 임의로 완화하지 않는다.
이미 존재하는 `runtime.pt`를 무작정 덮어쓰지 않는다. TensorRT/Gym 경고 자체와 최종 실패를 구분한다.

### B3. 점수와 실행 계획

```bash
python "$CLI" score --config "$CFG"
python scripts/openvla/headevents.py --config "$CFG" --event-root "$EVENTS"
python "$CLI" plan --config "$CFG"
```

두 점수는 같은 discovery episode만 사용한다. `headevents`는 기존 cluster를 유지하고
discovery-only Event score를 재집계한다. 원래 full500 점수에 새 hash만 붙이는 방식이 아니다.
정상 계획은 `planned`, `selection_eval_overlap: false`, 최대 4,000회다.
공개 실험에서는 후보 중복 제거 후 16개 feature / 18조건 / 3,600회가 선택됐다.

```bash
PLAN="$OUT/plan/rollout_plan.json"
PLAN_HASH=$(python -c 'import json,sys;from event_sae.research.output_head.provenance import fingerprint;p=json.load(open(sys.argv[1]));h=p.pop("plan_hash");assert h==fingerprint(p);print(h)' "$PLAN")
CODE_REV=$(git rev-parse HEAD)
git status --short
LOG="$OUT/full10-run.log"
```

### B4. 계획·비용을 확인한 후에만 실행

아래부터 실제 장시간 GPU 평가가 시작된다. 기존 조건별 시간 합계는 약 40.88시간이지만
새 장비의 소요 시간/비용 보장은 아니다. 현재 plan과 Pod 요금을 확인한 뒤 실행한다.

```bash
nohup env PYTHONUNBUFFERED=1 python "$CLI" run --config "$CFG" --execute \
  --approved-plan-hash "$PLAN_HASH" --expected-code-revision "$CODE_REV" >> "$LOG" 2>&1 &
echo $!
tail -f "$LOG"
```

- `tail -f`의 Ctrl+C는 로그 보기만 종료한다. Pod 자체 종료는 실행을 중단한다.
- tmux 재접속: `tmux attach -t head-full10`. 실행 중 code/config/split/score를 바꾸거나 git pull하지 않는다.
- 완료 기준: 최종 `status: completed`와 계획의 `total_rollouts`가 일치.
- 같은 환경에서 완료된 조건은 재사용할 수 있지만, 미완료 조건은 episode 단위 자동 복구가 아니다.
  일부 파일이 남아 재개가 차단되면 로그를 보존하고 원인을 확인한다. 조건 폴더를 임의 삭제하지 않는다.

### B5. 분석

```bash
python "$CLI" analyze --config "$CFG"
python scripts/openvla/headgroups.py --config "$CFG"
sed -n '1,180p' "$OUT/analysis/report.md"
sed -n '1,180p' "$OUT/analysis/task_groups_v1.md"
```

`analysis.json`의 `topk_contrasts`는 방법 간 차이/CI, `paired_effects.json`은 같은
평가 case의 raw/edit 성공 여부다. subgroup에서도 기존 순위와 후보를 바꾸지 않는다.
과거 숫자와 일치하도록 후보·초기상태·seed를 사후 조정하지 않는다.

## C. 새 실행 결과 백업·공개 — 완료 후에만

기존 공개본은 이미 검증됐으므로 다시 업로드할 필요 없다. 아래는 새 실행의 백업 절차다.
**기존 공개본을 덮어쓰지 않도록 별도 결과 저장소를 사용한다.**

실험과 분석을 완료한 뒤에만 최신 백업 도구를 받는다. Git 작업물이 깨끗한지 먼저 확인한다.
백업 도구는 결과에 기록된 실행 commit의 소스를 따로 archive하므로 과거 실행 이력을 바꾸지 않는다.

```bash
cd /workspace/Event-SAE-Head
git status --short
git switch main
git pull --ff-only origin main
BUNDLE=/workspace/full10-public-release-v1
python scripts/openvla/headbundle.py pack --workspace /workspace --output "$BUNDLE"
python scripts/openvla/headbundle.py verify --bundle "$BUNDLE"
```

pack은 기존 실험 환경의 PyYAML이 필요하다. 출력 폴더는 새 경로여야 한다.
전체 결과/생성 입력/설정/실행 소스를 보존하지만 원본 dense·전체 모델·SAE는 pinned 참조로 남긴다.

```bash
export HF_HUB_OFFLINE=0
export REPO=jiyeony/event-sae-head-full10-results-rerun
hf auth login
python -c 'import os;from huggingface_hub import HfApi;HfApi().create_repo(os.environ["REPO"],repo_type="dataset",private=False,exist_ok=True)'
python -c 'import os;from huggingface_hub import HfApi;assert not HfApi().repo_info(os.environ["REPO"],repo_type="dataset").private'
hf upload "$REPO" "$BUNDLE" . --repo-type dataset
HF_REV=$(python -c 'import os;from huggingface_hub import HfApi;print(HfApi().repo_info(os.environ["REPO"],repo_type="dataset").sha)')
printf 'Published revision: %s\n' "$HF_REV"
```

`REPO`는 새 실험에 맞는 아직 사용하지 않은 이름으로 정한다. `pack`이 만드는 README의
다운로드 예시는 원래 공개 저장소를 가리키므로, 새 저장소의 안내에는 위 REPO/HF_REV를 명시한다.
원본 bundle 안 파일을 직접 고치면 checksum 검증이 깨진다.
private 저장소의 공개 여부를 자동 변경하지 않는다. 공유 전에 민감한 사용자 파일이 없는지도 확인한다.

실제 원격 archive를 다시 받아 검사한다. checksum 파일만 비교하는 것으로 끝내지 않는다.

```bash
df -h /workspace
CHECK=$(mktemp -d /workspace/full10-download-check.XXXXXX)
hf download "$REPO" --repo-type dataset --revision "$HF_REV" --local-dir "$CHECK" --force-download
python "$CHECK/tools/headbundle.py" verify --bundle "$CHECK"
```

`verified`와 plan의 조건/rollout 수가 일치한 뒤 HF revision을 보관한다.
**그 밖에 필요한 Pod 파일까지 백업했다면 terminate해도 된다.**

## D. 개발 점검·이전 기록

```bash
git remote -v
git status --short
python -m pytest -q
git diff --check
```

`origin`은 Head여야 한다. Baseline/다른 팀원 저장소에는 push하지 않는다.
공용 설정 YAML의 입력은 비어 있으므로 `audit`이 경로 누락으로 blocked인 것은 정상이다.
GPU 결과 확인과 로컬 단위 테스트는 구분한다. macOS sandbox에서 OpenMP 공유 메모리 오류가
나면 환경 제약을 확인하며, 테스트 실패를 숨기거나 실험 수치를 바꾸지 않는다.

세부 CLI schema·과거 시행착오·SAE 학습/Baseline 재현은 [archive](archive)에 보관했다.
옛 문서 경로가 필요한 경우 문서 정리 전 commit `4a8dbf8`에서도 확인할 수 있다.
