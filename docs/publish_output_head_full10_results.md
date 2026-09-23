# Full10 실험 결과를 Hugging Face에 공개하고 복구하기

대상: 완료된 `full10-followup-v1` 실험. 공개 저장소는
[jiyeony/event-sae-head-full10-results](https://huggingface.co/datasets/jiyeony/event-sae-head-full10-results)다.

목적은 **3,600회 행동 평가의 원자료와 분석 결과를 함께 보존하고, 다른 사람이 GPU 없이
내려받아 검사하고 읽을 수 있게 하는 것**이다. 백업 과정에서 rollout을 다시 실행하지 않는다.

## 1. 저장되는 내용

| 포함 | 내용 |
|---|---|
| 이번 실험 결과 전체 | 18개 조건의 `result.json`, episode별 성공 여부, 행동 기록, 실행 로그 및 실제 생성된 부속 파일 |
| 점수·검증·분석 | Head-KL 점수, readout 캐시, 출력층 bundle, parity 기록, 전체·task별 분석과 신뢰구간 |
| 이번 실험 입력 기록 | `head-inputs/full10/`의 split, source/generation/eval manifest, runtime 입력, discovery Event score와 연결 증거 |
| 실행 설정·소스 | 실제 사용한 local YAML, 결과에 기록된 코드 revision의 소스 archive, 백업 시점 환경 정보 |
| 공개용 도구·요약 | GPU 없이 동작하는 검사·복구 도구, README, 분석 보고서, 파일별 SHA-256 |

약 281 GiB의 공개 원본 dense 데이터, OpenVLA 전체 가중치, 기존 SAE는 중복 업로드하지
않고 원본 저장소·revision·hash를 기록한다. **이번 실행으로 새로 생성된 산출물은 보관한다.**
영상은 기존 파일이 있을 때 포함된다. 이번 설정에서 영상을 저장하지 않았다면 백업이
새 영상을 만들거나 복원하지 않는다. HF 토큰, cache, `.git` 전체는 공개 bundle에 넣지 않는다.

“약 40시간”은 사용자 관찰에 따른 대략적인 시간이다. 정확한 시간은 저장된 실행 시간
기록을 따른다. 조건별 시간 합계와 다운로드·대기까지 포함한 Pod 사용 시간은 다를 수 있다.

## 2. 완료된 Pod에서 도구 업데이트

로그에 `status: completed`, `total_rollouts: 3600`이 나왔고 `analyze`와 `headgroups`가
성공한 뒤 실행한다. **실험이 아직 진행 중이면 여기서 git pull하지 않는다.**

```bash
cd /workspace/Event-SAE-Head
git status --short
git pull --ff-only origin main
```

수정 중인 tracked 파일이 있거나 pull에 오류가 나오면 덮어쓰지 말고 먼저 확인한다.
기존 결과에 기록된 실험 commit은 유지되며, 새 백업 도구의 commit과 구분해 보관된다.

## 3. 전체 결과 포장 및 로컬 검사

```bash
BUNDLE=/workspace/full10-public-release-v1
CLI=scripts/openvla/headbundle.py

python "$CLI" pack --workspace /workspace --output "$BUNDLE"
```

기본값은 아래 두 경로다.

- 결과: `/workspace/event-sae-head-results/full10-followup-v1`
- 설정: `/workspace/Event-SAE-Head/configs/local/head-full10.yaml`

파일을 읽고 압축하므로 시간이 걸릴 수 있다. 기존 결과 폴더를 수정하거나 지우지 않는다.
출력 폴더를 다른 실험에 재사용하지 않는다.
`pack`은 기존 실험 환경의 PyYAML을 사용해 설정을 frozen plan hash와 대조한다.
`verify`, `restore`, `inspect`는 Python 3.10 이상 표준 라이브러리만 사용한다.

```bash
python "$CLI" verify --bundle "$BUNDLE"
```

검사는 archive의 실제 bytes와 포함 파일의 hash를 확인하고, 임시 복구한 결과의 계획·
조건·episode 연결을 검사한다. 오류가 있으면 공개하거나 Pod을 종료하기 전에 해결한다.

출력 구조:

```text
full10-public-release-v1/
  README.md
  experiment.tar.gz
  SHA256SUMS
  manifest.json
  reports/
  tools/
    headbundle.py
    headbundle_verify.py
```

`reports/`는 웹에서 바로 확인할 분석 사본이다. 전체 원자료는 `experiment.tar.gz`에 있다.
`SHA256SUMS`는 bundle 안의 상대 경로를 사용하므로 다른 컴퓨터에서도 확인할 수 있다.

## 4. 공개 저장소에 업로드

토큰은 `hf auth login`의 숨겨진 입력창에만 입력한다. 채팅·문서·명령 인수에 넣지 않는다.

```bash
export HF_HOME=/workspace/cache/head-full10-hf
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=0
export REPO=jiyeony/event-sae-head-full10-results

hf auth login
hf auth whoami
```

새 **public dataset**을 만든다. 이미 같은 이름의 저장소가 있다면 다음 검사에서 공개
여부를 확인한다. 기존 private 저장소의 공개 범위를 자동으로 변경하지 않는다.

```bash
python -c 'import os;from huggingface_hub import HfApi;HfApi().create_repo(os.environ["REPO"],repo_type="dataset",private=False,exist_ok=True)'
```

```bash
python -c 'import os;from huggingface_hub import HfApi;assert not HfApi().repo_info(os.environ["REPO"],repo_type="dataset").private,"STOP: repository is private"'
```

공개 검사에 오류가 없으면 bundle 폴더 전체를 업로드한다.

```bash
hf upload "$REPO" "$BUNDLE" . --repo-type dataset
```

업로드가 끝난 revision을 기록한다. 이후 검사는 이 revision으로 고정한다.

```bash
HF_REV=$(python -c 'import os;from huggingface_hub import HfApi;print(HfApi().repo_info(os.environ["REPO"],repo_type="dataset").sha)')
printf 'Published HF revision: %s\n' "$HF_REV"
```

출력된 revision과 저장소 주소를 연구 기록에 남긴다. 논문·팀원 안내에도 `main` 대신
이 revision을 지정하면 나중에 저장소에 파일이 추가돼도 같은 공개본을 받는다.

## 5. 실제 원격 파일 재다운로드 후 검사

**checksum 파일만 확인하지 말고, 업로드한 archive 자체를 다시 받아 검사한다.**
로컬 bundle, 다운로드 사본과 임시 압축 해제 공간이 함께 필요하다. 여유 공간을 먼저 확인한다.

```bash
df -h /workspace
CHECK=$(mktemp -d /workspace/full10-download-check.XXXXXX)
hf download "$REPO" --repo-type dataset --revision "$HF_REV" --local-dir "$CHECK" --force-download
```

다운로드한 도구로 다운로드한 bundle을 검사한다.

```bash
python "$CHECK/tools/headbundle.py" verify --bundle "$CHECK"
```

이 단계까지 성공해야 **공개된 파일을 다시 내려받아 전체 결과를 읽을 수 있는 상태**다.
성공 출력과 HF revision을 보관한 뒤 Pod 종료를 판단한다. 웹에서 README와 분석 보고서가
보이는지도 확인한다. 이 과정이 완료되기 전에는 Pod을 terminate하지 않는다.

## 6. 다른 사람이 GPU 없이 결과 확인하기

Python과 `huggingface_hub` CLI만 있으면 된다. 확인하려는 공개 revision을 아래 변수에 넣는다.

```bash
REPO=jiyeony/event-sae-head-full10-results
HF_REV=공개한_commit_hash
BUNDLE=./full10-public-release-v1

hf download "$REPO" --repo-type dataset --revision "$HF_REV" --local-dir "$BUNDLE"
python "$BUNDLE/tools/headbundle.py" verify --bundle "$BUNDLE"
```

`README.md`와 `reports/`는 압축 해제 없이 읽을 수 있다. episode별 기록까지 확인하려면
비어 있는 새 경로로 복구한다.

```bash
python "$BUNDLE/tools/headbundle.py" restore --bundle "$BUNDLE" --destination ./full10-restored
python "$BUNDLE/tools/headbundle.py" inspect --root ./full10-restored/results/full10-followup-v1
```

검사·복구·결과 열람에는 OpenVLA 가중치, 원본 dense 데이터, GPU가 필요하지 않다.
기존 local YAML의 `/workspace/...` 경로는 실험 증거이므로 원문 그대로 보관한다.
그 YAML을 임의의 컴퓨터에서 바로 실행할 수 있다는 뜻은 아니다. 원래 분석 코드를 다시
실행하거나 점수·rollout까지 재현하려면 archive 안의 소스·환경 기록과 원본 입력 참조를
따라 필요한 입력과 경로를 복원해야 한다.

## 7. 연구 결과를 인용할 때의 범위

이번 공개본은 고정된 OpenVLA·Layer 31 SAE와 LIBERO Spatial 10개 task에 대한 후속 실험이다.
상위 3개 feature는 동시에 제거한 것이 아니라 각각 제거한 성공률 하락을 평균했다.
신뢰구간은 고정된 task·feature 후보군에서 초기상태를 재표집한 불확실성이다.

- 기존 원본 수집 시점의 초기상태와 일부 generation 설정은 검증되지 않았다.
- 기존 SAE와 Event cluster는 재사용했다. 학습까지 완전히 분리된 평가라고 주장하지 않는다.
- task 0–1은 pilot에 사용됐고, task 2–9도 자동으로 확증 실험이 되는 것은 아니다.
- 점수 계산에 10개 task를 사용했으므로, 보지 못한 task로의 일반화 실험이 아니다.

원자료·코드·hash 공개는 이번 결과의 확인과 재분석을 돕지만, 원수집 때 없던 과거 증거를
추가로 만들어 주지는 않는다.

HF 명령어 참고: [Hugging Face 공식 CLI 안내](https://huggingface.co/docs/huggingface_hub/guides/cli).
