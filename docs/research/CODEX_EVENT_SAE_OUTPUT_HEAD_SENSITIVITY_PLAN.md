## Codex 구현 지시서: Event-SAE 출력층 민감도로 feature의 행동적 중요도 예측하기

- 문서 버전: 1.1 / 2026-09-20
- 개발 대상 저장소: `https://github.com/jiyeon-yoon/Event-SAE-Head` (public)
- 기반 저장소: `https://github.com/jiyeon-yoon/Event-SAE-Baseline`
- 기반 코드 검토 commit: `56e9f012aa88532f9880259b371d9300a8013b9e`
- 저장 경로: `docs/research/CODEX_EVENT_SAE_OUTPUT_HEAD_SENSITIVITY_PLAN.md`
- 기존 재현 runbook의 고정 baseline: `fd3bc485668b8fb32b3948a9b642859f41b16277`
- 선택한 독립 실험: **③ 출력층 민감도 예측**
- 이 문서는 구현 요구사항이다. M0~M3 구현과 CPU 검증 상태·현재 CLI의 정확한 지원 범위는
  [실행 설명서](../output_head_sensitivity.md)를 참고한다. M4~M5 실제 검증·실험은 미실행이다.

v1.1은 전체 v1.0 설계를 유지하면서 다음 검토 결과를 반영한다.

- `Event-SAE-Head`를 독립 개발 대상으로 지정하고 기존 팀 저장소와 입력 데이터를 보호한다.
- Confirmatory evaluation split을 M4 pilot 전에 고정한다.
- Top-K 동점 처리와 무작위 audit 표집·중복 규칙을 명시한다.
- CPU CLI의 기존 eval 패키지 import 문제를 수정 지도에 포함한다.
- 선택한 비교 방법에 필요한 artifact만 필수 검증한다.
- Head/edit parity 결과를 실제 검증한 수치 연산 조건에 연결한다.
- 기존 paired 성공 로그로 계산 가능한 outcome switch 보조 통계를 추가한다.

### 0. Codex에게 먼저 전달할 작업 지시

당신은 이 저장소에서 연구 코드를 구현하는 엔지니어다. 아래 요구사항을 읽고, 기존 코드를 실제로 확인한 다음 변경을 구현하라. 설계 설명만 반환하지 말고 코드, 작은 테스트, 실행 설명서를 작성하라.

이번 연구의 질문은 다음과 같다.

> 기존 OpenVLA L31 SAE feature를 제거했을 때 발생하는 **출력층의 국소적 변화**를 기존 activation으로 계산하여, 그 feature를 closed-loop rollout에서 제거했을 때의 **성공률 감소**를 예측하거나 중요한 후보를 우선 선정할 수 있는가?

**처음 작업에서는 M0~M3의 개발과 synthetic CPU 테스트까지만 수행하라.** M4~M5에 필요한 명령과 계획 파일은 구현하되, 실제 모델 검증·실데이터 대량 계산·GPU rollout은 실행하지 마라. 기존 파일을 읽는 가벼운 metadata audit은 허용한다.

작업 순서:

1. 적용되는 `AGENTS.md`, 현재 Git root, checkout, 변경 중인 파일을 확인한다. 개발 root는 독립 저장소 `Event-SAE-Head`여야 한다. 이 문서의 기반 commit 이후 변경은 보존하고 반영한다. 과거 commit으로 강제 checkout/reset하지 않는다.
2. 실제 클래스·함수·artifact 형식을 확인하고 간단한 변경 계획을 작성한다.
3. 기존 동작을 보존하며 M0~M3를 구현한다. 이미 같은 기능이 있으면 재사용한다.
4. 네트워크·모델·LIBERO가 없는 환경에서도 실행할 수 있는 CPU 테스트를 작성하고 실행한다.
5. 변경 파일, 테스트 명령과 실제 결과, 실데이터 검증 여부, 차단 요인, 다음 승인 후 실행할 명령을 보고한다.

금지 사항:

- 신규 VLA 도입, OpenVLA fine-tuning, SAE 재학습, SAE dictionary 변경.
- 기존 discovery rollout 전체 재수집.
- object pose, contact/grasp, 단계별 reward, subgoal predicate 추가 수집 또는 이를 필수 입력으로 지정.
- 다른 제안 A~E의 실험을 이번 실험의 선행조건으로 추가.
- 기존 `configs/reproduction/`와 논문 규모 sweep의 보호 조건을 삭제하거나 완화.
- 대용량 데이터/모델 자동 다운로드, GPU 대량 계산, simulator rollout 자동 실행.
- 외부 업로드, paid API 호출, `git push`, 사용자의 기존 변경 덮어쓰기.
- 실측하지 않은 성능·시간·성공률·통계 결과 생성. Synthetic 결과를 실제 실험으로 표시.

실제 checkpoint나 데이터가 없어도 가능한 CPU 모듈과 테스트는 완성하라. 필요한 입력이 없다는 이유로 개발 전체를 중단하지 말고, 실데이터 단계만 `blocked`로 보고하라.

### 1. 고정할 범위와 연구 주장

#### 1.1 그대로 유지하는 것

- 기존 OpenVLA LIBERO-Spatial checkpoint.
- 기존 마지막 decoder block, L31의 `post_mlp_residual` SAE.
- 기존 dense activation, activation index, sparse Top-K, event score와 ranking artifact.
- 기존 residual-preserving single-feature suppression 연산.
- 기존 LIBERO task, 초기 상태, 성공/실패 판정.

기존 runbook은 10 tasks × 50 episodes, 369 dense shards, `4096 → 32768`, `k=64` SAE를 명시한다. 이는 **확인할 기존 구성**이지 새 코드에 무조건 hard-code할 상수가 아니다. 실제 manifest/config에서 검증한다. 전체 intervention sweep은 코드가 제공되어 있을 뿐, 모든 feature의 개입 결과가 이미 있다고 가정하지 않는다. [S1]

#### 1.2 새로 만드는 것

```text
기존 dense activation + 기존 SAE + 같은 OpenVLA의 final norm/lm_head
    → 각 policy forward에서 실제 다음 행동 token을 예측하는 hidden row 식별
    → feature를 하나씩 제거한 hidden state 계산
    → 실제 final norm + lm_head로 출력 변화량 계산
    → feature별 offline sensitivity score와 순위 생성
    → 기존 ranking·단순 활성 통계와 비교할 후보 panel 고정
    → 승인된 수만큼 새 intervention rollout 실행
    → offline score와 paired 성공률 감소의 관계·후보 선정 효율 평가
```

오프라인 점수는 우선 **학습이 필요 없는 계산식 기반 predictor**로 구현한다. 별도 회귀모델이나 신경망을 학습하는 것은 MVP에 포함하지 않는다.

#### 1.3 반드시 구분할 것

- 오프라인 점수 계산: 새로운 환경 rollout 없이 기존 activation을 다시 읽어 수행한다. 재계산 비용은 존재한다.
- 행동적 검증: 기존과 동일한 종류의 새 intervention rollout이 필요하다. 성공/실패, 초기 상태 hash, 개입 조건 등 최소 로그만 사용한다.
- 국소 출력 민감도: **baseline 문맥을 고정한 다음-token 출력 변화**다.
- closed-loop 효과: 바뀐 행동과 관측이 누적된 실제 rollout 결과다.

국소 민감도가 크다고 성공률이 반드시 감소하는 것은 아니다. 개입이 오히려 성공률을 높이거나, 출력은 바뀌어도 성공률은 같을 수 있다. 긍정·부정 결과를 모두 보존한다.

이번 연구로 “grasp 개념을 제거했다”, “특정 subgoal을 담당한다”, “인과 검증 없이 중요성이 증명됐다”는 결론을 내리지 않는다. 주장 범위는 **고정된 OpenVLA·SAE·평가 환경에서 후보 선정과 행동 영향 예측에 유용한가**다.

#### 1.4 저장소와 파일 경계

- 구현·테스트·문서 변경은 새 `Event-SAE-Head` 안에서 수행한다.
- 기존 `Event-SAE-Baseline`, `Event-SAE-Pipeline`, `event-sae-runtime` 폴더 및 그 Git metadata는 읽기 전용 참고 대상으로 유지한다.
- 기반 코드는 이미 새 저장소에 복제되어 있다. 수정이 필요한 기존 eval/config 파일도 새 저장소 내부 사본만 수정한다.
- `origin`은 새 Head 저장소, `upstream`은 Baseline 확인용이다. `upstream`의 push URL은 차단하고 기본 push remote는 `origin`으로 둔다. 최초 구현 지시에는 commit/push 또는 remote 변경이 포함되지 않는다.
- 기존 activation·SAE·모델 checkpoint는 외부 경로에서 읽는다. 파생 head bundle, readout cache, 실험 결과는 새 연구 output root에 쓴다. 기존 입력 경로를 output으로 지정하면 거부한다.
- Git에는 코드·공용 설정 예시·설계서·실행 설명서를 포함한다. `/artifacts/`, `/runs/`, `/data/`, `/checkpoints/`, `/research_outputs/`, 로컬 전용 경로 설정과 토큰은 포함하지 않는다.
- 기존 LICENSE와 원본·Baseline 출처를 보존한다. 새 연구의 측정 결과와 상속한 Baseline의 검증 상태를 문서에서 구별한다.

### 2. 현재 구현을 기준으로 한 재사용·수정 지도

아래는 검토한 commit의 코드에서 확인한 사항이다. 실제 checkout과 비교한 뒤 사용하라.

| 기존 파일 | 확인한 기능 | 이번 작업 |
|---|---|---|
| `event_sae/openvla/activations.py` | decoder 출력 dense 수집, forward별 index, `load_batch_topk_sae` | SAE loader 재사용. 기존 수집기를 확장하지 않는다. |
| `scripts/extract_topk.py` | dense → sparse Top-K, manifest/hash/resume | index 검증·배치 처리 방식을 참고. 전체 Top-K를 다시 생성할 필요는 없다. |
| `event_sae/openvla/eval/model.py` | pinned 모델/processor, BF16, crop/prompt, `predict_action(..., do_sample=False)` | 실모델 parity 검증에서 재사용. 오프라인 모듈의 top-level import로 가져오지 않는다. |
| `event_sae/openvla/intervene.py` | `x + Dec(z') - Dec(z)` single-feature hook | 행동 평가의 기준 연산으로 그대로 사용한다. |
| `event_sae/openvla/eval/__init__.py` | 현재 runner를 즉시 import하여 config import에도 simulator 의존성이 유입됨 | runner export를 lazy하게 바꾸거나 CPU CLI가 이 경로를 피하도록 구현한다. 기존 public import 호환성은 보존한다. |
| `event_sae/openvla/eval/config.py` | task subset, trial 수, YAML 설정 | 필요 시 명시적 trial index 선택만 backward-compatible하게 추가한다. |
| `event_sae/openvla/eval/runner.py` | 초기 상태 hash, episode별 성공/예외 기록 | 필요 시 trial index 선택 지원. 원래 기본 순회와 성공 판정은 보존한다. |
| `event_sae/scoring/rankings.py` | event/window/task/random ranking | 원래 baseline 정의 재사용. research에서는 불완전한 legacy matrix fallback을 엄격히 검증한다. |
| `event_sae/scoring/score_matrix.py` | event/window/task 점수 matrix 생성 | strict split이 필요할 때 기존 데이터의 episode filter 경로를 확인한다. 원 scoring 공식을 변경하지 않는다. |
| `scripts/openvla/intervene.py` | 단일 feature 실행, provenance, 결과 검증 | 새로운 research runner가 호출/재사용한다. |
| `scripts/openvla/evaluate_policy.py` | raw/reconstruction 평가 | raw baseline 실행 경로 재사용. reconstruction은 이번 실험의 baseline이 아니다. |
| `scripts/openvla/run_intervention_development_validation.py` | alpha=1 identity, 초기 상태 pairing, hook 검증 | 작은 구현 검증의 참고/재사용 대상이다. |
| `scripts/openvla/run_intervention_sweep.py` | 네 ranking 각 5개, 10 tasks × 50 trials 고정 sweep | **보존한다.** 새 ranking과 소규모 실험은 별도 research runner로 처리한다. |
| `docs/reproduce_libero_spatial_500.md` | 기존 입력·checkpoint·산출물 경로 | 기존 artifact 위치를 찾는 기준이다. runbook 자체를 새 실험으로 대체하지 않는다. |

관련 코드 근거는 [S1]~[S7]에 정리했다.

CPU import에서 특히 주의할 점: 현재 `from event_sae.openvla.eval.config import ...`도 먼저 `eval/__init__.py`를 실행하여 runner, LIBERO, TensorFlow를 가져올 수 있다. 단순히 model import를 함수 안으로 옮기는 것만으로 해결됐다고 판단하지 않는다. 새 저장소의 패키지 초기화를 lazy하게 만들거나 경량 config 경로로 분리하고, 별도 Python process에서 `--help`, `audit`, `plan`을 실행하여 모델·simulator import가 발생하지 않는지 검사한다.

#### 2.1 기본 model revision

기존 재현 설정에서 확인한 값은 다음과 같다. [S2]

```yaml
checkpoint: openvla/openvla-7b-finetuned-libero-spatial
revision: 962318cec55ac10993ff0f5f43eda9a270b4c873
code_revision: 47a0ec7fc4ec123775a391911046cf33cf9ed83f
center_crop: true
```

이 값과 실제 수집 데이터의 provenance가 일치하는지 확인한다. OpenVLA base checkpoint나 다른 suite의 `lm_head`로 대체하면 안 된다.

기존 환경 YAML에는 `transformers==4.48.1`이 명시되어 있다. 실제 설치 환경·lockfile·remote model code를 다시 확인한다. 공개 upstream 소스와 고정 HF remote code가 같다고 가정하지 않는다. [S8, S9]

### 3. 올바른 출력층 계산의 정의

#### 3.1 Hook 위치

현재 intervention은 `model.language_model.model.layers[layer_idx]`의 출력 tuple 첫 항목에 작용한다. 기존 L31은 final norm **이전**의 마지막 decoder 출력이다. 일반적인 해당 Llama 경로는 다음과 같다. [S3, S8]

```text
마지막 decoder block 출력 h
    → model.language_model.model.norm
    → model.language_model.lm_head
    → generation에서 사용할 다음-token logits
    → token 선택 및 OpenVLA action 변환
```

M0에서는 읽을 수 있는 local config로 마지막 decoder layer 여부를 확인하고, 모델 객체가 없는 경우 runtime 확인은 `pending`으로 기록한다. M4에서 실제 로드된 모델의 `layer_idx == len(layers) - 1`을 반드시 검증한다. 중간 층 activation을 `norm → lm_head`에 바로 넣은 결과는 실제 downstream 출력이 아니므로 이번 모드에서 거부한다. 별도의 logit-lens 실험으로 조용히 전환하지 않는다.

#### 3.2 Feature suppression

기준 연산은 기존 hook과 동일해야 한다.

```python
# 설명용 의사코드. 실제 sae 구현과 dtype 경로를 사용한다.
h = cached_hidden.to(original_hidden_dtype)
x = h.to(torch.float32)
z = sae.encode(x)
z_edit = z.clone()
z_edit[..., feature_id] *= alpha
x_edit = x + (sae.decode(z_edit) - sae.decode(z))
h_edit = x_edit.to(original_hidden_dtype)

logits_base = lm_head(final_norm(h))
logits_edit = lm_head(final_norm(h_edit))
```

- 기본 `alpha=0.0`, identity 검증 `alpha=1.0`.
- `Dec(z_edit)`만 넣는 reconstruction replacement로 바꾸지 않는다.
- decoder가 affine임이 검증되면 수학적으로 `x_edit = x + (alpha-1) * z_i * d_i`다.
- `d_i`는 실제 checkpoint의 decoder 방향이다. weight의 행/열 방향을 추측하지 말고 `decode`와 비교 테스트한다.
- decoder bias는 차분에서 사라지지만, 코드에서 직접 잘못 빼거나 두 번 더하지 않는다.
- feature를 제거한 뒤 SAE를 다시 encode하여 “제거 여부”를 판단하지 않는다. 개입은 **원래 encode로 얻은 latent 한 항목을 편집하는 연산**이다.

#### 3.3 수치 정확도: reference와 fast path를 구분

기존 hook은 FP32 SAE 연산 후 원래 hidden dtype으로 cast한다. 이 cast와 실제 RMSNorm의 계산 순서를 유지한다.

`Dec(z_edit)-Dec(z)`와 `z_i*d_i`는 수학적으로 같아도 유한 정밀도에서는 다를 수 있다. BF16 경계 근처에서는 작은 차이가 argmax를 바꿀 수 있다.

따라서:

1. 작은 batch에서는 **기존 hook 방식의 reference edit + 실제 norm/head**를 구현한다.
2. 빠른 single-direction edit를 추가할 때 hidden 오차, logit 오차, argmax 불일치율을 reference와 비교한다.
3. 통과 여부와 허용오차를 artifact에 기록한다. 통과시키려고 허용오차를 자동 확대하지 않는다.
4. runtime-equivalent, tolerance-validated fast, FP32 diagnostic을 서로 다른 모드로 표시한다.
5. 최종 후보에 대해서는 reference 경로로 오프라인 점수를 재검사하는 기능을 제공한다.

`d_i @ W_unembed`만 계산한 값은 이번 predictor의 대체물이 아니다. final norm, 실제 활성값, 원래 hidden state, 출력 분포를 반영해야 한다.

### 4. 기존 activation에서 readout row 복원

이 단계가 틀리면 이후 모든 결과가 무의미하다. feature 점수 구현보다 먼저 검증한다.

#### 4.1 사용할 기존 index

dense index는 forward별로 대략 다음 필드를 제공한다. [S4]

```text
layer_idx
shard_path
row_start, row_end          # end-exclusive
matrix row의 episode_num
step_in_episode
task_id, task_episode_idx
global_forward_idx
```

collector는 한 forward의 row들을 flatten하여 저장하고, forward 경계를 shard 중간에서 나누지 않도록 구현되어 있다. 원래 inference가 batch size 1이라는 점은 데이터와 코드 provenance로 확인해야 한다. 파일의 row 수만 보고 batch size를 임의로 단정하지 않는다.

#### 4.2 정확한 매핑 규칙

검증된 batch-size-1 generation에서는 **각 forward에서 다음 token을 예측하는 마지막 유효 row**를 선택한다. 해당 기존 데이터가 padding 없는 단일 입력임이 확인되면 `row_end - 1`이다.

- 하나의 environment step 안의 forward들을 `global_forward_idx`로 정렬한다.
- 첫 forward의 마지막 prompt row가 첫 행동 token을 예측한다.
- KV cache를 쓰는 후속 forward의 현재 row가 다음 행동 token을 예측한다.
- 기존 경로가 7D action을 생성한다면 일반적으로 한 action query당 7개 prediction forward가 대응된다. **실제 분포와 generation code로 확인한 뒤** `action_dim_index=0..6`을 부여한다.
- “마지막 7개 row”, “281개 중 특정 위치”, “token_idx가 큰 row” 같은 하드코딩은 금지한다.
- 마지막으로 생성된 action token 자체를 입력으로 넣는 추가 forward가 항상 있다고 가정하지 않는다.
- sparse artifact의 `token_idx=0`도 cached decode의 유효 readout일 수 있다.
- `global_forward_idx` 숫자 자체의 연속성은 강제하지 않는다. subset/병합으로 생긴 간격과 실제 누락을 구별하고, row range 및 원본 기록을 검증한다.
- 원본 task pair run들을 병합하면서 바뀐 episode/forward ID를 올바르게 사용한다. 다른 run의 같은 숫자 ID를 동일 sample로 취급하지 않는다.

#### 4.3 Audit 보고서와 실패 조건

다음 정보를 먼저 출력한다.

```text
총 tasks / episodes / environment steps / forwards
forward별 row 수 분포
environment step별 forward 수 분포
batch-size-1와 padding/generation 경로의 근거
7D mapping 성공/실패 step 수와 사유
누락·중복·비연속 index·shard 누락·다른 layer 혼입
unknown 또는 잘못된 task_episode_idx
```

strict mode에서 예상과 다른 그룹은 오류다. 탐색용 exclude 모드를 제공하더라도 누락 이유·비율을 명시하고, 결과에서 모집단이 달라졌음을 표시한다. 누락된 action dimension을 0으로 채우지 않는다. 잘못된 index를 임의로 복원하지 않는다.

#### 4.4 SAE encode의 batch 의존성 검사

저장된 sparse Top-64는 후보 확인·캐시 탐색에는 쓸 수 있지만, 없는 feature를 무조건 실제 0이라고 단정하지 않는다. BatchTopK inference에서 실제 nonzero 수와 저장 Top-K의 관계를 검사한다.

새 predictor에서는 원칙적으로 dense readout에 **기존 SAE의 실제 `encode`**를 적용한다. 단, 먼저 다음 동등성을 검증한다.

```text
원래 전체 forward rows를 encode한 뒤 readout만 선택한 z
vs.
readout rows만 모아 encode한 z
```

eval encode가 row별로 독립적이면 readout-only encode를 사용한다. batch grouping에 따라 달라지면 원래 forward grouping으로 encode하고 readout만 추출한다. 다른 batch 크기 때문에 실험할 feature가 달라지는 것을 허용하지 않는다.

#### 4.5 파생 readout cache

기존 dense 파일은 수정하지 않는다. 선택한 소규모 readout만 새 연구 디렉터리에 저장한다.

```text
source_run_id / task_id / task_episode_idx / episode_uid
step_in_episode / forward_idx / action_dim_index
source_shard / source_row
hidden                      # 선택 row의 원래 값
실제 nonzero feature ids/values 또는 검증된 sparse encoding
source index / SAE / mapping / split hashes
```

캐시는 선택한 sample만 포함하며, 전체 `[모든 row, 32768]` dense latent나 전체 edited logits를 저장하지 않는다. 여러 score가 같은 readout cache를 재사용하게 한다. 기존 수집은 read-only다.

### 5. 같은 모델의 output-head bundle 준비

#### 5.1 필요한 값

기존 local checkpoint 또는 기존 runtime model에서 다음만 추출한다.

- final norm 종류, weight, epsilon, dtype, 정확한 forward 정의/version.
- `lm_head.weight`, 존재하면 bias, dtype, 출력 vocabulary 크기.
- hidden dimension, layer 수, target layer/hook 명세.
- model revision, remote code revision, local 파일 hash.
- action token 변환에 필요한 effective vocab size, bin centers, 실제 action dimension, unnormalization stats.
- generation config와 실제 적용되는 logits processor/warper 정보.

`output_head.safetensors` + `output_head_manifest.json` 같은 작은 파생 bundle로 보관한다. 전체 7B 모델을 매 점수 계산마다 다시 로드하지 않는다.

Safetensors index에서 필요한 tensor가 들어 있는 local shard만 읽는 경로를 우선 검토한다. tensor key, tied weights, dtype를 확인할 수 없으면 임의로 추정하지 않는다. 안전하게 추출할 수 없는 포맷은 기존 runtime model에서 한 번 추출하는 경로를 제공한다.

**새 모델을 받는 작업이 아니다.** local checkpoint가 없으면 위치를 보고하고 해당 단계만 차단한다. 모델 다운로드는 사용자의 별도 승인 없이는 실행하지 않는다.

#### 5.2 전체 vocab과 action subset을 구별

공개 upstream OpenVLA는 생성된 token ID를 effective vocab size와 bin centers로 변환하고 범위를 clip한 뒤 action을 unnormalize한다. 실제 pinned code를 검사하여 같은 규칙을 사용한다. [S9]

따라서 다음을 금지한다.

- `lm_head`의 마지막 256열/행을 검증 없이 action vocabulary라고 가정.
- tokenizer 전체 길이, padded model vocab, effective vocab을 같은 값으로 취급.
- action bin의 개수와 bin center의 개수를 구별하지 않음.
- 실제 generation이 full vocab을 고려하는데 action subset에만 softmax/argmax하고 이를 실제 정책 출력이라고 부름.
- gripper 후처리된 `actions.json`만으로 원래 token ID를 정확히 역산했다고 주장.

MVP의 주 점수는 **raw head full-vocabulary KL**로 정의한다. 실제 generation에 별도 processor가 있다면 이것은 processor 이전 점수임을 명시한다. processor 적용 후의 token 선택 비교는 정확히 재현할 수 있을 때만 제공한다.

action subset 점수를 추가한다면 `conditional_action_vocab_*`처럼 이름을 분리하고, 원래 full-vocab 확률질량 중 action subset에 들어가는 비율도 함께 저장한다.

### 6. 오프라인 점수와 aggregation

#### 6.1 주 점수: full-vocabulary KL

동일한 baseline 문맥의 readout row `r`에 대해:

```text
l0(r)   = LMHead(FinalNorm(h(r)))
li(r)   = LMHead(FinalNorm(Edit_i(h(r), alpha)))
logp0   = log_softmax(float32(l0))
logpi   = log_softmax(float32(li))
s_i(r)  = sum_v exp(logp0_v) * (logp0_v - logpi_v)
```

정의는 `KL(p_base || p_edit)`로 고정한다. 방향을 바꾸거나 temperature를 label 결과에 맞춰 사후 조정하지 않는다. alpha=1 및 inactive feature는 0이 되어야 한다. 작은 음수 반올림은 사전 설정한 tolerance 안에서만 0으로 처리하고, 큰 음수/NaN/Inf는 오류로 보고한다.

**실제 head forward의 dtype는 보존하고**, log-softmax·KL reduction은 FP32 이상으로 수행한다. 처음부터 head weight 전체를 FP32로 바꾼 별도 실험은 reference/runtime과 분리한다.

#### 6.2 같은 계산에서 얻을 보조 지표

MVP에는 다음 정도만 넣는다.

| 지표 | 정의/주의 |
|---|---|
| `full_vocab_argmax_flip_rate` | raw head top-1 token이 바뀌는 비율. processor가 있으면 실제 생성 token 변화와 구별. |
| `decoded_bin_change_rate` | 실제 검증된 ID→bin 변환 후 값이 바뀌는 비율. 서로 다른 token이 같은 clip된 bin이 될 수 있음. |
| `normalized_bin_abs_change` | 해당 action dimension의 normalized bin 값 차이. 번역/회전의 실제 단위를 섞어 물리적 거리로 부르지 않음. |
| `readout_activation_frequency` | 같은 sample에서 실제 z_i가 nonzero인 비율. |
| `mean_readout_activation` | 같은 sample에서 z_i 평균. 모든 sample을 분모에 포함. |
| `mean_edit_norm` | 같은 sample에서 실제 hidden 변화 norm 평균. 값이 큰 feature와의 비교용. |

보조 지표를 계산할 metadata가 없으면 `unavailable`과 사유를 기록한다. KL은 head bundle이 유효하면 별도로 계산할 수 있다.

#### 6.3 분모와 가중치

Feature universe는 기본적으로 checkpoint의 전체 dictionary ID 집합으로 둔다. 모든 feature에 대해 편집 forward를 수행한다는 뜻은 아니다. 선택한 readout에서 한 번도 활성화되지 않은 feature는 그 sample에 대한 점수가 정확히 0이며 `inactive_in_sample`로 표시한다. 이를 전체 데이터에서 dead라고 부르지 않는다. 기존 event 상위 feature가 readout에서 inactive여도 비교 panel에서 자동 제거하지 않는다.

feature가 켜진 row만 계산하는 최적화는 허용한다. 그러나 **점수의 분모는 모든 선택된 readout**이다. inactive row의 점수 0도 포함해야 한다. active-only 평균은 이름을 달리한 진단값일 뿐 주 score로 대체하지 않는다.

기본 집계는 다음과 같다.

```text
한 step의 action dimensions 평균
→ 한 episode의 선택된 steps 평균
→ 한 task의 episodes 평균
→ tasks의 동일 가중 평균
```

episode 길이 또는 특정 task의 token 수가 점수를 지배하지 않게 한다. 모든 신규 readout 통계에는 같은 sample manifest와 가중치를 사용한다. Feature별 제외 row를 다르게 만들지 않는다.

각 action dimension, task별 score도 저장하되 결과가 좋은 dimension만 사후 선택하여 메인 점수로 바꾸지 않는다.

#### 6.4 Autoregressive 한계

후속 action dimension의 저장된 hidden state는 **baseline에서 생성된 앞선 action tokens**에 조건화되어 있다. 첫 token의 편집이 다음 token의 문맥까지 바꾼 효과는 기존 hidden rows만으로 계산되지 않는다.

따라서:

- 점수의 명칭에 `fixed_prefix` 또는 동등한 설명을 포함한다.
- 7개 편집 결과를 붙여 “실제 counterfactual action sequence”라고 저장하지 않는다.
- 값이 커졌다는 이유만으로 성공률 감소를 예측 완료했다고 보고하지 않는다.
- 별도 개입 rollout이 이 한계를 포함한 predictor의 실제 유용성을 평가한다.

### 7. 실모델 일치성 검증: 대량 실험 전 통과해야 할 관문

이 절차는 M4에서 사용자 승인 후 작은 범위로 수행한다. 새로운 discovery dataset을 만드는 작업이 아니다.

#### 7.1 Head parity

같은 실제 forward에서 마지막 block hidden과 원래 downstream logits를 소수 확인하고, 저장/추출한 head bundle로 계산한 logits와 비교한다.

- baseline logits 오차, argmax 불일치율.
- actual norm epsilon/dtype/weight, model revision.
- generation processor 적용 여부와 stage.
- 원래 BF16 hidden을 FP32 파일에서 복원했을 때의 cast 일치성.

원본 logits가 기존 파일에 없다고 데이터 전체를 다시 수집하지 않는다. 승인된 소수 `get_action` 호출로 consistency test를 수행하거나, actual runtime 검증 전 상태로 명시한다. 기존 영상에서 복원한 JPEG 프레임을 원 수집 input과 bit-identical하다고 주장하지 않는다.

#### 7.2 Edit parity

동일한 고정 입력/문맥에서 기존 hook의 feature edit와 offline reference edit가 같은 readout logits를 만드는지 확인한다.

- alpha=1 identity.
- 실제 활성 feature의 alpha=0.
- inactive feature.
- 적어도 첫 prediction forward와 후속 cached forward.
- `ALL rows` original hook와 readout-only 계산의 차이를 확인한다.

이 확인은 새로운 token-localization 연구가 아니다. **MVP의 offline score가 기존 hook과 같은 연산을 평가하는지 확인하는 구현 테스트**다. BatchTopK encode가 batch-dependent하면 original forward grouping을 유지한다.

새 연구의 런타임 검증 상태가 `not_run`, `stale` 또는 `failed`이면 full sweep을 시작하지 않는다.

#### 7.3 Parity 보고서의 적용 범위와 재사용

`runtime_parity.json`과 `edit_parity.json`의 `passed`는 다음 검증 조건이 현재 실행과 일치할 때만 유효하다.

```text
model weights / remote-code / processor identity
SAE checkpoint 및 encoder 구현 identity
output-head bundle hash / norm 설정
target layer / hook 종류 / readout mapping version
hidden/head dtype / reduction dtype / edit backend
encoder grouping / 수치 연산 설정 / 허용오차
검사한 구현 fingerprint / 검증 코드 revision
검증 sample identity / 실제 검사 항목과 결과
```

수치 연산에 영향을 주는 항목이 바뀌면 과거 `passed`를 사용하지 않고 `stale`로 처리하여 해당 검사를 다시 수행한다. 코드 revision만 바뀐 경우에도 관련 구현이 동일하다는 근거 없이 통과 상태를 승계하지 않는다.

후보 수, rollout 수, 저장 경로 등 수치 연산과 무관한 변경만으로 모든 parity를 다시 실행할 필요는 없다. 검증에 사용한 batch/shape 범위 밖의 새 실행 방식은 대표 입력으로 추가 확인한다.

`score`의 실데이터 대량 실행과 `run` 모두 현재 설정에 대한 head parity 및 edit parity를 요구한다. Alpha=1 identity는 별도 실행 검증이며 head/edit parity를 대체하지 않는다. Synthetic 테스트는 이 실모델 gate를 우회하는 근거가 아니다.

입력 identity가 바뀐 보고서, 필수 검사 항목 누락, 허용오차 변경으로 강제 통과시킨 보고서를 거부하는 테스트를 작성한다.

### 8. 계산량과 저장량을 제한하는 구현

#### 8.1 샘플링을 먼저 고정

전체 281GB급 입력을 매 feature마다 반복해서 순회하지 않는다. 먼저 index로 task/episode/step 목록을 만들고, 선택한 readout을 한 번 추출한다.

권장 시작 설정은 **예산 예시**다. 실제 존재하는 IDs를 확인하여 manifest로 고정한다.

| 단계 | 오프라인 sample 예시 | 목적 |
|---|---|---|
| Synthetic | toy hidden/head/SAE 수십 rows | CPU correctness |
| Tiny real check | 2 tasks × 2 episodes × 8 steps × 7 readouts, 최대 224 rows | mapping·head·edit 검증 |
| Discovery score | 최대 10 tasks × task당 5~10 episodes × episode당 20 steps × 7 readouts | 일차 후보 ranking |
| 확장 | 이미 선택된 discovery 범위 안에서 sample 수 증가 | score 안정성 확인 |

짧은 episode는 실제 존재하는 steps만 사용한다. padding 가짜 step을 만들지 않는다. sampling은 baseline 성공 여부나 feature 개입 결과를 보고 바꾸지 않는다. step 균등 분산 sampling 같은 규칙을 seed와 함께 기록한다.

#### 8.2 연산 최적화의 우선순위

1. reference path부터 작은 입력에서 정확히 구현한다.
2. readout cache와 baseline logits/log-prob를 재사용한다.
3. 실제 active `(row, feature)` pair만 편집한다. 나머지 항목은 점수 0이다.
4. row/feature pair batch 크기를 제한한다. 중간 `[all rows, all features, full vocab]` tensor는 금지한다.
5. 많은 feature의 전체 logits를 디스크에 쓰지 말고 score sufficient statistics만 누적한다.
6. GPU 메모리 예산과 작업 수 상한을 설정하고 실행 전 dry-run에 표시한다.
7. 필요하면 full-vocab projection을 block 단위로 수행한다. 정확한 KL을 위해 global logsumexp/정규화가 유지되어야 한다.
8. algebraic RMSNorm projection 재사용 최적화는 **선택 사항**이다. BF16 cast까지 실제 reference와 같다고 증명하지 못하면 approximate mode로 분리한다. MVP를 이것 때문에 지연하지 않는다.

추가 Top-K 저장 절단은 silent approximation이므로 금지한다. 활성 feature를 일부만 정확 계산한다면 `screened_subset`으로 표시하고 전체 dictionary 평가라고 부르지 않는다.

#### 8.3 Resume와 provenance

계산 결과는 atomic write로 저장하고, resume 전에 다음을 검사한다.

```text
source index/shard identity, checkpoint hash, head bundle hash,
model/code revision, mapping version, sample/split manifest hash,
alpha, score definition, dtype mode, aggregation,
feature universe, implementation version
```

점수 설정이 바뀌면 완료된 artifact를 재사용하지 않는다. 경로명만 같다는 이유로 캐시를 신뢰하지 않는다. 기존 대형 shard hash가 제공되면 재사용하여 불필요한 전체 재스캔을 피하되 검증 수준을 기록한다.

### 9. 비교 실험 설계

#### 9.1 비교할 predictor

이번에는 predictor 한 종류를 구현하는 독립 실험이다. 다른 제안 실험을 같이 실행하지 않는다. 비교군은 후보 선정의 추가 가치를 확인하는 데 필요한 점수들이다.

**주 비교:**

- 기존 `event_aligned`.
- 새 `head_full_vocab_kl_fixed_prefix`.
- 같은 readout sample의 `mean_readout_activation`.
- 같은 readout sample의 `readout_activation_frequency` 또는 `mean_edit_norm`.

**기존 artifact가 유효하면 함께 보고:** `window_mean`, `task_mean`, original `random_alive`.

readout 평균 활성 비교는 중요하다. 새 score의 이득이 head 민감도 때문인지, 단지 행동 readout 위치만 골랐기 때문인지 구분한다.

decoder가 단위 norm이면 `mean_edit_norm`과 활성 평균이 거의 중복될 수 있다. 중복도를 보고하고 이를 독립적인 새로운 predictor라고 과장하지 않는다.

#### 9.2 전체 feature score를 준비

기존 `candidates.jsonl`의 top-5 행만으로 전체 score 상관분석을 하지 않는다. `event_feature_scores.pt` 등의 matrix에서 평가 panel에 필요한 feature의 suite score를 원래 함수와 같은 방식으로 추출한다.

Research loader는 **선택한 predictor에 필요한 artifact만 필수 검증**한다.

| 선택한 방법 | 필수 입력 | 누락 시 처리 |
|---|---|---|
| `event_aligned` | `matrix_raw`, row metadata, canonical coverage | 해당 비교를 실행 불가로 보고 |
| `window_mean` | `matrix_window_mean`, row metadata, event-count 가중치 | 선택되어 있으면 오류 |
| `task_mean` | `matrix_task_mean`, task 식별·중복 제거·timestep 가중치 | 선택되어 있으면 오류 |
| head / readout activation / frequency / edit norm | 같은 sample의 새 score와 provenance | 해당 predictor 생성 또는 입력 필요 |

선택하지 않은 window/task matrix가 없다는 이유로 event/head 주 비교를 막지 않는다. 기존 legacy `matrix`를 `event_aligned`에 사용하는 것은 해당 값이 원 event score라는 provenance가 확인될 때만 허용한다. window/task matrix를 event matrix로 대체하는 fallback은 계속 금지한다. [S5]

점수의 task 범위와 discovery 범위를 검사한다. 2-task head score와 10-task event suite score처럼 범위가 다른 값은 동일 모집단의 직접 비교로 합치지 않는다. 맞는 범위로 재집계할 수 없으면 `legacy_reference`로 구분하거나 해당 비교를 차단한다. 기존 event/window/task 공식 자체의 token/cluster 집계 방식은 재현 baseline으로 유지하고 명시한다.

원래 random-alive는 informed top features를 제외하는 조건부 기준선이다. 새 audit panel의 discovery readout-alive 무작위 표본과 구분한다. 두 표본을 같은 이름으로 보고하지 않는다.

#### 9.3 사전에 feature panel 고정

평가 feature를 event top 몇 개에만 한정하면 비교가 편향된다. MVP의 후보 규칙은 다음처럼 고정한다.

1. 주 비교 predictor 각각의 상위 K개를 선택한다. 최초 K=3이며, 각 score를 내림차순 정렬하고 동점은 `feature_id` 오름차순으로 처리한다.
2. 별도 audit 표본은 **같은 discovery readout sample에서 실제 encode 값이 한 번 이상 양수인 feature 집합**에서 균일하게 비복원 추출한다. 최초 6개, 고정 seed를 사용한다. 기존 sparse Top-K에서 누락됐다는 이유만으로 alive 여부를 결정하지 않는다.
3. 균일 표집 전에 top-K feature를 audit 모집단에서 제외하지 않는다. 겹치면 두 membership을 보존하고 다시 뽑아 채우지 않는다. audit 표본 자체의 크기는 6개이며, top-K와 합친 unique feature 수는 작아질 수 있다.
4. 명시적으로 포함한 기존 event/window/task 후보가 있으면 출처를 남겨 추가한다. Readout에서 inactive인 event 후보도 자동 삭제하지 않는다.

정렬된 audit 모집단의 hash, 표집 알고리즘/version, seed, 실제 draw 순서, top-K tie rule과 각 feature membership을 `candidates.jsonl` 및 plan manifest에 기록한다. Audit 모집단이 요청 수보다 작으면 조용히 표본 수를 줄이지 않고 계획 오류를 보고한다.

이 MVP는 score 기반 상·중·하 층화 표집을 사용하지 않는다. 이를 후속으로 추가하려면 기준 score, 경계, 층별 수, 동점·중복 규칙을 별도 config와 manifest로 사전에 정의해야 한다.

feature ID를 deduplicate하고 **동일 feature는 한 번만 rollout**한다. 모든 후보 출처를 기록한다. Feature cap을 넘으면 임의로 잘라서 어떤 predictor의 top-K를 누락시키지 말고, 실행하지 않은 plan을 출력하여 예산 조정을 요청한다.

두 가지 결과 집합을 구분한다.

1. **Top-K union:** 각 방법의 실제 top-K 선정 효과 비교.
2. **Audit sample:** discovery readout에서 alive였던 집합의 균일 표본에서 score–행동 효과 관계 탐색. 이는 전체 SAE dictionary의 균일 표본이 아니다. 6개는 탐색용 시작 예산이며 강한 상관관계 주장의 표본 수로 간주하지 않는다.

전체 dictionary에서 균일 표집한 것이 아니라면 상관계수의 모집단을 “평가한 panel”로 한정한다. 평가하지 않은 feature의 ΔSR을 0으로 채우지 않는다.

#### 9.4 Predictor와 label의 데이터 누수 방지

다음 두 모드를 명시적으로 지원하라.

**Confirmatory evaluation split은 M4 pilot 전에 고정한다.** M3에서 split 생성 기능을 구현하고, 실제 입력이 준비되면 M4 pilot 전에 원본 초기 상태 ID/hash로 split manifest를 만들고 hash를 기록한다. M4 pilot·설정 선택·후보 진단에는 discovery/validation만 사용하며, 봉인한 evaluation 상태의 개입 label을 사용하지 않는다. M5에서 편리한 상태로 evaluation split을 다시 선택하지 않는다.

기존 실험에서 이미 개입 label을 확인한 상태가 있다면 이력과 겹침을 기록한다. 향후 평가 후보의 label을 이미 사용하여 방법·설정을 선택했다면 해당 결과는 retrospective/pilot로 보고하며 confirmatory라는 이름으로 전환하지 않는다. 이 조건은 SAE 학습 데이터의 기존 중복과 별도로 판정한다.

**Pilot / 기존 결과 탐색 모드**

- 기존 고정 discovery500 artifact를 그대로 재사용할 수 있다.
- 평가 초기 상태가 discovery에 포함되면 `discovery_eval_overlap=true`를 표시한다. 이미 본 개입 label과의 중복은 별도 `evaluation_labels_previously_used` 필드로 기록한다.
- 이 결과는 구현·예비 타당성 확인이다. unseen-state 일반화 결과로 부르지 않는다.

**Confirmatory / ranking-selection 분리 모드**

- 기존 50개 초기 상태/episodes를 task별로 discovery, validation, evaluation으로 미리 나눈다. 예: 35/5/10. 이는 사전 예시이며 실제 IDs와 상태 중복을 확인한다.
- split은 **episode/initial state 단위**다. 같은 episode의 token을 서로 다른 split에 나누지 않는다.
- head score 및 활성 통계는 discovery split으로만 계산한다. alpha·score 종류·aggregation 선택은 validation 이후 고정하고 evaluation label을 보지 않는다.
- event/window/task baseline을 동일 discovery 범위로 비교하려면 기존 activation/event artifact에서 해당 episode만 사용해 점수를 **오프라인 재계산**한다. 이미 있는 event descriptor를 다시 만들거나 이미지/환경 데이터를 재수집하는 작업이 아니다.
- 기존 cluster 구성을 고정 재사용한다면 그 cluster 정의가 전체 데이터에서 온 고정 reference라는 점을 보고한다. selection count·coverage·평균값은 discovery 범위에 맞게 다시 계산한다.
- 이 필터링이 아직 구현되지 않았거나 artifact만으로 불가능하면 기존 점수는 `legacy_all500_reference`로 분리한다. 이를 strict held-out comparator로 표시하지 않는다.
- SAE 자체는 기존 전체 데이터로 학습된 checkpoint를 유지한다. 따라서 confirmatory 결과도 **고정된 SAE/표현 아래 ranking 선정과 개입 평가의 분리**이지, SAE 학습까지 완전히 분리한 일반화 검증은 아니다.

최종 평가의 baseline과 intervention은 같은 evaluation 초기 상태에서 새로 실행한다. 이전 raw 결과의 설정·코드·seed·상태 hash가 일치하지 않으면 재사용하지 않는다.

#### 9.5 명시적 trial index 지원

현재 runner는 `range(num_trials_per_task)`를 사용한다. discovery 밖의 evaluation indices를 지정하려면 최소한의 backward-compatible 확장이 필요하다. [S7]

권장 형태:

```python
# 기존 EnvConfig에 선택적으로 추가할 필드의 예시
trial_indices: list[int] | None = None
```

- `None`이면 기존과 동일하게 `range(num_trials_per_task)`.
- 지정하면 원래 LIBERO initial-state index를 그대로 사용한다.
- 빈 목록, 중복, 음수, 범위 초과를 거부한다.
- 명시 목록이 있을 때 `num_trials_per_task`와 길이가 다르면 오류로 하여 모호성을 없앤다.
- 로그의 `task_episode_idx`는 순회 순번이 아니라 실제 원본 index다.
- `episode_num`은 run-local ID로만 사용한다.
- 전체 초기 상태 배열을 새로 저장할 필요는 없다. 기존과 같은 hash와 실제 index를 기록한다.
- 기존 reproduction YAML의 의미와 legacy fingerprint는 가능한 한 그대로 유지한다. optional 필드가 `None`인 기존 config의 canonicalization 호환성을 테스트한다. schema를 바꿀 필요가 있으면 legacy/new reader를 명시적으로 분리한다.

### 10. 새 rollout은 작은 research runner로 실행

#### 10.1 조건

- raw baseline.
- 구현 확인용 alpha=1 identity: 소수 feature/상태로 확인.
- 각 unique feature의 alpha=0 intervention.

서로 다른 predictor가 같은 feature를 선택하면 결과를 공유한다. 모든 feature를 동시에 제거하는 것이 아니라 **feature 하나씩 독립 개입**한다.

MVP에서는 기존 all-row L31 hook를 유지한다. offline readout score를 계산했다는 이유로 실제 rollout의 개입 위치를 LAST-only로 변경하지 않는다. 그렇게 하면 intervention 자체가 바뀐다.

#### 10.2 최소 로그

필수:

```text
run/protocol ID, model/code/SAE/head hashes,
suite, task_id, original task_episode_idx, initial_state_sha256,
seed, feature_id, alpha, hook_start_step,
success, num_actions, caught_exception,
hook 실행 여부와 target activation 제거 검증
```

alpha=1 검증에서는 기존 `save_actions=true`를 사용하여 행동열 일치도 확인한다. 일반 평가에서는 선택적으로 끈다. 영상·기본 궤적도 이번 실험의 필수 입력은 아니다.

새 object/contact/reward/subgoal 로그는 추가하지 않는다. Head score용 메타데이터와 모델 내부 진단값은 환경 의미정보 확장 수집과 구별하되, 대량 per-token 로그 저장은 피한다.

#### 10.3 Pairing과 오류 처리

raw/intervention을 다음과 같은 키로 대응시킨다.

```text
suite + task_id + original task_episode_idx + initial_state_sha256
+ 동일한 seed/rollout protocol
```

global episode_num만으로 join하지 않는다. 환경이 추가 난수를 사용하는지도 audit한다. 상태 hash가 같다는 사실만으로 RNG 상태까지 같다고 주장하지 않는다. 추가 per-episode seeding이 필요하면 연구용 설정에서만 task/원본 trial ID로부터 조건 독립적으로 유도하고, 모든 조건에 동일 적용하며 legacy 기본값을 보존한다. baseline 성공 episode만 골라 primary analysis를 하지 않는다. 성공→실패, 실패→성공을 모두 보존한다.

예외·중단·누락은 task failure와 구별한다. 기존 CLI처럼 runtime 오류가 있는 condition은 유효한 completed result로 인정하지 않는다. 연구용 분석기는 잘못된 결과나 mismatched pair를 조용히 버리지 말고 오류/제외 사유를 보고한다.

#### 10.4 예산 예시와 승인

```text
총 rollout 수 = (raw 1개 + unique intervention features 수 + 별도 no-op 조건 수)
               × 평가 task 수 × task당 평가 초기 상태 수
```

예시:

- 작은 pilot: 6 features + raw + no-op, 2 tasks × 5 trials → 80 rollouts.
- 본 비교의 예시: 24 features + raw, 10 tasks × 10 trials → 2,500 rollouts.
- 위 본 비교 외 별도 no-op 1개를 같은 전체 범위에 수행하면 100회가 더 든다. 보통 no-op은 사전 소규모 검증으로 제한한다.

위 숫자는 실행 지시나 충분한 검정력의 보장이 아니다. 실제 deduplicated panel로 계획을 다시 산정한다. GPU 시간은 고정값을 단정하지 말고 pilot의 측정 시간으로 추산한다.

`plan`은 GPU를 쓰지 않고 조건 수·예상 호출 수·입력 상태·총 예산을 출력해야 한다. `run`은 실행 플래그, 승인된 plan hash, rollout 상한을 받아야 한다. 사용자 승인 없이 실행하지 않는다.

### 11. 최종 분석과 성공 기준

#### 11.1 Label

feature i의 primary label:

```text
drop_i = mean_task(SR_raw,task - SR_edit_i,task)
```

값이 클수록 성공률을 크게 떨어뜨린다. 논문식 `delta_sr = SR_edit - SR_raw`도 필요하면 함께 저장하되 부호를 명시한다. 음수 drop은 개선을 뜻하며 0으로 자르지 않는다.

기존 paired 성공 로그로 다음 보조 통계를 함께 계산한다. 추가 rollout이나 환경 state 기록은 필요하지 않다.

```text
n_success_to_failure
n_failure_to_success
n_success_to_success
n_failure_to_failure
outcome_switch_rate = mean_task((n_success_to_failure + n_failure_to_success) / valid_pairs_task)
```

KL은 출력 변화의 크기, primary drop은 성능 변화의 방향을 포함한다. 두 방향의 outcome 변화가 상쇄되는 경우를 보조 통계로 보여준다. 결과를 본 뒤 outcome switch를 새로운 primary label로 바꾸지는 않는다.

#### 11.2 주 분석

1. **예측 관계:** 평가 panel에서 offline score와 drop의 Spearman correlation. 유효 feature 수와 score/label의 tie 비율도 표시한다.
2. **후보 선정 효율:** 같은 K에서 각 predictor가 선택한 feature들의 평균 drop. 평가된 동일한 universe와 top-K coverage를 표시한다.
3. **추가 정보:** readout 평균 활성/발화 빈도/편집 norm보다 head score가 더 유용한지 비교한다.
4. **Event-SAE 진단:** event score는 높지만 head score는 낮은 feature, 반대 feature의 실제 drop과 불확실성을 비교한다. 사례 선택 규칙은 label 확인 전에 고정한다.

오프라인 점수가 상관을 보이지 않는 결과도 유효하다. “head는 민감하지만 rollout에서 회복되는 경우” 등의 해석은 가능성으로만 제시하고, 관측되지 않은 세부 행동 이유를 발명하지 않는다.

#### 11.3 불확실성

- 각 feature의 drop과 predictor 간 top-K 평균 drop 차이는 **같은 초기 상태를 함께 재표본추출하는 paired bootstrap**으로 CI를 구한다.
- 고정된 10 tasks에 대한 주장이라면 task 내부 초기 상태 resampling을 기본으로 한다. task 모집단으로 일반화하려는 분석은 task resampling을 추가한 별도 estimand로 표시한다.
- correlation에 대한 label 불확실성은 같은 bootstrap draw를 전체 feature에 공유하여 반영한다.
- 이는 평가한 고정 panel에 조건부인 불확실성이다. feature panel 표집 자체의 불확실성과 같다고 주장하지 않는다.
- feature/task가 중복되는 결과를 독립 sample로 부풀리지 않는다. token 수를 n으로 사용하지 않는다.
- 모든 score/label이 같으면 correlation을 0으로 꾸미지 말고 undefined로 보고한다.
- 작은 pilot 결과로 유의성을 단정하지 않는다. feature 수와 paired rollout 수를 같이 표시한다.
- 모든 관측 pair에서 차이가 같아서 bootstrap CI 폭이 0이면 `degenerate_bootstrap`으로 표시한다. 관측된 상태에서 차이가 없다는 결과이며 영향 부재의 증명으로 해석하지 않는다.

MVP에서는 학습식 predictor를 만들지 않는다. 후속으로 회귀모델을 추가한다면 feature 단위 train/test 분리와 validation을 별도로 설계해야 하며, 같은 feature의 task 행을 무작위 분할하여 성능을 과장하면 안 된다.

#### 11.4 계산 비용 효율

“시간을 줄였다”는 주장은 다음을 모두 기록한 경우에만 사용한다.

```text
head bundle 준비 시간
readout 추출 시간·I/O
SAE encoding 시간
score 계산 시간 / GPU peak memory
후보별 rollout 시간·총 횟수
캐시 사용 여부
```

전체 dictionary의 진짜 중요한 feature를 전부 측정하지 않았으므로 `global recall@K` 또는 `oracle global ranking`을 보고하지 않는다. 평가 panel 내 top-K 효과와 실제 사용 예산을 보고한다.

### 12. 권장 신규 모듈과 API

다음은 제안 경로다. 기존 비슷한 모듈이 있다면 중복을 만들지 않고 조정해도 된다. 거대한 범용 framework보다 이 실험에 필요한 작은 모듈을 선호한다.

```text
event_sae/research/output_head/
    __init__.py
    config.py          # strict config, budgets, split/sample schema
    readouts.py        # index audit, readout mapping, selective cache
    head.py            # local head export/load, norm/head forward, parity
    sensitivity.py     # reference edit, score computation, aggregation
    candidates.py      # original ranking adapter, panel, dedup, plan
    results.py         # paired label assembly, statistics, reports
    provenance.py      # hash, atomic write, compatibility checks

scripts/openvla/output_head_sensitivity.py
configs/research/openvla/output_head_sensitivity.yaml
docs/output_head_sensitivity.md

tests/test_output_head_readouts.py
tests/test_output_head_sensitivity.py
tests/test_output_head_candidates.py
tests/test_output_head_results.py
tests/test_output_head_cli.py
```

핵심 API 예시:

```python
def audit_inputs(config) -> dict: ...
def build_split_manifest(source_episode_manifest, split_spec) -> dict: ...
def build_readout_manifest(index_path, sample_spec, generation_spec) -> dict: ...
def prepare_readout_cache(manifest, sae, output_dir) -> dict: ...
def export_local_output_head(local_model_or_snapshot, output_dir) -> dict: ...
def edit_feature_reference(hidden, sae, feature_id, alpha, hidden_dtype): ...
def score_features(readout_cache, head_bundle, sae, score_config) -> dict: ...
def build_evaluation_plan(scores, comparison_artifacts, eval_manifest, budget) -> dict: ...
def assemble_paired_effects(raw_result, feature_results, protocol) -> dict: ...
def analyze_prediction(scores, paired_effects, analysis_config) -> dict: ...
```

모든 API에 type hint, 입력 검증, 명확한 예외를 넣는다. 디바이스를 전역 상수로 가정하지 않는다. CPU 테스트에서 TensorFlow/LIBERO/실제 Hugging Face 모델을 import하지 않도록 lazy import를 사용한다.

### 13. 설정 예시

아래 YAML은 **구현할 schema 예시**다. 기존 config 파일의 경로·내용을 덮어쓰지 않는다. `null`인 필수 경로는 audit에서 누락으로 보고하고, 무거운 실행 전에 반드시 채워야 한다.

```yaml
schema_version: output_head_sensitivity_v1
experiment_name: openvla_l31_fixed_prefix_head_sensitivity

scope:
  model_family: openvla
  suite: libero_spatial
  layer_idx: 31
  capture_target: post_mlp_residual
  require_final_decoder_layer: true
  retrain_sae: false
  recollect_discovery: false
  collect_extended_env_state: false

inputs:
  dense_dir: null
  source_episode_manifest: null  # task / 원본 trial index / initial-state hash
  sae_checkpoint: null
  event_scores_path: null
  existing_candidates_path: null
  existing_result_dirs: []
  local_model_snapshot: null
  output_head_bundle: null

model:
  checkpoint: openvla/openvla-7b-finetuned-libero-spatial
  revision: 962318cec55ac10993ff0f5f43eda9a270b4c873
  code_revision: 47a0ec7fc4ec123775a391911046cf33cf9ed83f
  unnorm_key: libero_spatial
  expected_live_hidden_dtype: bfloat16
  allow_download: false

splits:
  split_seed: 2026
  # 예시다. 실제 기존 상태 수와 중복을 먼저 확인하고 명시적으로 고정한다.
  discovery_per_task: 35
  validation_per_task: 5
  evaluation_per_task: 10
  require_initial_state_disjoint: true
  evaluation_labels_previously_used: null  # unknown; false로 임의 가정하지 않는다.

sampling:
  mode: pilot
  split_manifest: null
  sample_seed: 2026
  task_ids: [0, 1]
  max_episodes_per_task: 2
  max_steps_per_episode: 8
  step_selection: evenly_spaced
  require_complete_action_query: true
  # Confirmatory 모드에는 M4 이전에 고정한 실제 manifest가 필수다.
  require_frozen_eval_before_pilot: true

scoring:
  alpha: 0.0
  primary_metric: head_full_vocab_kl_fixed_prefix
  logits_stage: raw_head
  edit_backend: reference
  arithmetic_mode: runtime_matched
  reduction_dtype: float32
  aggregation: equal_task_episode_step_dimension
  include_inactive_as_zero: true
  feature_universe: all_dictionary_features
  pair_batch_size: 32
  max_scored_pairs: 20000
  full_vocab: true
  save_per_row_logits: false

selection:
  methods:
    - event_aligned
    - head_full_vocab_kl_fixed_prefix
    - mean_readout_activation
    - readout_activation_frequency
  top_k: 3
  random_audit_features: 6
  random_seed: 2026
  topk_tie_break: feature_id_ascending
  audit_population: discovery_readout_alive
  audit_sampling: uniform_without_replacement
  audit_overlap_policy: keep_memberships_no_redraw
  max_unique_features: 24
  require_all_method_topk: true

rollout:
  execute: false
  env_seed: 0
  alpha: 0.0
  hook_start_step: 0
  eval_manifest: null
  base_eval_config: configs/reproduction/openvla/libero_spatial_intervention_layer31.yaml
  max_total_rollouts: 100
  require_clean_git: true
  require_head_parity: true
  require_edit_parity: true
  require_current_parity_fingerprint: true
  require_identity_check: true
  save_actions_for_identity: true
  save_video: false
  save_trajectory_records: false
  collect_activations: false

analysis:
  primary_label: success_rate_drop
  bootstrap_replicates: 2000
  bootstrap_seed: 2026
  confidence_level: 0.95
  allow_missing_pairs: false
  require_declared_selection_eval_overlap: true
  secondary_labels: [outcome_switch_rate]
  report_outcome_transition_counts: true

output:
  root_dir: null
  overwrite: false
  resume: true
```

이 예시에서는 계획이 `max_total_rollouts`를 초과하면 실행하지 않고 필요한 예산을 보고한다. 후보 수를 임의로 줄여 실험 설계를 변경하지 않는다.

`split`은 `inputs.source_episode_manifest`와 `splits`의 명시된 규칙을 사용하여 output root에 split manifest를 생성한다. 실제 task별 상태 수가 합계와 다르거나 상태 hash provenance가 확인되지 않으면 confirmatory split을 생성하지 않는다. 생성한 파일을 `sampling.split_manifest`에 연결하고 그 evaluation 목록에서 `rollout.eval_manifest`를 만든다. Pilot만 수행하는 경우에는 confirmatory split이 필수가 아니며, 중복·기존 label 사용 상태를 정직하게 기록한다.

### 14. 목표 CLI와 실행 설명서

하나의 CLI에 subcommand를 두어도 된다. 아래 명령은 **개발 후 제공해야 할 사용 예시**이며 현재 저장소에서 이미 실행된 명령이 아니다.

먼저 `configs/research/openvla/output_head_sensitivity.yaml`에 실제 local 입력 경로와 별도 output root를 설정한다. 경로가 없다고 기존 runbook의 대용량 downloader를 자동 호출하지 않는다.

```bash
# 1. 입력 metadata, 코드 차이, 필요한 artifact 확인
python scripts/openvla/output_head_sensitivity.py audit \
  --config configs/research/openvla/output_head_sensitivity.yaml

# 2. synthetic CPU tests: 실제 모델·simulator 불필요
python -m pytest -q tests/test_output_head_*.py

# Confirmatory를 계획하면 M4 전에 split manifest를 먼저 만든다.
# 실제 source episode/index 정보와 split 규칙을 config에 채운 뒤 실행한다.
python scripts/openvla/output_head_sensitivity.py split \
  --config configs/research/openvla/output_head_sensitivity.yaml

# 3. 사용자 승인 후: 기존 local checkpoint에서 head만 파생 저장
python scripts/openvla/output_head_sensitivity.py export-head \
  --config configs/research/openvla/output_head_sensitivity.yaml

# 4. 사용자 승인 후: 기존 데이터의 선택 readout cache 생성
python scripts/openvla/output_head_sensitivity.py prepare \
  --config configs/research/openvla/output_head_sensitivity.yaml

# 5. 사용자 승인 후: 작은 실제 forward로 runtime consistency 확인
python scripts/openvla/output_head_sensitivity.py validate \
  --config configs/research/openvla/output_head_sensitivity.yaml \
  --allow-model-execution

# 6. offline score 계산. 사전 pair/메모리 예산을 넘으면 중단
python scripts/openvla/output_head_sensitivity.py score \
  --config configs/research/openvla/output_head_sensitivity.yaml

# 7. 여기까지는 새 intervention rollout을 실행하지 않는다.
python scripts/openvla/output_head_sensitivity.py plan \
  --config configs/research/openvla/output_head_sensitivity.yaml

# 8. 사용자 승인 후에만 실행. plan이 출력한 실제 hash를 사용한다.
python scripts/openvla/output_head_sensitivity.py run \
  --config configs/research/openvla/output_head_sensitivity.yaml \
  --approved-plan-hash "$APPROVED_PLAN_HASH" \
  --expected-code-revision "$(git rev-parse HEAD)" \
  --execute

# 9. 유효한 paired 결과만 분석
python scripts/openvla/output_head_sensitivity.py analyze \
  --config configs/research/openvla/output_head_sensitivity.yaml
```

`prepare/score`는 오프라인이지만 실데이터와 GPU 계산을 사용할 수 있으므로 최초 Codex 작업에서 자동 실행하지 않는다. `--help`, audit, split, plan은 모델 다운로드나 CUDA 초기화 없이 동작해야 한다. `split`은 기존 episode metadata를 사용하여 로컬 연구 manifest만 생성하며, 기존 데이터 파일은 수정하지 않는다.

`$APPROVED_PLAN_HASH`가 비어 있거나 현재 계획과 다르면 `run`을 거부한다. 개발 중인 새 코드를 실행할 때 과거 baseline commit을 `expected-code-revision`에 넣지 않는다. 사용자 정책에 따라 검토·commit한 새 revision으로 실행하며, commit/push를 임의로 수행하지 않는다.

### 15. 필수 테스트

#### 15.1 실제 모델 없이 수행할 CPU 테스트

| 테스트 | 통과 조건 |
|---|---|
| tiny RMSNorm + lm_head 기준 forward | 직접 모듈 계산과 일치 |
| alpha=1 | edit identity, KL=0, flip=0 |
| inactive feature | score=0, 분모에는 포함 |
| decoder bias 존재/부재 | residual-preserving 차분 연산이 정확 |
| decoder 방향 orientation | sparse edit와 실제 decode 차분이 tolerance 내 일치 |
| BF16/FP32 경로 | CPU 지원 시 dtype별 reference 검증, 미지원은 명시적 skip |
| 배치 분할 | row별 독립 encoder fixture에서 batch 변경으로 점수/선정이 변하지 않음 |
| batch-dependent encoder fixture | readout-only encode를 잘못 허용하지 않음 |
| forward 매핑 | multi-row prefill + 1-row decode에서 올바른 7 readout |
| variable prompt length | 고정 token offset 없이 동일 매핑 |
| malformed mapping | 누락·중복·8/6개 forward·unknown batch·잘못된 row range를 거부 |
| merged IDs | task/원본 episode가 다른 같은 local ID를 섞지 않음 |
| Top-K truncation | sparse 누락 feature를 무조건 dead로 판정하지 않음 |
| full vocab vs action subset | 둘이 다른 argmax를 주는 toy case를 올바르게 구분 |
| action decoding | padded/effective vocab, bin center 수, clip 경계 정확 |
| aggregation | 길이가 다른 episode/task라도 명시한 동일 가중 결과 |
| inactive 최적화 | dense reference와 희소 pair 최적화 점수가 동일 |
| candidate dedup | 여러 방법의 같은 feature를 한 번만 실행 계획 |
| audit 표집 | 고정 모집단/seed의 재현성, top-K와 겹쳐도 재추출하지 않음, score 동점은 ID 순서 |
| 선택적 비교군 | 선택하지 않은 window/task 누락은 주 비교를 막지 않음; 선택한 누락 방법은 명시적 오류 |
| candidate cap | 어느 방법의 top-K를 silent truncation하지 않음 |
| split | 같은 episode/초기 상태가 금지된 split 양쪽에 있으면 오류 |
| evaluation 사전 고정 | Confirmatory pilot의 eval label 사용·미고정 split을 거부하고 기존 label 사용 이력 기록 |
| label assembly | 같은 hash끼리만 pairing; drop 부호·개선 사례 보존 |
| 통계 | 상수 vector는 undefined, missing label은 0이 아님 |
| trial_indices | None의 legacy 동작, 명시 원본 index, 잘못된 list 오류 |
| provenance/resume | 다른 SAE/head/alpha/mapping/sample이면 캐시 재사용 거부 |
| parity 적용 범위 | SAE/head/dtype/edit/mapping/관련 구현이 다른 passed 보고서 재사용 거부 |
| CLI | 별도 Python process의 --help/audit/split/plan에서 실제 model·LIBERO·TensorFlow import 없음 |
| synthetic end-to-end | tiny artifact → scores → plan → synthetic results → 분석 완료 |

테스트에서 임의 feature ID와 synthetic 결과를 만들 수 있지만 output schema에 `synthetic=true`를 넣고, 실제 실험 output 경로와 분리한다.

#### 15.2 승인 후 실제 환경 테스트

- local head의 baseline logits parity.
- reference와 original hook의 edited readout logits parity.
- alpha=1에서 raw action sequence/성공 결과 일치.
- alpha=0에서 target latent가 실제 제거되는지 확인.
- 동일 초기 상태 hash 사용.
- 작은 pilot의 모든 condition이 예외 없이 완료.
- score throughput, peak memory, rollout wall time 측정.

실험 가설이 성립하는지와 구현 테스트 통과는 다르다. feature 제거에서 성공률이 감소하지 않는다고 구현 실패로 판정하지 않는다.

### 16. 산출물과 schema

권장 output 구조:

```text
<research_output>/
  audit.json
  source_inventory.json
  split_manifest.json
  sample_manifest.json
  head/
    output_head.safetensors
    output_head_manifest.json
  readouts/
    manifest.json
    shard_*.pt
  validation/
    runtime_parity.json
    edit_parity.json
  scores/
    feature_scores.csv
    feature_task_scores.csv
    score_manifest.json
    score_diagnostics.json
  plan/
    candidates.jsonl
    eval_manifest.json
    rollout_plan.json
  runs/
    raw/
    identity/
    feature-<id>/
  analysis/
    paired_effects.csv
    predictor_comparison.csv
    uncertainty.json
    figures/
    report.md
```

`feature_scores.csv` 최소 열:

```text
feature_id, alpha, head_full_vocab_kl_fixed_prefix,
full_vocab_argmax_flip_rate, decoded_bin_change_rate,
mean_readout_activation, readout_activation_frequency, mean_edit_norm,
num_readouts, num_active_readouts, sampled_task_count,
score_status
```

`paired_effects.csv` 최소 열:

```text
feature_id, alpha, num_tasks, num_valid_pairs,
sr_raw, sr_edit, drop, delta_sr,
drop_ci_low, drop_ci_high,
selection_eval_overlap, evaluation_labels_previously_used, protocol_id,
n_success_to_failure, n_failure_to_success,
n_success_to_success, n_failure_to_failure, outcome_switch_rate
```

`predictor_comparison.csv` 최소 열:

```text
predictor, evaluated_feature_count, spearman,
correlation_ci_low, correlation_ci_high,
top_k, topk_coverage, topk_mean_drop, topk_mean_drop_ci,
feature_universe, population_scope
```

임의 데이터를 채우지 않는다. 계산할 수 없는 값은 `null/NaN`과 사유를 별도로 기록한다. graph는 score–drop scatter, 같은 K의 평균 drop 비교, 실제 계산비용 비교 정도로 제한한다. 과도한 dashboard는 필요 없다.

최종 `report.md`에는 다음이 포함되어야 한다.

- 무엇이 실제 측정됐고 무엇이 미실행인지.
- 기존 artifact의 재사용 범위와 split 중복.
- head/SAE/generation 수치 검증 결과.
- feature별 결과와 predictor 비교, CI.
- 예산·캐시·하드웨어·실제 시간.
- 좋은 결과뿐 아니라 undefined/negative/missing 결과.
- 국소 출력 변화와 closed-loop 성공의 차이, 고정 SAE·단일 모델 범위.

### 17. 단계별 완료 기준

| 단계 | 개발·실험 내용 | 완료 기준 | 최초 Codex 작업 |
|---|---|---|---|
| M0 | 코드/입력 audit, 기존 동작 확인 | 실제 함수·형식과 차단 요인을 보고 | 수행 |
| M1 | readout mapping/cache, head bundle loader/exporter | synthetic mapping·head tests 통과 | 구현·CPU 테스트 |
| M2 | reference sensitivity, 집계, budget/resume | toy reference와 일치, inactive/alpha1 정상 | 구현·CPU 테스트 |
| M3 | ranking adapter, split/후보 plan, paired 분석, CLI·문서 | synthetic end-to-end, legacy 회귀 테스트; confirmatory용 split 생성 경로 준비 | 구현·CPU 테스트 |
| M4 | 작은 실모델 parity, 실제 offline score, 작은 rollout pilot | 현재 조건의 parity 통과와 실측 자원 보고; confirmatory이면 pilot 전에 eval split 고정, pilot은 discovery/validation에서만 | 별도 승인 후 |
| M5 | M4 전 고정한 eval split과 evaluation label을 보기 전 확정한 panel로 본 비교 | 예산 내 paired 결과/CI/한계 보고; eval split 재선택 금지 | 별도 승인 후 |

M3까지 끝났더라도 M4를 하지 않았으면 “실제 모델 검증 완료”라고 쓰지 않는다. M4가 실패하면 M5를 돌리지 않는다. 실패 원인을 수정하되 새 모델·새 SAE·확장 데이터 수집으로 몰래 연구 범위를 바꾸지 않는다.

### 18. Codex가 마지막에 보고할 내용

다음 형식으로 결과를 보고하라.

```text
1. 구현한 범위 / 남긴 범위
2. 수정·추가한 파일과 각 역할
3. 기존 재현 동작에 미치는 영향
4. 실제 실행한 테스트 명령·pass/fail/skip
5. 실데이터·실모델 검증 여부와 차단 요인
6. 다음 단계에 필요한 local 입력 경로
7. 승인 후 실행할 정확한 명령
8. 예상 조건 수·rollout 수: plan으로 계산한 값만 사용
```

“아마 작동한다”는 대신 검증된 부분과 미검증 부분을 구분한다. 이름만 있는 함수, `pass`, 고정 난수 점수, 가짜 성공률로 완료 처리하지 않는다.

### 19. 확인한 소스와 검증 범위

아래 URL은 문서 작성 시 확인한 코드의 출처다. Codex에서는 우선 실제 checkout과 설치된 runtime을 확인하라. 새로운 module/CLI/통계 설계는 이 문서의 **제안**이며 기존 논문의 구현이라고 주장하지 않는다.

- [S1] 저장소 runbook / 기존 데이터·산출물·평가 단계:
  `https://github.com/jiyeon-yoon/Event-SAE-Baseline/blob/56e9f012aa88532f9880259b371d9300a8013b9e/docs/reproduce_libero_spatial_500.md`
- [S2] model loading 및 고정 평가 설정:
  `https://github.com/jiyeon-yoon/Event-SAE-Baseline/blob/56e9f012aa88532f9880259b371d9300a8013b9e/event_sae/openvla/eval/model.py`
  `https://github.com/jiyeon-yoon/Event-SAE-Baseline/blob/56e9f012aa88532f9880259b371d9300a8013b9e/configs/reproduction/openvla/libero_spatial_intervention_layer31.yaml`
- [S3] 기존 residual-preserving hook 및 CLI:
  `https://github.com/jiyeon-yoon/Event-SAE-Baseline/blob/56e9f012aa88532f9880259b371d9300a8013b9e/event_sae/openvla/intervene.py`
  `https://github.com/jiyeon-yoon/Event-SAE-Baseline/blob/56e9f012aa88532f9880259b371d9300a8013b9e/scripts/openvla/intervene.py`
- [S4] dense/sparse 수집·변환 형식:
  `https://github.com/jiyeon-yoon/Event-SAE-Baseline/blob/56e9f012aa88532f9880259b371d9300a8013b9e/event_sae/openvla/activations.py`
  `https://github.com/jiyeon-yoon/Event-SAE-Baseline/blob/56e9f012aa88532f9880259b371d9300a8013b9e/scripts/extract_topk.py`
- [S5] 기존 ranking 정의와 legacy fallback:
  `https://github.com/jiyeon-yoon/Event-SAE-Baseline/blob/56e9f012aa88532f9880259b371d9300a8013b9e/event_sae/scoring/rankings.py`
- [S6] 기존 full sweep의 고정 조건:
  `https://github.com/jiyeon-yoon/Event-SAE-Baseline/blob/56e9f012aa88532f9880259b371d9300a8013b9e/scripts/openvla/run_intervention_sweep.py`
- [S7] runner의 초기 상태/성공 로그와 config 및 CPU import 확인 대상:
  `https://github.com/jiyeon-yoon/Event-SAE-Baseline/blob/56e9f012aa88532f9880259b371d9300a8013b9e/event_sae/openvla/eval/runner.py`
  `https://github.com/jiyeon-yoon/Event-SAE-Baseline/blob/56e9f012aa88532f9880259b371d9300a8013b9e/event_sae/openvla/eval/config.py`
  `https://github.com/jiyeon-yoon/Event-SAE-Baseline/blob/56e9f012aa88532f9880259b371d9300a8013b9e/event_sae/openvla/eval/__init__.py`
- [S8] 환경 YAML 및 해당 Transformers 버전의 Llama 출력 경로:
  `https://github.com/jiyeon-yoon/Event-SAE-Baseline/blob/56e9f012aa88532f9880259b371d9300a8013b9e/environment-openvla.yml`
  `https://github.com/huggingface/transformers/blob/v4.48.1/src/transformers/models/llama/modeling_llama.py`
- [S9] OpenVLA 공식 upstream의 generation/action decoding 참고 소스. **고정 HF remote code와 동일하다고 검증한 것은 아님**:
  `https://github.com/openvla/openvla/blob/main/prismatic/extern/hf/modeling_prismatic.py`

v1.0 작성 시 원격 소스를 검토했고, v1.1에서는 실제 `Event-SAE-Head` checkout의 기반 commit과 계획서의 구현·실험 일관성을 재검토했다. 실제 500-episode 입력, 학습된 SAE tensor, pinned OpenVLA runtime을 실행한 검증은 별도 단계로 남아 있다.
