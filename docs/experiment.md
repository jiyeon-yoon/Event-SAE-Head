# 실험 3 — 논문 규모 결과: 출력층 민감도 예측

현재 상태: **10-task·3,600-rollout 후속 실험 + 2026-09-27 저장 결과 재분석 완료**.
여기서 “논문 규모”는 수행 규모를 뜻하며, 출판 가능성이나 확증 실험을 보장하는 표현은 아니다.
실행·다운로드·복구 명령은 [runbook.md](runbook.md)만 보면 된다.

## 0. 한 줄 요약

**고정된 OpenVLA·SAE에서 Head-KL은 Event score와 평균 활성값보다 제거 시 성공률 하락이
큰 feature를 상위에 골랐고, 변경량과 밀접하지만 동일하지 않은 순위를 보였다.
다만 의미 있는 행동 개념을 발견했거나 로봇 성능을 개선했다는 결과는 아니다.**

## 1. 무엇을 돌렸나

연구 질문은 **“어떤 feature를 끄면 로봇의 수행이 크게 나빠질지, 전체 rollout을 하기
전에 출력 변화로 예측할 수 있는가?”**다. 로봇 자체가 아니라 **feature 선별 기준**을 평가했다.

| 항목 | 완료한 실험 |
|---|---|
| 모델·SAE | OpenVLA LIBERO-Spatial, Layer 31의 기존 SAE; 둘 다 고정 |
| 원본 입력 | 10 task × 50 = 500개 기존 episode, task-pair 저장소 5개 병합 |
| 분리 | 점수 계산용 250 / 검증용 50 / 평가용 200 episode |
| 점수 | 14,000 readout에서 32,768개 feature 점수화 |
| 비교 | Event score / Head-KL / 평균 활성 크기 / 활성 빈도 |
| 후보 | 각 방법 Top-3의 합집합 10개 + 무작위 audit 6개 = 16개 feature |
| 실제 평가 | 단일 feature 제거 16조건 + raw + identity = 18조건 × 200회 = 3,600회 |
| 실행 시간 | 조건별 기록 시간 합계 40.877시간; 다운로드·준비·대기·요금 산정 시간과 다름 |

1. 저장된 내부 상태에서 feature 하나의 기여만 빼고, 최종 norm/head를 거친
   **전체 vocabulary 확률분포의 KL(original || edited)**을 계산했다.
2. 기존 생성 문맥을 고정한 점수를 모아 feature 순위를 정했다. 이 단계도 내부 개입이지만,
   새로운 관측·행동을 연속 생성하는 시뮬레이션은 아니다.
3. 후보와 평가 계획을 동결한 뒤, 실제 closed-loop rollout에서 feature를 각각 제거했다.
4. 동일한 초기상태·seed의 원본 실행과 비교해 성공률 하락을 측정했다.

32,768은 SAE의 사전 크기이지, 알려진 행동 능력이나 의미 있는 개념의 개수가 아니다.
전체를 점수화했지만 실제 제거 평가는 **16개 feature**에 한정된다.
OpenVLA/SAE 재학습, 새 모델 도입, 원본 activation 재수집은 하지 않았다.

Head와 Event는 동일한 250개 discovery episode를 사용했다. 다만 Head는 episode당
8개 step의 출력 readout을, Event는 기존 event/window용 시계열·token 집계를 사용한다.
동일 episode 범위이지 완전히 동일한 token 집합·인코딩 집계는 아니다.

## 2. 결과

### 2.1 원래 계획한 비교

원본 성공률은 **80.0%**다. 성공률 하락 = 원본 성공률 − 제거 후 성공률이며,
음수는 제거 후 성능이 개선됐다는 뜻이다.

| 선정 방법 | Top-3 평균 성공률 하락 | Top-K 합집합 내 Spearman |
|---|---:|---:|
| **Head-KL** | **52.50%p** | **0.648** |
| Event score | 23.67%p | 0.212 |
| 평균 활성 크기 | 1.50%p | 0.200 |
| 활성 빈도 | −5.83%p | 0.082 |

- Top-3는 세 feature를 **동시에 제거한 것이 아니라 각각 제거한 결과의 평균**이다.
- Head-KL − Event 차이: **28.83%p**, paired bootstrap 95% CI **[26.50, 31.17]%p**.
- Spearman은 네 방법 Top-3 합집합 **10개 feature 안에서** 계산한 보조 지표다.
- task 2–9 subgroup에서도 Head-KL **50.42%p**, Event **21.88%p**로 같은 방향이었다.
- Head-KL Top-3: `24830, 18471, 7024`; Event Top-3: `30729, 24360, 7024`.

### 2.2 2026-09-27 사후 재분석: 큰 신호를 고르는 것뿐인가?

새 rollout 없이 보관된 `mean_edit_norm`과 기존 성공률을 연결했다.
변경량은 **feature 제거 전후 내부 벡터 차이의 L2 크기 평균**이다.
둘 다 0점인 비활성 feature 26,759개를 제외하고, discovery readout에서 활성화된
**6,009개**를 주 분석 대상으로 삼았다. 이 수는 전체 원수집의 alive feature 수가 아니다.

| Head-KL ↔ 평균 변경량 비교 | 결과 |
|---|---:|
| 전체 활성 feature 순위 상관(Spearman) | **0.882316** |
| 상위 3개 중 공통 | **1/3** |
| 상위 10개 중 공통 | **8/10** |
| 상위 100개 중 공통 | **96/100** |

| Feature | Head-KL 순위 | 변경량 순위 | 실제 성공률 하락 |
|---|---:|---:|---:|
| 24830 | 1 | 19 | 80.0%p |
| 18471 | 2 | 2 | 0.0%p |
| 7024 | 3 | 31 | 77.5%p |
| 13724 | 4 | 1 | 3.5%p |
| 25653 | 6 | 3 | 1.0%p |

새 비교 기준의 전체 Top-3가 우연히 모두 기존 16개에 포함되어 추가 실행 없이 평가 가능했다.

| 사후 비교 기준 | 전체 Top-3 | 평균 성공률 하락 |
|---|---|---:|
| 평균 변경량 | 13724, 18471, 25653 | **1.50%p** |
| 전체-vocabulary 최상위 토큰 변경률 | 24830, 18471, 13724 | **27.83%p** |

평균 변경량과 평균 활성값의 순위 상관은 **0.999999**로, 이 설정에서 두 baseline은 사실상
중복된다. 토큰 변경률은 물리적 action 변화량이나 action-token 전용 지표가 아니다.
성공률 하락과의 상관도 모집단에 의존한다: 기존 Top-K 합집합 10개에서는
Head-KL **0.648**, 변경량 **0.200**; 평가된 16개 전체에서는 **0.385**, **0.188**이다.

수치 출처: 공개본의 [요약](https://huggingface.co/datasets/jiyeony/event-sae-head-full10-results/blob/69c33d6a943c253dc97cf9a3968095276f7e9540/reports/summary.json),
[전체 보고서](https://huggingface.co/datasets/jiyeony/event-sae-head-full10-results/blob/69c33d6a943c253dc97cf9a3968095276f7e9540/reports/report.md),
[task subgroup 보고서](https://huggingface.co/datasets/jiyeony/event-sae-head-full10-results/blob/69c33d6a943c253dc97cf9a3968095276f7e9540/reports/task_groups_v1.md),
[재분석 수치·입력 해시](../research_metadata/head_reanalysis_20260927.json).

## 3. 해석

### 확인한 것

Head-KL은 **변경량이 큰 후보들과 상당히 겹치면서도, 최상위 순서를 달리해 큰 성공률 하락을
일으킨 두 feature를 더 위로 올렸다.** 평균 변경량 순위의 단순 복사라는 설명과는 맞지 않는다.
하지만 변경량의 제곱·극단값·활성 상황까지 통제한 것은 아니므로 크기 효과가 완전히 제거됐다는 뜻은 아니다.

### 아직 확인하지 않은 것

“끄면 실패한다”와 “잡기·놓기를 의미하는 feature다”는 다르다. 전체 동작을 손상시키는 방향인지,
특정 행동에 선택적으로 관여하는지 아직 구분하지 않았다. 자연 발생 실패의 원인을 찾아냈거나,
로봇 성공률·안전성을 개선한 것도 아니다. 사후 평균 하락 0은 모든 trial 행동이 같았다는 뜻이 아니다.

### 연구 위치와 기존 연구와의 관계

목표 기여는 **로봇 정책 내부 분석을 위한 개입 후보 선별**이다. 출력 KL 계산 자체나
SAE feature 제거 자체의 최초 제안이 아니다. [Sensitive Directions](https://arxiv.org/html/2410.12555v1)는
언어 모델에서 내부 방향 변경과 출력 KL을, [Dr.VLA](https://arxiv.org/html/2603.19183v2)는
feature 일반성과 행동 개입을 이미 연구했다.

우리가 비교한 Event score는 [Event-SAE의 구체적인 event-aligned 순위 기준](https://arxiv.org/html/2605.17204v1#S3.SS4)이지
보편적인 로봇 평가 표준이 아니다. Event-SAE 전체의 해석가능성보다 우수하다고 주장하지 않는다.
Head-KL의 가치가 강화되려면 타 연구 기반 선별 기준보다도 제한된 평가 예산에서 유용한 후보를 잘 찾아야 한다.
**40.88시간의 평가를 수행했다는 사실은 40시간을 절약했다는 증거가 아니다.**

## 4. 한계

- 한 모델·한 SAE·한 task suite의 결과다. 다른 모델·SAE·환경으로 일반화하지 않는다.
- 기존 SAE와 전체 500개 episode에서 만든 Event cluster를 재사용했다. 점수 계산과
  평가 trial을 분리한 것이 SAE/cluster 학습까지 완전히 분리했다는 뜻은 아니다.
- 원수집 초기상태 hash 및 일부 batch/padding/KV-cache 설정은 검증되지 않았다.
  현재 LIBERO 상태 hash와 parity는 새 실행을 검증할 뿐, 과거 수집의 동일성을 증명하지 않는다.
- task 0·1은 pilot에 사용됐다. task 2–9도 label 사용 이력이 독립적으로 검증되지 않아
  확증 실험으로 간주하지 않는다. 점수 계산에 전 task를 사용했으므로 held-out task 평가도 아니다.
- 원 계획은 새 full10 평가 전에 동결됐지만, 변경량·토큰 변경률 재분석은 **사후 탐색적 비교**다.
- 실제 행동 평가는 선택된 16개 후보에 한정된다. 신뢰구간은 고정된 task·feature 후보군에
  조건부이며, 점수 추정·SAE 학습·다른 task/seed 변동까지 포함하지 않는다.
- Head-KL Top-3의 `18471`은 평균 하락이 0이었다. 큰 효과는 주로 `24830`, `7024`에 있다.
  모든 상위 feature가 중요하다는 결론이나 전체 dictionary의 완벽한 순위는 주장하지 않는다.
- 점수는 fixed-prefix **전체 vocabulary KL**이다. action-token 전용 KL이나 디코딩된 물리적
  action 변화는 검증된 decoder metadata가 없어 이 실험에서 측정되지 않았다.
- Head와 Event의 시간/token 집계·legacy TopK 인코딩 조건이 같지 않다. 점수 수식만을 완벽히
  통제한 비교로 해석하지 않으며, 후속 비교에서 공통 시계열 표현과 변경점을 명시해야 한다.
- 현재 결과만으로 의미 있는 행동 feature, 안전한 steering, 다른 모델로의 일반화를 입증하지 않는다.

## 5. Ablation study

### 완료: 제거·대조 실험

- 단일 feature 16개 각각 제거, raw 및 identity 대조, 기존 행동 기록의 identity 일치 확인.
- Event / Head-KL / 평균 활성값 / 활성 빈도 기준 비교와 무작위 audit 6개 평가.
- task 0·1 및 2–9 subgroup 기술통계. 순위는 전체 계획 그대로 유지했다.

### 완료: 사후 재분석 (새 개입 ablation은 아님)

- Head-KL–평균 변경량 순위 상관 및 Top-K 겹침 분석.
- 이미 저장한 평균 변경량·토큰 변경률로 Top-3 평균 하락과 고정 후보군 상관 재계산.
- 이는 **동일한 크기의 방향을 개입한 실험**, **여러 제거 강도의 실험**, **새 Top-K 후보의
  rollout**을 대신하지 않는다.

### 아직 미실행: 제안된 추가 검증

- 같은 변경량의 feature/방향 대조: 큰 신호·일반적인 손상 효과와 분리.
- 완전 제거 대신 약한 억제 및 행동 단계별 개입: 특정 행동 관여 여부 확인.
- 전체 vocabulary KL vs action-token KL/물리적 action 변화.
- Dr.VLA 기반 일반성 점수와 비교. 아래 점검과 규칙 확정까지만 완료했고 점수·새 rollout은 미실행.

2026-09-27 사용자 요청으로 **다섯 검증 모두 전체 10-task에서 진행하는 후속 개발**을 시작했다.
기존 실험 소스는 고정하고 별도 브랜치/모듈에 계산 kernel·시간축 latent 재생성·통계·예산 설계를
추가했다. 이는 실제 다섯 실험을 완료했다는 뜻이 아니다. 새 GPU rollout·실제 action/Dr.VLA
점수·phase 판별 검증은 아직 남아 있다. [개발 상태·전체 실행 규모](runbook.md#followup-all5)를 참고한다.

현재는 **④·⑤ 오프라인 비교를 먼저 수행**한다. 백업 36개 파일의 별도 복원·해시 검증,
pinned 모델/tokenizer의 256개 action token 매핑 확인, 기존 6개 지표 비교 재계산을 완료했다.
action 점수 CLI, 전체 시계열 처리, 독립 라벨 검토 자료, 보정·비교 보고서까지 연결했으나,
**새 action-KL/Dr.VLA 점수는 아직 없다.** 기존 SAE 복원·GPU 검증·원본 activation 재인코딩·
실제 영상 기반 일반성 라벨이 필요하다. 이 단계의 새 rollout은 **0회**이며,
기존 실험을 덮어쓰지 않는다. [1단계 실행 순서](runbook.md#stage1-offline)를 따른다.

### 다음 단계 결정 — 2026-09-27

**지금 전면 재수집·재학습은 하지 않는다.** 먼저 기존 dense activation에서 손실 없는
전체 시계열 SAE 기록을 재생성한다. 이는 데이터 재수집이나 OpenVLA 재학습과 다르다.

Dr.VLA 계산 가능 여부 점검 결과:

- 공개 원본 dense 369개와 index의 pinned 경로·크기를 확인했다. 약 **280.85 GiB**이며
  이번에 전체 tensor를 내려받거나 재인코딩하지는 않았다.
- Head cache는 250 episode × 8 step뿐이라 onset/run-length에 부적합하다.
- Head readout 14,000개 중 **1,581개는 활성 feature가 64개 초과**(최대 333개).
  legacy 자료는 token당 Top-64만 보존하며, 같은 인코딩/누락 없음이 인증되지 않았다.
- 따라서 공개 TopK 목록에서 빠진 feature를 0으로 처리해 Dr.VLA를 바로 계산하지 않는다.

[실제 점검 수치](../research_metadata/drvla_feasibility_20260927.json),
[후속 비교 규칙](../configs/research/openvla/drvla_comparison_protocol_v1.json),
[다음 단계와 실행 경계](runbook.md#d-drvla-비교-점검과-후속-규칙)를 참고한다.


## 6. Hugging Face 저장소는 각각 무엇인가

같은 연구의 **원본 → 가공 → 중간 산출물 → 실험 결과**를 별도로 보관한 것이다.
저장소들이 모두 같은 데이터의 중복 사본인 것은 아니다.

| 저장소 | 역할 | 이번 Head 연구에서의 용도 |
|---|---|---|
| `event-sae-libero-spatial-tasks-0-1` … `8-9` | 원본 500회 수집을 2 task씩 나눈 5개 저장소. dense activation·index·action·영상 등 | Head 점수 계산과 기존 SAE의 기반 원본 |
| [libero-spatial-openvla-rollouts-500](https://huggingface.co/datasets/jiyeony/libero-spatial-openvla-rollouts-500) | 위 5개를 task/episode/step 표와 영상으로 정리한 파생 데이터. 500 rollouts, 62,558 steps | 행동·영상 확인용. dense activation의 대체물이 아니며 `headfull download`의 별도 입력 다운로드 대상도 아님 |
| [event-sae-libero-spatial-reproduction](https://huggingface.co/datasets/jiyeony/event-sae-libero-spatial-reproduction) | 기존 Baseline의 TopK SAE 활성·event·cluster·ranking·fidelity·대조/개입 검증 결과 | full10의 discovery-only Event score를 다시 계산할 때 TopK/event/cluster 입력 사용 |
| [event-sae-head-pilot-results](https://huggingface.co/datasets/jiyeony/event-sae-head-pilot-results) | 초기 task 0·1, 4 case, 72-rollout pilot 백업 | 개발 이력. full10 열람에는 불필요. 기존 기록상 private, 현재 비로그인 API는 401 |
| [event-sae-head-full10-results](https://huggingface.co/datasets/jiyeony/event-sae-head-full10-results) | **이번 10-task·3,600-rollout의 최종 공개 결과** | 결과 열람·백업 복구·저장된 성공률 하락 재확인에 필요한 저장소 |

원본 5개: [0–1](https://huggingface.co/datasets/jiyeony/event-sae-libero-spatial-tasks-0-1),
[2–3](https://huggingface.co/datasets/jiyeony/event-sae-libero-spatial-tasks-2-3),
[4–5](https://huggingface.co/datasets/jiyeony/event-sae-libero-spatial-tasks-4-5),
[6–7](https://huggingface.co/datasets/jiyeony/event-sae-libero-spatial-tasks-6-7),
[8–9](https://huggingface.co/datasets/jiyeony/event-sae-libero-spatial-tasks-8-9).
각 저장소는 2 task × 50 = 100회 수집이다. `rollouts-500/manifest.json`에도 이 5개 출처가 기록되어 있다.

데이터셋과 별개로 재계산에 필요한 모델 저장소:

- [기존 Layer 31 SAE](https://huggingface.co/jiyeony/event-sae-openvla-libero-spatial-layer31-paper)
- [OpenVLA LIBERO-Spatial](https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial)

**결과를 읽거나 GPU 없이 검증하려면 full10-results만 있으면 된다.** 그러나 Head/Event
점수부터 새 rollout까지 다시 실행하려면 원본 5개 + reproduction의 필요한 입력 + SAE +
OpenVLA가 필요하다. full10 백업에는 약 281 GiB 원본 dense와 전체 모델 가중치가 중복 포함되지
않으며 고정 revision/hash로 참조한다. 따라서 다른 저장소를 삭제해도 된다는 뜻은 아니다.

## 7. 어디에 무엇이 보관됐나

- 코드: [Event-SAE-Head](https://github.com/jiyeon-yoon/Event-SAE-Head).
- 실제 실행 commit: `8924a8d9b62edb16d261785cbc2124729017f2b1`.
- 공개 결과 revision: `69c33d6a943c253dc97cf9a3968095276f7e9540`.
- archive: `experiment.tar.gz`, 1,534,202,180 bytes, 181개 파일 목록.
- SHA-256: `07eeb5c182ee4e195eb526362db9232ae1717de4c94b9b6da2fd62ef288de8bc`.
- 원격 재다운로드 검증: 사용자 RunPod 출력에서 `verified`, 18조건, 3,600회 확인.
- 실제 저장된 결과·action·로그·점수·readout cache·head·parity·plan·분석·설정·당시 소스가 포함된다.
  원래 경로 문자열은 보존된다. 저장하지 않은 새 rollout 영상은 백업에도 없다.

이전 설계 및 Baseline/pilot 문서는 [archive](archive)에 보관한다.
옛 문서의 “미실행/미구현”은 당시 기록이며 현재 상태는 이 문서를 기준으로 한다.
