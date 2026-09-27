# 실험 ③ — 출력층 민감도 예측: 연구·결과·데이터 안내

현재 상태: **LIBERO-Spatial 전체 10-task 후속 실험과 공개 백업 완료**.
실행·다운로드·복구 명령은 [runbook.md](runbook.md)만 보면 된다.

## 0. 한 줄 요약

고정된 OpenVLA/SAE에서 **feature 제거 전후 출력 확률분포의 KL(Head-KL)**이
기존 Event score보다 실제 행동에 중요한 feature를 더 잘 선별했다.
이는 이번 평가 범위의 후속 결과이며, 다른 모델/환경으로의 일반화를 보장하지 않는다.

## 1. 무엇을 돌렸나

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

먼저 저장된 activation의 feature 기여를 하나씩 제거하고, **동일한 생성 문맥
(fixed prefix)의 전체 vocabulary 출력 분포**가 얼마나 변하는지 KL로 점수화했다.
이후 선별된 feature를 실제 closed-loop rollout에서 각각 제거하여 성공률 하락을 측정했다.
SAE/정책 재학습, 새 모델 도입, 원본 activation 재수집은 하지 않았다.

Head와 Event는 동일한 250개 discovery episode를 사용했다. 다만 Head는 episode당
최대 8개 step, Event는 기존 event/window 정의에 필요한 discovery step을 사용한다.
동일 episode 범위라는 뜻이지, 두 점수가 완전히 같은 token 집합에서 계산됐다는 뜻은 아니다.

## 2. 결과

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

수치 출처: 공개본의 [요약](https://huggingface.co/datasets/jiyeony/event-sae-head-full10-results/blob/69c33d6a943c253dc97cf9a3968095276f7e9540/reports/summary.json),
[전체 보고서](https://huggingface.co/datasets/jiyeony/event-sae-head-full10-results/blob/69c33d6a943c253dc97cf9a3968095276f7e9540/reports/report.md),
[task subgroup 보고서](https://huggingface.co/datasets/jiyeony/event-sae-head-full10-results/blob/69c33d6a943c253dc97cf9a3968095276f7e9540/reports/task_groups_v1.md).

## 3. 해석

이번 범위에서는 **이벤트와 관련된 feature를 고르는 것보다, 제거할 때 출력 분포가
크게 바뀌는 feature를 고르는 편이 실제 행동을 망가뜨리는 feature를 더 잘 찾았다.**
로봇 성공률을 높인 실험이 아니라, 행동에 중요한 feature를 사전 선별하는 실험이다.
단순히 자주 켜지거나 활성값이 큰 feature가 반드시 중요한 것은 아니었다.

## 4. 한계

- 한 모델·한 SAE·한 task suite의 결과다. 다른 모델·SAE·환경으로 일반화하지 않는다.
- 기존 SAE와 전체 500개 episode에서 만든 Event cluster를 재사용했다. 점수 계산과
  평가 trial을 분리한 것이 SAE/cluster 학습까지 완전히 분리했다는 뜻은 아니다.
- 원수집 초기상태 hash 및 일부 batch/padding/KV-cache 설정은 검증되지 않았다.
  현재 LIBERO 상태 hash와 parity는 새 실행을 검증할 뿐, 과거 수집의 동일성을 증명하지 않는다.
- task 0·1은 pilot에 사용됐다. task 2–9도 label 사용 이력이 독립적으로 검증되지 않아
  확증 실험으로 간주하지 않는다. 점수 계산에 전 task를 사용했으므로 held-out task 평가도 아니다.
- 실제 행동 평가는 16개 후보에 한정된다. 신뢰구간은 고정된 task·feature 후보군에
  조건부이며, 점수 추정·SAE 학습·다른 task/seed 변동까지 포함하지 않는다.
- Head-KL Top-3의 `18471`은 평균 하락이 0이었다. 큰 효과는 주로 `24830`, `7024`에 있다.
  모든 상위 feature가 중요하다는 결론이나 전체 dictionary의 완벽한 순위는 주장하지 않는다.

## 5. Ablation study

완료: 단일 feature 16개 제거, raw/identity 대조(행동 기록 동일성 확인),
네 선정 기준 비교, 무작위 audit 6개 평가.

추가 검증 후보(**아직 미실행**): 전체 vocabulary KL vs action-token KL,
활성/편집 벡터 크기를 맞춘 비교, 제거 강도 변화, Top-K 변화.
현재는 제거·baseline 비교가 완료된 것이며, Head-KL 구성 요소를 분리한 ablation은 남아 있다.

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
