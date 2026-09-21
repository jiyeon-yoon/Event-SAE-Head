# 출력층 민감도 전체 10-task 후속 실험 설계

2026-09-21. 사용자의 요청: 2-task pilot을 넘어 논문을 목표로 전체 실험을 수행한다.
추천 규모는 **LIBERO-Spatial 10개 task × task당 20개 평가 초기상태**다.
이는 비용과 평가 범위를 고려한 초기 설계이며, 통계 검정력이나 게재를 보장하는
최소 표본 수라는 뜻은 아니다. 새 Pod은 아직 없고, GPU 실험을 실행하지 않았다.

## 연구 질문과 고정 범위

하나의 고정된 OpenVLA·Layer-31 SAE에서, feature를 제거한 뒤 계산하는
fixed-prefix full-vocabulary KL이 feature 제거로 인한 closed-loop 성공률
하락을 얼마나 잘 예측하는지 검증한다. 기존 SAE와 데이터는 재사용한다.
새 VLA 도입, 모델 fine-tuning, SAE 재학습은 포함하지 않는다.

표본 수만 늘리는 것으로 논문의 신규성이 생기는 것은 아니다. 기여는 기존
Event ranking과 출력층 민감도 ranking의 공정한 비교, 실패 사례 및 계산 비용을
포함한 예측력 평가에 둔다. 장기 행동 결과가 국소 KL과 항상 일치한다고 가정하지 않는다.

## 규모와 rollout 예산

| 항목 | 고정할 설계 |
|---|---|
| task | 0–9 전체 |
| 기존 task당 50 episode | discovery 25 / validation 5 / evaluation 20 |
| Head readout | 최대 250 episodes × 8 steps × 7 dimensions = 14,000 |
| 비교 순위 | Event score, Head-KL, 평균 활성 크기, 활성 빈도 |
| Top-K | 각 방법 top-3의 합집합 |
| 무작위 비교군 | discovery에서 실제 활성화된 feature 중 6개, 고정 seed |
| 최대 서로 다른 feature | 4 × 3 + 6 = 18개 |
| 평가 초기상태 | 10 × 20 = 200개 |
| control | raw + alpha=1 identity |
| 최대 실제 실행 | (18 + 2) × 200 = **4,000 rollouts** |

이는 feature별로 모델 가중치를 재학습하는 실험이 아니다. feature 제거 조건을
바꾸며 동일한 task/초기상태를 다시 실행한다. 후보 중복이 있으면 총 실행 수는
4,000보다 작아지고, `plan`이 실제 횟수를 계산한다.

32,768개 feature는 오프라인 점수 계산 대상이다. 모두를 200개 case에서 rollout하는
것이 아니다. 평가 완료 후 더 유리한 K, 후보, 초기상태를 다시 선택하지 않는다.
4000회는 총량 상한으로 설정하며, 합집합을 예산에 맞게 임의로 잘라내지 않는다.

실제 raw success가 낮은 task에서는 성공률 하락의 여지가 작다. 결과를 본 뒤
해당 task를 제외하지 않고 raw success와 feature별 signed drop을 함께 보고한다.

## 데이터 분리와 이력

Head와 Event는 **동일한 25개 discovery episode**로 점수를 계산한다. 평가 20개
episode의 activation/event를 후보 선정에 포함하지 않는다. Validation 5개는
정해진 구현 검증에만 사용하며 평가 결과를 보고 파라미터를 조정하지 않는다.

기존 task 0·1의 4-case pilot 결과는 이미 관찰했다. 따라서:

- task 0·1은 개발 이력이 있는 exploratory subgroup으로 표시한다.
- task 2–9의 새로운 평가 결과를 주 근거로 삼되, 해당 label을 이전에 방법 선정에
  사용하지 않았다는 이력을 확인해야 한다. task 번호만으로 confirmatory가 되지는 않는다.
- 전체 10-task 결과와 두 subgroup 결과를 같이 보고하고, subgroup 분석에서도
  동결된 global feature 순위를 재계산하지 않는다.
- 새 protocol의 동결 시점을 기존 M4 이전으로 소급하지 않는다.
- 기존 SAE 및 고정 cluster membership을 재사용했다는 범위를 명시한다.
  ranking/evaluation 분리는 SAE 학습이나 cluster 학습의 완전한 held-out을 의미하지 않는다.

공개 prompt records에는 원수집 initial-state hash가 없다. 따라서 구현한
`followup` 모드는 원본 task/trial과 현재 LIBERO 초기상태를 연결하되,
과거 상태는 `historical_initial_state_sha256: null`로 남긴다. 현재 상태 hash는
새 rollout 검증에 사용하며, 과거와의 동일성을 증명했다고 해석하지 않는다.
실험은 trial-ID 및 현재 상태가 분리된 post-pilot followup이고 confirmatory가 아니다.

`headfull.py setup`은 새 평가 전 현재 시각으로 split을 동결한다.
재실행은 기존 동결 시점을 유지한다. `frozen_before_pilot: false`이며 과거 pilot
이전을 소급 주장하지 않는다. 기존 strict confirmatory 경로는 완화하지 않았다.

## 무엇을 결과로 비교할까

사전 주 비교는 **Head-KL top-3 평균 성공률 하락 − Event top-3 평균 성공률 하락**이다.
각 task에 같은 가중치를 주고, 모든 feature에 동일한 평가 초기상태를 사용한다.
기존 분석의 shared-state/task-stratified bootstrap으로 차이의 불확실성을 함께 보고한다.
신뢰구간이 0을 포함하면 우열이 불확실하다고 보고한다.

보조 결과:

- 고정된 후보 panel에서 점수와 실제 하락의 Spearman correlation.
- task별 raw 성공률과 feature 제거 성공률, 개선/악화 전이 수.
- 평균 활성 크기·빈도 baseline 및 무작위 audit feature 비교.
- feature의 활성률, 편집 크기(`mean_edit_norm`)와 KL의 관계.
- Head score 계산 시간/메모리, 전체 rollout 시간, 상위 후보 발견에 필요한 평가 비용.

Top-K 합집합의 상관계수를 전체 32,768개 feature의 상관계수로 일반화하지 않는다.
무작위 audit 6개도 전체 dictionary를 정밀 추정할 큰 표본은 아니다.
기존 bootstrap은 고정된 평가 task/panel 안의 불확실성을 나타낸다.
다른 task 분포로의 일반화에는 task-level 변동과 추가 검증이 필요하다.

## 사용 가능한 입력과 준비 작업

5개 task-pair activation 저장소, SAE/model pinned revision과 다운로드는
[기존 재현 기록](reproduce_output_head_sensitivity_pilot.md)을 따른다.
기존 `download_libero_spatial_reproduction_inputs.py` / `merge_runs.py`를 재사용할 수 있다.
Merge는 tensor를 symlink해 사본 생성을 피하고 episode/forward ID를 일관되게 다시 매긴다.
task-pair 파일을 단순 복사해서 같은 이름의 shard/index를 덮어쓰면 안 된다.

공개 파일 목록의 합계는 source 5개 저장소 약 **280.993 GiB**, dense shard 369개다.
추가로 Event TopK, 모델 약15 GB, SAE와 새 결과 공간이 필요하다. 기존 400 GB
디스크 설정을 참고하되 새 Pod의 실제 free space를 확인한다. 이는 GPU 요금 견적은 아니다.

2026-09-21 public repository file inventory에서 아래 원시 artifact의 존재를 확인했다:

- 저장소: `jiyeony/event-sae-libero-spatial-reproduction` (dataset)
- revision: `f7eb3c8b6e7975481db7f50c82d06384d03c2e8a`
- `events/event_features.jsonl`
- `clusters/cluster_assignments.jsonl`, `clusters/clusters.jsonl`
- `topk/manifest.json`, `topk/shard_*.pt`

TopK manifest의 `activation_index_sha256`는
`d6da7ab63bfd945e00a507584ba61beee0b2a9f4cb3b21de25bf616f5662ccea`이고,
SAE hash는 기존 `18443083...f3ba9d6`와 일치했다. 재병합한 index와 이 hash를
대조하여 같은 episode/forward mapping인지 확인한다. sparse tensor에는 task ID가
없으므로 원본 prompt의 global episode mapping을 사용한다. `fidelity_shard_*.pt`는
Event 재계산에 필요 없다.

공개 수집 로그만으로 batch/padding/KV-cache 설정까지 확인되지는 않았다.
새 registry 도구가 전체 index의 7-forward 구조와 cached single-row 구조를 검사한다.
batch=1/no-padding/use-cache 해석은 현재 collector 코드와 구조에 근거한 가정으로
기록하며 `historical_runtime_flags_verified: false`를 유지한다. 이 해석 manifest는
followup에서만 허용된다. 새 parity는 현재 실행의 일치성을 검증할 뿐 과거 설정을 증명하지 않는다.

Event 재계산에는 기존 cluster membership을 유지하되 discovery episode만 사용한다.
window, task mean, episode coverage도 discovery 범위로 다시 집계한다.
기존 50-episode aggregate score에 새로운 split hash만 붙여 사용하는 것은 허용하지 않는다.
`headevents.py`가 episode filtering과 최종 score scope 발급까지 연결한다.
Head sample, source registry, split, TopK SAE/index checksum을 대조한다.
실제 점수를 discovery만으로 다시 계산하고 일치할 때만 설정의 Event 입력을 연결한다.

## 실행 전 완료해야 할 단계

1. 이전 결과 백업 확인. [복구 안내](resume_output_head_sensitivity.md).
2. 전체 입력 metadata 및 episode identity 검증, 후속 protocol 동결.
3. 25/5/20 split manifest 및 eval manifest 생성.
4. 같은 discovery로 Head/Event 재계산하고 scope 일치를 검사.
5. 새 코드·장비에서 head/edit/identity parity 검증.
6. deduplicate된 최종 plan의 횟수와 소요 비용 확인.
7. 고정 plan을 실행한 뒤 전체·subgroup 분석과 백업.

아직 접근 가능한 새 GPU Pod은 없으며 위 전체 실행을 완료했다고 주장하지 않는다.
전체 rollout의 시간은 기존 점수 계산 시간 676초로 추정할 수 없다.
같은 runtime에서 raw rollout 등 고정된 첫 조건의 실제 시간을 측정해 남은 시간과
RunPod 화면에 표시된 시간당 요금으로 비용을 계산한다. timing을 이유로 어려운
case나 결과를 삭제하지 않는다.

## 설정 템플릿

`configs/research/openvla/output_head_sensitivity_full10.yaml`은 위 규모를 명시한다.
경로/이력이 없는 상태에서 audit가 blocked인 것은 의도된 동작이다.
경로와 manifest는 `headfull.py setup`으로 생성한다. 템플릿만으로 GPU가 실행되지는 않는다.

```bash
python scripts/openvla/output_head_sensitivity.py audit --config configs/research/openvla/output_head_sensitivity_full10.yaml
```

GPU 실행은 검증된 local config와 동결 plan이 만들어진 후 기존 `run` 명령을 사용한다.
기존 pilot artifact에는 새 결과를 덮어쓰지 않는다.

## 이번 개발과 검증 결과

- `score_cluster_features(..., allowed_episode_nums=...)` 및
  `scripts/score_cluster_features.py --allowed-episodes-path ...` 추가.
  task mean까지 제외 episode의 영향을 받지 않는지 synthetic test로 확인했다.
- `scripts/openvla/headgroups.py` 추가. 원 plan의 feature/Top-K를 유지한 채
  task 0·1과 task 2–9의 평균/전이 수/CI를 paired rows에서 다시 계산한다.
  완료된 full10 분석 후 다음을 실행한다:

```bash
python scripts/openvla/headgroups.py --config "$CFG"
```

출력은 `analysis/task_groups_v1.json`, `analysis/task_groups_v1.md`이며
기존 전체 분석 파일을 수정하지 않는다.

초기 설계 단계의 254-test 검증 이후 identity와 후속 protocol/scope 연결을 구현했다.
최종 실행 절차와 검증 범위는 [전체 실행 안내](run_output_head_full10.md)를 따른다.
실제 전체 데이터 검증·GPU score·4000회 이내 rollout은 새 Pod에서 수행해야 한다.
