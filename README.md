# Event-SAE Head

**실험 ③ 출력층 민감도 예측:** SAE feature를 제거했을 때 OpenVLA 출력 확률분포의
변화(Head-KL)로 실제 행동에 중요한 feature를 미리 선별할 수 있는지 검증한다.

## 문서는 두 개만 읽으면 됩니다

1. **[연구·결과·데이터 안내](docs/experiment.md)** — 무엇을 했는지, 결과·한계·ablation, HF 저장소별 역할.
2. **[실행·복구 안내](docs/runbook.md)** — GPU 없는 결과 확인, 새 Pod 실험, 분석·백업 명령.

## 현재 상태

- 기존 OpenVLA·Layer 31 SAE 고정. 재학습·새 데이터 수집 없이 수행.
- 2-task / 72-rollout pilot 이후 **10-task / 3,600-rollout 후속 실험 완료**.
- 16개 feature 제거 + raw/identity 대조군 = 18조건, 조건당 200회 평가.
- Top-3 평균 성공률 하락: Head-KL **52.50%p**, Event **23.67%p**.
- Head-KL − Event 차이 **28.83%p**, 조건부 95% CI **[26.50, 31.17]%p**.
- 고정된 모델·SAE·task·feature 후보군의 후속 결과다. 완전한 held-out 확증 실험은 아니다.

Top-3는 세 feature를 **각각 하나씩 제거한 실험의 평균**이다. 로봇 성공률을 높였다는
뜻이 아니라, 제거하면 성공률이 크게 떨어지는 feature를 선별했다는 뜻이다.

**[공개 실험 결과·원자료](https://huggingface.co/datasets/jiyeony/event-sae-head-full10-results/tree/69c33d6a943c253dc97cf9a3968095276f7e9540)**
에는 보고서, 3,600회 평가 목록, 전체 archive, GPU 없는 검증·복구 도구가 있다.
결과를 읽는 데는 이 저장소만 있으면 된다. 점수·rollout을 다시 계산할 때는 원본 입력도 필요하다.

## 코드와 기록의 구분

- 실제 full10 실행 commit: `8924a8d9b62edb16d261785cbc2124729017f2b1`.
- HF 공개본 revision: `69c33d6a943c253dc97cf9a3968095276f7e9540`.
- GitHub `main`에는 이후 백업 도구와 문서 정리가 포함된다. 과거 실행 commit과 다르다.
- 기반: [Event-SAE-Baseline](https://github.com/jiyeon-yoon/Event-SAE-Baseline), 분리 기준 commit `56e9f012aa88532f9880259b371d9300a8013b9e`.
- 이전 설계·pilot·Baseline 문서는 [docs/archive](docs/archive)에 보관했다. 현재 시작점은 위 두 문서다.

이 저장소만 독립적으로 변경한다. Baseline, 다른 팀원의 저장소, 원본 데이터는 수정하지 않는다.

## License

MIT. 원본 및 외부 의존성의 라이선스는 각 저장소를 따른다.
