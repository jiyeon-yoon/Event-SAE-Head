# Event-SAE Head

OpenVLA L31 SAE feature의 출력층 민감도가 closed-loop feature 제거 시 성공률 감소를
예측하고, 평가할 feature 후보를 선정하는 데 유용한지 연구한다.

- 전체 설계서: [출력층 민감도 연구 계획 v1.1](docs/research/CODEX_EVENT_SAE_OUTPUT_HEAD_SENSITIVITY_PLAN.md)
- 기반 저장소: [jiyeon-yoon/Event-SAE-Baseline](https://github.com/jiyeon-yoon/Event-SAE-Baseline)
- 기반 commit: `56e9f012aa88532f9880259b371d9300a8013b9e`
- 범위: 기존 OpenVLA·L31 SAE·LIBERO-Spatial 데이터를 재사용하며, 신규 모델 도입이나 SAE 재학습은 하지 않는다.
- 새 연구 상태: 설계서 준비. M0~M3 구현과 CPU 테스트, M4~M5 실모델 검증·실험은 아직 수행하지 않았다.

이 연구는 독립 저장소에서 개발한다. 실험 입력과 파생 모델 tensor, cache, rollout 결과는
Git 외부에 보관한다. 설계서의 최초 구현 범위는 M0~M3 및 synthetic CPU 테스트다.

아래 내용은 기반 Baseline에서 상속한 재현 안내와 검증 기록이다.
출력층 민감도 연구의 새 CLI와 실험 완료 상태를 의미하지 않는다.

---

## 기반 Baseline 소개

[xc-j/Event-SAE](https://github.com/xc-j/Event-SAE)의 공개 코드를 기반으로
OpenVLA + LIBERO-Spatial 실험을 재현하고 검증할 수 있게 정리한 baseline이다.

- 원본 논문: [Event-Grounded Sparse Autoencoders for Vision-Language-Action Policies](https://arxiv.org/abs/2605.17204)
- 고정 코드: `fd3bc485668b8fb32b3948a9b642859f41b16277`
- 범위: LIBERO-Spatial 10 tasks × 50 rollouts, OpenVLA Layer 31

## 무엇이 추가됐나

원본의 activation 수집, BatchTopK SAE 학습, AWE, event clustering, feature
ranking, intervention 코어는 유지했다. 그 위에 다음 재현·검증 계층을 추가했다.

- 논문 설정과 OpenVLA model/code revision 고정
- 5개 GPU에서 분할 수집한 500 rollouts의 검증·무손실 병합
- 전체 activation의 FVE, alive fraction, L0와 보조 MSE 평가
- AWE → clustering → feature ranking 일괄 실행 및 결과 검증
- Raw policy와 SAE reconstruction policy의 closed-loop 성공률 비교
- 동일 initial state 기반 intervention 구현 검증과 전체 sweep 실행기

## 데이터 수집

수집 코어는 원본 Event-SAE의 `scripts/openvla/collect_activations.py`를 사용했다.

```text
LIBERO-Spatial 10 tasks × task당 50 rollouts = 500 rollouts
→ RGB + task instruction으로 OpenVLA가 7D action 예측
→ LIBERO가 action 실행
→ 매 policy forward의 Layer 31 post-MLP residual 저장
```

실제 수집은 RTX 4090 Pod 5개에서 task를 `0–1`, `2–3`, `4–5`, `6–7`,
`8–9`로 나눠 진행했다. 각 부분은 Hugging Face에 올린 뒤 global episode ID와
activation index를 함께 다시 매핑해 병합한다.

저장 항목은 dense activation, activation index, 7D action, EEF pose, gripper,
task 정보, 성공 로그, rollout MP4다. Object pose, contact/grasp, reward, subgoal
predicate를 추가로 기록하는 확장 수집기는 이 baseline 범위에 포함하지 않는다.

## 처음 실행할 때

두 경로를 혼동하지 않는다.

1. **기존 500 rollouts와 학습된 SAE로 재현** — 처음 확인할 때 권장
   [LIBERO-Spatial 500 재현 Runbook](docs/reproduce_libero_spatial_500.md)을 순서대로 실행한다.
2. **activation 수집과 SAE 학습부터 다시 수행** — 비용과 시간이 더 필요
   [OpenVLA 가이드](docs/openvla.md)의 Phase 1부터 실행한다. 수집은 RTX 4090으로
   가능하지만, 논문 설정의 SAE 학습은 24 GB VRAM에서 OOM이 확인되어 H100 80 GB
   환경에서 검증했다.

기존 데이터 재현에 사용하는 RunPod 설정:

| 항목 | 값 |
|---|---|
| Container image | `ghcr.io/jiyeon-yoon/event-sae-runtime@sha256:ef895731ffd73985e1183c964b479bdcc7e97dfeabea6c8777b903c54ade1405` |
| GPU | RTX 4090 1장 이상 |
| Container disk | 400 GB |
| Volume disk | 0 GB 가능. 종료 전 결과를 외부에 업로드 |

최소 시작 명령:

```bash
event-sae-init
event-sae-verify --require-gpu

cd /workspace
git clone https://github.com/jiyeon-yoon/Event-SAE-Baseline.git
cd Event-SAE-Baseline
git checkout --detach fd3bc485668b8fb32b3948a9b642859f41b16277
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q
```

이후 입력 다운로드부터는 [재현 Runbook](docs/reproduce_libero_spatial_500.md)을 따른다.

## 검증 상태

- Spatial-500 discovery pipeline: 완료
- Raw vs SAE reconstruction Hooked SR: 완료
- Intervention 개발 검증: 완료
- 논문 규모 전체 intervention sweep: 실행 코드 제공, 전체 실험은 선택 사항

파생 결과는 [Hugging Face](https://huggingface.co/datasets/jiyeony/event-sae-libero-spatial-reproduction)에서 확인할 수 있다.

## 주요 파일

```text
event_sae/                         Event-SAE 코어와 평가 코드
scripts/openvla/                  수집·재현·Hooked SR·intervention CLI
configs/reproduction/openvla/     고정 재현 설정
docs/reproduce_libero_spatial_500.md
tests/                             병합·평가·hook·intervention 검증
```

## License

MIT. 원본 및 외부 의존성의 라이선스는 각 저장소를 따른다.
