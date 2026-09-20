# OpenVLA 출력층 민감도 Pilot: RunPod 재현 기록

이 문서는 2026-09-20~21에 RTX 4090 RunPod에서 실제로 수행한
Event-SAE output-head sensitivity pilot을 설명하고, 종료 전에 남긴 명령을
재현 가능한 순서로 정리한다.

이것은 원본 terminal transcript를 무작정 복사한 문서가 아니다. 잘못된 줄바꿈,
빠진 공백, deprecated `huggingface-cli`는 바로잡고 `hf`로 통일했다.
대화에 최초 download 한 줄이 전체 남아 있지 않은 부분은 최종 경로와
pinned revision에 맞춰 재구성했으며, 해당 부분에 명시했다.

## 1. 동료에게 한 문장으로 설명하기

> 기존 Event-SAE가 “특정 이벤트와 함께 켜지는 feature”를 찾는다면, 이 연구는
> “그 feature를 없앴을 때 OpenVLA의 최종 출력 분포가 얼마나 변하는가”를
> output head에서 직접 점수화하고, 그 점수가 실제 로봇 성공률 하락을 잘
> 예측하는지 검증한 확장이다.

기존 OpenVLA, LIBERO-Spatial activation, Layer 31 BatchTopK SAE를 그대로 재사용했다.
OpenVLA를 fine-tuning하지 않았고 SAE도 재학습하지 않았다.
오프라인 주 점수는 최종 레이어 fixed-prefix readout에서 feature 하나를
`alpha=0`으로 제거한 뒤 전체 vocabulary에 대해 계산한
`KL(p_base || p_edit)`다. Closed-loop feature 조건도 `alpha=0`, identity control은
`alpha=1`을 사용했다.

## 2. 현재 어디까지 끝났나

```text
기존 task 0·1 activation + 기존 Layer-31 SAE
→ 5,600개 fixed-prefix readout 구성
→ 실제 OpenVLA final RMSNorm + lm_head 추출
→ BF16 실모델/개입 parity 통과
→ 32,768개 SAE feature 전체의 output-head KL 계산
→ Event score와 Head discovery의 task/trial episode 범위를 일치
→ 16개 feature를 LIBERO closed loop에서 suppression
→ raw/identity control 포함 72 rollout
→ paired 분석과 private Hugging Face 백업
```

| 단계 | 상태 | 이번에 확인한 것 |
|---|---|---|
| M0 | 완료 | 실제 입력, revision, checksum, 실행 gate audit |
| M1 | 완료 | 5,600 readout cache, final RMSNorm/lm_head bundle export |
| M2 | 완료 | 전체 32,768 feature, 308,780 active pair의 Head-KL scoring |
| M3 | 완료 | exact Event-score adapter, 후보 plan, paired 분석 |
| M4 | **pilot 완료** | 실모델 parity, GPU score, 72회 closed-loop pilot |
| M5 | **미수행** | 독립 held-out task/state에서의 confirmatory 재검증 |

따라서 현재 상태는 **“방법을 end-to-end로 구현했고, GPU와 실제 closed
loop에서 돌아가며, 유망한 초기 신호까지 확인한 상태”**다. “일반적으로
Head-KL가 Event score보다 우수하다”는 최종 결론은 아직 아니다.

## 3. 실행 규모와 고정 식별자

| 항목 | 값 |
|---|---|
| GPU | RTX 4090 1장 |
| Container image | `ghcr.io/jiyeon-yoon/event-sae-runtime@sha256:ef895731ffd73985e1183c964b479bdcc7e97dfeabea6c8777b903c54ade1405` |
| Container disk | 400 GB (권장/사용 설정; terminal 로그에서 재측정하지 않음) |
| 실험 Git commit | `9cadd10b8498b2a6ba5e204b6338c9f796a3ffa5` |
| OpenVLA weights | `962318cec55ac10993ff0f5f43eda9a270b4c873` |
| OpenVLA remote code | `47a0ec7fc4ec123775a391911046cf33cf9ed83f` |
| task 0–1 activation data | `3babb75de5cedf5bed93daf192c955cd43cdff36` |
| SAE repository | `adb776b08f5b8ec4ea67556cf3e3bc60d91d607a` |
| Event-score artifact | `f7eb3c8b6e7975481db7f50c82d06384d03c2e8a` |

RunPod에서 순차적으로 받아 검증한 주요 개발 revision:

| Revision | 역할 |
|---|---|
| `50447daab62fae0da92b7a441f51170fba6a4e23` | 출력층 민감도 workflow 최초 구현 |
| `07e48ad4eacca89ac9056fa82687add22c042c9a` | sparse OpenVLA text config 지원 |
| `0eda3412113a1d3fbe21d7b9579e0b7b69c54453` | local `headsetup.py` 추가 |
| `a4755e749b30ea99532a2caa11baf08f1282f681` | live runtime input capture 추가 |
| `7c1ec113f5d8572b6bdb4f0da3f8ef01fd5c4a89` | BF16 parity protocol 분리 |
| `9cadd10b8498b2a6ba5e204b6338c9f796a3ffa5` | exact Event-score scope adapter 추가; 최종 pilot revision |

주요 checksum:

| Artifact | SHA-256 |
|---|---|
| `trainer_0/ae.pt` | `18443083d320d2ad3c607431f307697639966ad92075bdfe031398557f3ba9d6` |
| public `event_feature_scores.pt` | `a82c23c588a12e897c20fec31e31263a111d23fba8c67731de24e5fdafd571f3` |
| exported `output_head.safetensors` | `439bb2fd2e4a43039c90a955ffcb4e947a37c26cd0c3c4421be6f3440a173093` |
| 최종 result archive | `057e99bb74673c0ee28e599192875af4b338cfd274d8267d035988057067f1bf` |

기타 식별자:

```text
processor_identity:
sha256:9c614b767d034bf966dac2d8c4866647590f5f8981af9d99a18d85d6f67b5a67

head_bundle_fingerprint:
6708d397a30de0c85ce73fd0ced8178b3ff8ccc57b36c202065b20504e9d622e

canonical_rollout_plan_hash:
5e6080c49252221a13040a4b6925272037fbea4a2dbe59f5ae15e5622e3f1f53

rollout_protocol_id:
0635307680c947230efc178facd59e1e0ea8215b61a47a33a17741e91a2ac0b1
```

## 4. RunPod 실행 기록

### 4.1 환경과 저장소

```bash
event-sae-init
event-sae-verify --require-gpu

cd /workspace
git clone https://github.com/jiyeon-yoon/Event-SAE-Head.git
cd /workspace/Event-SAE-Head
git fetch origin main
git checkout --detach 9cadd10b8498b2a6ba5e204b6338c9f796a3ffa5
git rev-parse HEAD
```

최종 실험을 그대로 재현할 때는 detached commit을 쓰는 편이 명확하다. 개발
중 detached HEAD에서 `git pull` 오류가 난 뒤 최신 `main`으로 전환했던 명령은
다음과 같다.

```bash
cd /workspace/Event-SAE-Head
git switch main
git pull --ff-only origin main
git rev-parse --short HEAD
```

정상 출력은 `9cadd10`이었다.

### 4.2 공용 설정 audit와 local config

```bash
python scripts/openvla/output_head_sensitivity.py audit \
  --config configs/research/openvla/output_head_sensitivity.yaml

test $? -eq 2
```

공용 YAML은 경로가 `null`이므로 `status: blocked`, exit code 2가 정상이다. 주요
blocker는 `dense_dir`, `sae_checkpoint`, `event_scores_path`, `generation_manifest`,
model/head bundle, `output.root_dir`이었다.

```bash
mkdir -p /workspace/head-inputs
mkdir -p /workspace/event-sae-head-results/pilot
mkdir -p configs/local
cp configs/research/openvla/output_head_sensitivity.yaml configs/local/head.yaml
```

이 image에는 `nano`가 없었다. 이후 설정은 `sed`, Python, 제공된 CLI로 수정했다.
다운로드 전에 입력이 없음을 확인한 명령은 다음과 같다.

```bash
find /workspace -type f \( \
  -name "activation_index.jsonl" -o \
  -name "ae.pt" -o \
  -name "generation_manifest.json" -o \
  -iname "*event*score*" -o \
  -iname "*source*episode*" \
\) -print

find /root/.cache/huggingface/hub -maxdepth 5 \
  -type d -iname "*openvla*" -print 2>/dev/null
```

### 4.3 입력 download

다음은 최종 경로와 revision에서 재구성한 download 명령이다. 대화에는 최초
download의 완전한 shell line이 남아 있지 않지만, 다운로드된 공개 저장소,
pinned revision, 최종 파일 경로는 검증했다.

```bash
HF_HUB_OFFLINE=0 hf download \
  jiyeony/event-sae-libero-spatial-tasks-0-1 \
  --repo-type dataset \
  --revision 3babb75de5cedf5bed93daf192c955cd43cdff36 \
  --include "sae_activations/post_mlp_residual/*" \
  --local-dir /workspace/head-inputs/spatial-tasks-0-1

HF_HUB_OFFLINE=0 hf download \
  jiyeony/event-sae-openvla-libero-spatial-layer31-paper \
  --revision adb776b08f5b8ec4ea67556cf3e3bc60d91d607a \
  --include "trainer_0/ae.pt" "trainer_0/config.json" \
  --local-dir /workspace/head-inputs/sae-spatial-l31

HF_HUB_OFFLINE=0 hf download \
  openvla/openvla-7b-finetuned-libero-spatial \
  --revision 962318cec55ac10993ff0f5f43eda9a270b4c873 \
  --local-dir /workspace/head-inputs/openvla-spatial
```

이번 head pilot에는 `jiyeony/libero-spatial-openvla-rollouts-500`을 직접 쓰지
않았다. 위 task 0–1 activation dataset, SAE, OpenVLA snapshot, 기존 Event score가
직접 입력이다.

파일 확인:

```bash
DENSE_DIR=/workspace/head-inputs/spatial-tasks-0-1/sae_activations/post_mlp_residual
SAE_FILE=/workspace/head-inputs/sae-spatial-l31/trainer_0/ae.pt

find "$DENSE_DIR" -maxdepth 1 -name "layer_31_shard_*.pt" | wc -l
sha256sum "$SAE_FILE"
find /workspace/head-inputs/openvla-spatial \
  -maxdepth 1 -name "model-*.safetensors" | wc -l
du -sh /workspace/head-inputs/openvla-spatial
```

실제 확인값:

```text
activation shards: 66
SAE SHA-256: 18443083d320d2ad3c607431f307697639966ad92075bdfe031398557f3ba9d6
model shards: 4
model snapshot size: 15G
```

### 4.4 generation manifest와 기본 YAML

JSON은 직접 편집하다가 string 안에 literal newline을 넣어
`Invalid control character`를 만들지 않도록 `json.dump`로 생성한다.

```bash
python -c 'import json; p="/workspace/head-inputs/generation_manifest.json"; d={"source_run_id":"spatial-tasks-0-1","dataset_id":"jiyeony-event-sae-spatial-0-1","suite":"libero_spatial","layer_idx":31,"action_dim":7,"batch_size":1,"padding":"none","use_cache":True,"index_scope":"complete","evidence":"HF revision 3babb75de5cedf5bed93daf192c955cd43cdff36; collection log and 7-forward index verified"}; open(p,"w").write(json.dumps(d,indent=2)+"\n")'

python -m json.tool /workspace/head-inputs/generation_manifest.json \
  > /dev/null && echo GENERATION_MANIFEST_OK
```

기본 경로 연결:

```bash
CFG=configs/local/head.yaml

sed -i -E 's|^([[:space:]]*)dense_dir:.*|\1dense_dir: /workspace/head-inputs/spatial-tasks-0-1/sae_activations/post_mlp_residual|' "$CFG"
sed -i -E 's|^([[:space:]]*)sae_checkpoint:.*|\1sae_checkpoint: /workspace/head-inputs/sae-spatial-l31/trainer_0/ae.pt|' "$CFG"
sed -i -E 's|^([[:space:]]*)generation_manifest:.*|\1generation_manifest: /workspace/head-inputs/generation_manifest.json|' "$CFG"
sed -i -E 's|^([[:space:]]*)root_dir:.*|\1root_dir: /workspace/event-sae-head-results/pilot|' "$CFG"
```

Pilot에서 `source_episode_manifest: null`은 허용됐다. Confirmatory split을 주장하려면
검증된 initial-state/source-episode manifest가 필요하다.

### 4.5 2-episode smoke prepare

공용 설정의 기본값인 task당 2 episode로 먼저 readout 경로를 확인했다.

```bash
python scripts/openvla/output_head_sensitivity.py prepare --config "$CFG"

find /workspace/event-sae-head-results/pilot \
  -maxdepth 3 -type f | sort
```

정상 기준:

```text
status: prepared
num_readouts: 224
elapsed_seconds: 13.13
readouts/manifest.json
readouts/shard_000000.pt
sample_manifest.json
```

### 4.6 output head setup/export

수동으로 알 수 없는 hash/metadata를 작성하지 않고, 실제 snapshot을 읽는
setup CLI를 사용했다.

```bash
python scripts/openvla/headsetup.py \
  --config "$CFG" \
  --snapshot /workspace/head-inputs/openvla-spatial \
  --export
```

이 명령은 YAML의 `local_model_snapshot`, `head_export_metadata`,
`scoring.device: cuda:0`도 자동으로 설정했다. 실제 확인값:

```text
target_layer: 31 / num_layers: 32
hidden_dim: 4096
vocab_size: 32064
hidden/norm/head dtype: bfloat16
norm: llama_rms_norm_v1, eps=1e-6
head bundle fingerprint: 6708d397a30de0c85ce73fd0ced8178b3ff8ccc57b36c202065b20504e9d622e
weights SHA-256: 439bb2fd2e4a43039c90a955ffcb4e947a37c26cd0c3c4421be6f3440a173093
```

### 4.7 runtime parity input capture

첫 실행은 OpenVLA custom processor code가 local cache에 없어 실패했다. 파일명
사이에 공백이 빠진 `modeling_prismatic.pyprocessing_prismatic.py`는 잘못된
명령이므로 재사용하지 않는다.

```bash
HF_HUB_OFFLINE=0 hf download \
  openvla/openvla-7b \
  configuration_prismatic.py modeling_prismatic.py processing_prismatic.py \
  --revision 47a0ec7fc4ec123775a391911046cf33cf9ed83f

cat /workspace/cache/huggingface/hub/models--openvla--openvla-7b/refs/main

python scripts/openvla/headinputs.py \
  --config "$CFG" \
  --allow-simulator
```

정상 기준:

```text
status: captured
num_calls: 2
tasks: [0, 1]
trials: [0, 0]
runtime_inputs: /workspace/head-inputs/runtime.pt
policy_model_loaded: false
policy_action_executed: false
```

즉 이 단계는 processor와 simulator observation을 캡았지만 7B policy action을 생성하거나
환경에 실행하지는 않았다.

### 4.8 실모델 BF16 parity

```bash
python scripts/openvla/output_head_sensitivity.py validate \
  --config "$CFG" \
  --allow-model-execution
```

최초 구현에서는 BF16 연산 grouping 차이를 `atol=rtol=1e-4`로 동일하게
비교해 `max_abs_error=0.125/0.25`, `argmax_mismatch_rate=0`으로 차단됐다.
`7c1ec11` revision에서 parity protocol을 분리한 뒤 같은 명령이 통과했다.

```text
status: passed
elapsed_seconds: 11.55
validation: /workspace/event-sae-head-results/pilot/validation
```

`transformers==4.40.1`, `tokenizers==0.19.1`을 기대하는데 실행 환경이
`4.48.1`, `0.21.2`였다는 warning은 남았다. 그래서 version warning을 무시한
것이 아니라 실제 model/head/edit parity 통과를 실행 gate로 사용했다.

### 4.9 224-readout smoke score

```bash
python scripts/openvla/output_head_sensitivity.py score --config "$CFG"
```

실제 결과:

```text
status: scored
num_readouts: 224
num_scored_features: 32768
active_pair_count: 12212
elapsed_seconds: 39.60
```

`action_decoding_status: unavailable`은 verified action-bin metadata가 없다는 뜻이다.
주 지표인 fixed-prefix full-vocabulary KL는 계산됐다.

### 4.10 4개 evaluation case manifest

task 0/1의 initial-state index 0/1을 사용했다.

```bash
python -c 'import json; from libero.libero import benchmark as b; from event_sae.openvla.eval.runner import _initial_state_sha256 as h; s=b.get_benchmark_dict()["libero_spatial"](); c=[{"suite":"libero_spatial","task_id":t,"task_episode_idx":i,"initial_state_sha256":h(s.get_task_init_states(t)[i]),"initial_state_hash_provenance":"live_libero_task_suite_state_bytes_v1"} for t in (0,1) for i in (0,1)]; json.dump({"mode":"pilot","discovery_eval_overlap":False,"evaluation_labels_previously_used":None,"cases":c},open("/workspace/head-inputs/eval_manifest.json","x"),indent=2)'

python -m json.tool /workspace/head-inputs/eval_manifest.json
sed -i -E 's|^([[:space:]]*)eval_manifest:.*|\1eval_manifest: /workspace/head-inputs/eval_manifest.json|' "$CFG"
```

생성된 state hash:

| task/trial | SHA-256 |
|---|---|
| 0/0 | `9fea34420cba1eb90e3b028aa230b9edbab18ed0663096630f4e1359b9c8386d` |
| 0/1 | `cd29f337e10c70021c43588a8c99cae066a8dccd90cca18720deeec5ede38f87` |
| 1/0 | `f29b353397c823eba7a31e27f8345b36d1890a7b719f9628e47729089dfdf4b7` |
| 1/1 | `540a7afd0b9809f7cd077a65250c09227904c752cd4844bdfe3e2180daa4d037` |

이 때 `plan`은 `inputs.event_scores_path`가 없어 차단됐다. Public Event score가
task 0–9 전체로 계산된 artifact이므로, 4-episode smoke에 바로 붙이지 않았다.

### 4.11 task 0·1의 100-episode matched config

Event score와 같은 task 0/1, task당 50 episode로 Head discovery를 다시 만들었다.

```bash
cp configs/local/head.yaml configs/local/head-matched.yaml
CFG=configs/local/head-matched.yaml

cp /workspace/head-inputs/eval_manifest.json \
  /workspace/head-inputs/eval_manifest_matched.json

sed -i 's/"discovery_eval_overlap": false/"discovery_eval_overlap": true/' \
  /workspace/head-inputs/eval_manifest_matched.json
sed -i 's/"evaluation_labels_previously_used": null/"evaluation_labels_previously_used": true/' \
  /workspace/head-inputs/eval_manifest_matched.json

sed -i -E 's|^([[:space:]]*)root_dir:.*|\1root_dir: /workspace/event-sae-head-results/matched-task01-v1|' "$CFG"
sed -i -E 's|^([[:space:]]*)output_head_bundle:.*|\1output_head_bundle: /workspace/event-sae-head-results/pilot/head|' "$CFG"
sed -i -E 's|^([[:space:]]*)event_scores_path:.*|\1event_scores_path: null|' "$CFG"
sed -i -E 's|^([[:space:]]*)max_episodes_per_task:.*|\1max_episodes_per_task: 50|' "$CFG"
sed -i -E 's|^([[:space:]]*)max_scored_pairs:.*|\1max_scored_pairs: 500000|' "$CFG"
sed -i -E 's|^([[:space:]]*)eval_manifest:.*|\1eval_manifest: /workspace/head-inputs/eval_manifest_matched.json|' "$CFG"
```

`discovery_eval_overlap: true`, `evaluation_labels_previously_used: true`는 오류가 아니라
이 pilot의 실제 한계를 정직하게 기록한 것이다.

### 4.12 matched prepare, validate, score

```bash
python scripts/openvla/output_head_sensitivity.py prepare --config "$CFG"

python scripts/openvla/output_head_sensitivity.py validate \
  --config "$CFG" \
  --allow-model-execution

python scripts/openvla/output_head_sensitivity.py score --config "$CFG"
```

실제 결과:

```text
prepare:
  status: prepared
  num_readouts: 5600
  elapsed_seconds: 50.69

score:
  status: scored
  num_readouts: 5600
  num_scored_features: 32768
  active_pair_count: 308780
  executed_pairs: 308780
  elapsed_seconds: 676.60
  peak_memory_allocated_bytes: 1346489856
```

5,600은 `100 episodes × 8 sampled steps × 7 action dimensions`다. Peak allocation은
score가 보고한 CUDA allocation이며 전체 process/GPU 메모리 상한을 뜻하지는 않는다.

### 4.13 public Event score를 동일 task/trial episode 범위로 제한

```bash
EVENT_DIR=/workspace/head-inputs/event-public
EVENT_SRC="$EVENT_DIR/scores/event_feature_scores.pt"
EVENT_SUBSET=/workspace/head-inputs/event_feature_scores_tasks_0_1_exact.pt
mkdir -p "$EVENT_DIR"

HF_HUB_OFFLINE=0 hf download \
  jiyeony/event-sae-libero-spatial-reproduction \
  scores/event_feature_scores.pt \
  --repo-type dataset \
  --revision f7eb3c8b6e7975481db7f50c82d06384d03c2e8a \
  --local-dir "$EVENT_DIR"

sha256sum "$EVENT_SRC"
```

기대 checksum:

```text
a82c23c588a12e897c20fec31e31263a111d23fba8c67731de24e5fdafd571f3
```

```bash
python scripts/openvla/subset_event_scores.py \
  --source "$EVENT_SRC" \
  --sample-manifest /workspace/event-sae-head-results/matched-task01-v1/sample_manifest.json \
  --head-scores /workspace/event-sae-head-results/matched-task01-v1/scores/scores.json \
  --output "$EVENT_SUBSET" \
  --task-id 0 \
  --task-id 1 \
  --suite libero_spatial \
  --expected-source-sha256 a82c23c588a12e897c20fec31e31263a111d23fba8c67731de24e5fdafd571f3 \
  --expected-sae-sha256 18443083d320d2ad3c607431f307697639966ad92075bdfe031398557f3ba9d6 \
  --expected-episodes-per-task 50 \
  --update-config "$CFG"
```

정상 기준은 task 0/1 각 50 episode, 총 100 episode, `relation: exact_equal`이다.
여기서 exact는 가용한 `suite/task_id/task_episode_idx` 식별자가 같다는 뜻이다.
Event source에 initial-state hash가 없고 Head sample의 hash도 `null`이어서 state-byte
동일성까지 증명한 것은 아니다. 또한 adapter는 이미 집계된 Event matrix의 task
row만 필터링했다. Event는 보존된 전체 timestep을 쓰고 Head score는 episode당
고르게 선택한 8 steps를 썼으므로 timestep sampling은 동일하지 않다.

### 4.14 rollout plan

```bash
python scripts/openvla/output_head_sensitivity.py plan --config "$CFG"
```

실제 계획:

```text
status: planned
top_k: 3
selection_eval_overlap: true
unique features: 16
evaluation cases: 4
conditions: raw + identity + 16 feature edits = 18
total_rollouts: 18 × 4 = 72
```

선정된 feature ID:

```text
2723, 2875, 3165, 5632, 6553, 7024, 12409, 13724,
17857, 18471, 24360, 24830, 25653, 27955, 28491, 30729
```

### 4.15 `9cadd10` plan-hash 오류 확인과 이력적 workaround

이 절은 **실제 실험 commit `9cadd10`을 정확히 재현할 때만** 필요하다.
`score_vectors`의 integer feature key가 JSON 저장 후 string key가 되어 재로드 후
정렬 순서가 달라지는 canonicalization bug가 있었다. 후보/점수/조건을
바꾼 것이 아니라, 저장된 JSON 표현의 hash만 다시 계산했다.

```bash
PLAN=/workspace/event-sae-head-results/matched-task01-v1/plan/rollout_plan.json
LOG=/workspace/event-sae-head-results/matched-task01-v1/rollout.log

PLAN_HASH=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["plan_hash"])' "$PLAN")
CODE_REV=$(git rev-parse HEAD)
printf 'PLAN_HASH=%s\nCODE_REV=%s\n' "$PLAN_HASH" "$CODE_REV"

python -c 'import json,sys;from event_sae.research.output_head.provenance import fingerprint;p=json.load(open(sys.argv[1]));h=p.pop("plan_hash");print("PLAN_VALID=",h==fingerprint(p))' "$PLAN"
```

`PLAN_VALID=False`일 때 아래 검사로 정확히 이 알려진 bug인지 확인했다.

```bash
python -c 'import json,sys;from event_sae.research.output_head.provenance import fingerprint as f;p=json.load(open(sys.argv[1]));h=p.pop("plan_hash");p["score_vectors"]={m:{int(k):v for k,v in x.items()} for m,x in p["score_vectors"].items()};print("KNOWN_JSON_KEY_BUG=",h==f(p))' "$PLAN"
```

`KNOWN_JSON_KEY_BUG=True`일 때만 수행:

```bash
BACKUP="${PLAN%.json}.pre-canonical-hash-fix.json"
cp -n "$PLAN" "$BACKUP"

python -c 'import json,sys;from event_sae.research.output_head.provenance import fingerprint as f,atomic_write_json as w;p=sys.argv[1];d=json.load(open(p));old=d["plan_hash"];d["plan_hash"]=f({k:v for k,v in d.items() if k!="plan_hash"});w(p,d,overwrite=True);print("OLD=",old);print("NEW=",d["plan_hash"])' "$PLAN"

python -c 'import json,sys;a=json.load(open(sys.argv[1]));b=json.load(open(sys.argv[2]));a.pop("plan_hash");b.pop("plan_hash");print("UNSIGNED_PLAN_UNCHANGED=",a==b)' "$BACKUP" "$PLAN"
```

반드시 `UNSIGNED_PLAN_UNCHANGED=True`여야 한다. 최초 hash는 `9380eb...`,
보정 후 report에 기록된 hash는
`5e6080c49252221a13040a4b6925272037fbea4a2dbe59f5ae15e5622e3f1f53`이다.
앞으로는 수동 workaround가 아니라 source/test에서 JSON round-trip 안정 hash를
구현한 새 revision을 쓰는 것이 맞다.

### 4.16 72 closed-loop rollouts

```bash
PLAN_HASH=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["plan_hash"])' "$PLAN")
CODE_REV=$(git rev-parse HEAD)
test -n "$PLAN_HASH"
test -n "$CODE_REV"
test "$CODE_REV" = "9cadd10b8498b2a6ba5e204b6338c9f796a3ffa5"
test -z "$(git status --porcelain)"

nohup env PYTHONUNBUFFERED=1 \
  python scripts/openvla/output_head_sensitivity.py run \
  --config "$CFG" \
  --execute \
  --approved-plan-hash "$PLAN_HASH" \
  --expected-code-revision "$CODE_REV" \
  > "$LOG" 2>&1 &

tail -f "$LOG"
```

실제 완료 표시:

```text
status: completed
total_rollouts: 72
protocol_id: 0635307680c947230efc178facd59e1e0ea8215b61a47a33a17741e91a2ac0b1
```

`status: completed`가 출력된 뒤 process cleanup에서 `EGL_NOT_INITIALIZED` destructor
warning이 나왔다. 72개 result 저장과 completed 상태 후의 EGL context 정리
warning이므로 이 실행에서는 결과 실패로 판정하지 않았다.

### 4.17 분석

```bash
python scripts/openvla/output_head_sensitivity.py analyze \
  --config "$CFG" \
  > /workspace/event-sae-head-results/matched-task01-v1/analyze_cli.json

sed -n '1,220p' \
  /workspace/event-sae-head-results/matched-task01-v1/analysis/report.md
```

### 4.18 결과 압축과 private Hugging Face 백업

백업 저장소:

```text
https://huggingface.co/datasets/jiyeony/event-sae-head-pilot-results
```

이 저장소는 private이다. Hugging Face token은 terminal의 숨겨진 입력창에만
넣고 chat, shell command, 문서, Git에 남기지 않는다.

```bash
ARCHIVE=/workspace/event-sae-head-results/matched-task01-v1-results.tar.gz

tar -C /workspace/event-sae-head-results \
  -czf "$ARCHIVE" \
  matched-task01-v1

sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
ls -lh "$ARCHIVE" "${ARCHIVE}.sha256"

hf auth login
hf auth whoami

python -c 'from huggingface_hub import HfApi;print(HfApi().create_repo("jiyeony/event-sae-head-pilot-results",repo_type="dataset",private=True,exist_ok=True))'

HF_XET_HIGH_PERFORMANCE=1 hf upload \
  jiyeony/event-sae-head-pilot-results \
  "$ARCHIVE" \
  matched-task01-v1-results.tar.gz \
  --repo-type dataset

hf upload \
  jiyeony/event-sae-head-pilot-results \
  "${ARCHIVE}.sha256" \
  matched-task01-v1-results.tar.gz.sha256 \
  --repo-type dataset

hf upload \
  jiyeony/event-sae-head-pilot-results \
  "$CFG" \
  config/head-matched.yaml \
  --repo-type dataset

hf upload \
  jiyeony/event-sae-head-pilot-results \
  /workspace/head-inputs/eval_manifest_matched.json \
  manifests/eval_manifest_matched.json \
  --repo-type dataset
```

`.sha256` 생성 전에 checksum을 upload하려고 해 한 번 `FileNotFoundError`가
발생했다. 위 순서처럼 archive 다음 checksum을 먼저 생성하면 된다.

### 4.19 원격 백업 무결성 확인

```bash
python -c 'from huggingface_hub import HfApi;i=HfApi().repo_info("jiyeony/event-sae-head-pilot-results",repo_type="dataset",files_metadata=True);[print(f"{x.rfilename}\t{x.size}") for x in sorted(i.siblings,key=lambda x:x.rfilename)]'
```

실제 원격 목록:

```text
.gitattributes                                   2504
config/head-matched.yaml                         3644
manifests/eval_manifest_matched.json             1205
matched-task01-v1-results.tar.gz             226950882
matched-task01-v1-results.tar.gz.sha256            133
```

```bash
REMOTE_SHA=$(hf download \
  jiyeony/event-sae-head-pilot-results \
  matched-task01-v1-results.tar.gz.sha256 \
  --repo-type dataset)

cat "$REMOTE_SHA"
sha256sum "$ARCHIVE"
```

양쪽의 checksum은 모두 다음과 같았다.

```text
057e99bb74673c0ee28e599192875af4b338cfd274d8267d035988057067f1bf
```

원격 archive 무결성을 확인했으므로, 아래처럼 임시 Pod의 token을 먼저
제거한 다음 RunPod UI에서 Pod을 terminate해도 된다.

```bash
hf auth logout
```

## 5. Pilot 결과

### 5.1 실제 성공률 하락 예측

Predictor top-3 합집합(`topk_union`)에서 closed-loop signed success-rate drop과의
관찰 point estimate는 다음과 같다. 전체 16개는 random audit feature도 포함하며,
audit sample의 Spearman은 관찰 drop의 변동이 부족해 `None`이었다.

| Predictor | Spearman | 각 predictor top-3의 평균 drop |
|---|---:|---:|
| `head_full_vocab_kl_fixed_prefix` | 0.6877 | 0.7500 |
| `event_aligned` | 0.4022 | 0.4167 |
| `mean_readout_activation` | 0.0908 | 0.2500 |
| `readout_activation_frequency` | -0.0479 | 0.0000 |

Head와 Event top-3 평균 drop의 관찰 차이는 `0.7500 - 0.4167 = 0.3333`이다.
다만 이 문서는 predictor 간 상관 차이나 top-k contrast의 유의성을 주장하지
않는다. Exact shared-bootstrap 불확실성은 백업된 `analysis/analysis.json`과
`predictor_comparison.csv`를 함께 보고 판단해야 한다. 이 pilot에서 Head-KL는
Event score보다 **더 큰 관찰 point estimate**를 보였다.

### 5.2 feature별 관찰 효과

| Feature | 4-pair 평균 signed drop |
|---:|---:|
| 7024 | 1.00 |
| 24830 | 1.00 |
| 13724 | 0.25 |
| 18471 | 0.25 |
| 25653 | 0.25 |
| 30729 | 0.25 |
| 나머지 10개 | 0.00 |

Event ranking도 무효하지 않았다. Feature 7024는 Event 상위 후보였고
4/4 case에서 성공을 실패로 바꾸었다. 결과는 “Event score는 틀렸다”가
아니라, **Head-KL를 추가하면 행동적 중요도 선별이 더 좋아질 가능성**을
보였다.

`0.25`인 네 feature의 bootstrap CI는 모두 `[0, 0.5]`로 0과 호환되는 약한
신호다. `0.00`인 열 feature도 “이 4 pairs에서 변화를 감지하지 못했다”는
뜻이지, 모집단에서 효과가 없다는 증거가 아니다.

## 6. 해석할 때 반드시 말해야 할 한계

- LIBERO-Spatial task 0·1만 사용했다.
- feature당 evaluation pair가 4개밖에 없고, 32,768개 중 16개만 rollout했다.
- discovery/evaluation overlap이 `true`이고 evaluation label도 이전에 사용됐다.
- Event와 Head는 `suite/task/trial` episode identity는 맞춰지만 initial-state byte hash로
  동일성을 증명하지 못했고, Event의 전체 timestep과 Head의 episode당 8-step
  sampling도 다르다.
- 평가 모집단이 predictor top-k 합집합이라 selection bias가 있다.
- `[1, 1]`, `[0, 0]` bootstrap CI는 4개 결과가 모두 같아서 생긴
  degenerate interval이지 모집단에 대한 확실성이 아니다. 4개가 모두 양의
  효과일 때 `p=0.125`는 4 pairs를 독립으로 보고 selection, label reuse, 2-task
  clustering, 16-feature multiplicity를 무시한 naive two-sided sign-test 값일 뿐이다.
- verified action decoding metadata가 없어 지표는 action-space KL가 아니라
  fixed-prefix **full-vocabulary KL**다. Non-action logit 변화도 포함될 수 있다.
- Spearman 차이에 대한 CI/검정은 아직 없다. 전체 feature 회수율이나 일반적
  우월성을 주장하면 안 된다.
- Random audit sample의 Spearman이 정의되지 않아 순위 관계를 독립적으로
  확인해 주지 못했다.

현재 안전한 결론은 다음이다.

> 작은 overlapping pilot에서 output-head KL가 Event score보다 행동적 효과 예측에서
> 더 큰 관찰 point estimate를 보였으며, 이 차이를 held-out task에서 재검증해야 한다.

## 7. 남은 작업

1. 현재 ranking, top-k, 설정, 분석 protocol을 동결한다.
2. discovery에 쓰지 않은 Spatial task 2–3 또는 2–5와 새 initial state를 쓴다.
3. Top-k enrichment를 볼 목적이면 Head/Event top-3 합집합과 control을 사전 고정한다.
   전반적 rank correlation/예측자 우월성을 보려면 이와 별도로 충분히 큰
   prespecified random feature panel을 두어야 한다.
4. 임의로 정한 “최소 40”을 쓰지 말고, feature condition당 필요 paired episode 수를
   최소 의미 paired success difference, discordant-pair rate, multiplicity로 사전 power 계산한다.
5. 각 raw-vs-feature 이진 paired 비교에는 McNemar를, Head-vs-Event top-k contrast에는
   shared-state/task-stratified bootstrap을 사용한다. 고정된 tasks 밖으로 일반화하려면
   task도 cluster로 resample/model해야 한다.
6. 가능하면 verified action-token decoding 지표를 추가한다.
7. `9cadd10` plan-hash JSON round-trip bug를 source와 regression test에서 영구 수정한다.

## 8. 빠른 오류 대응표

| 증상 | 원인/대응 |
|---|---|
| 공용 config audit가 blocked | 경로가 `null`인 정상 안전장치 |
| `nano: command not found` | `sed`, Python, setup CLI 사용 |
| `Invalid control character` | JSON string 안 literal newline; `json.dump` 재생성 |
| `processing_prismatic.py` 없음 | pinned `openvla/openvla-7b` remote code를 cache에 download |
| dependency version warning | warning만으로 판단하지 말고 실제 parity 결과 확인 |
| Event score scope mismatch | task/episode 모집단을 맞추고 exact adapter 사용 |
| 새 output root에서 score 차단 | 해당 root/fingerprint로 `validate` 재실행 |
| approved plan hash empty | `PLAN`, `PLAN_HASH`, `CODE_REV`를 출력해 확인 |
| `PLAN_VALID=False` at `9cadd10` | known integer/string-key bug인지 확인 후 이력적 workaround |
| `EGL_NOT_INITIALIZED` at exit | `status: completed`/72 results 후 cleanup warning인지 확인 |
| checksum upload file 없음 | `sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"` 먼저 실행 |

## 9. Git에 넣지 말아야 할 것

- Hugging Face access token
- OpenVLA 15 GB snapshot
- dense activation shards
- SAE checkpoint의 중복 복사본
- `runtime.pt`, readout cache, rollout trajectory, result tarball
- `configs/local/*.yaml`

Git에는 이 runbook과 코드만 넣고, 대용량 실험 artifact는 pinned 원본
저장소와 private result archive에서 복구한다.
