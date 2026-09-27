# openpi pipeline

> 보관 문서 — 당시 Baseline 안내다. 현재 Head 연구는 [연구·결과](../experiment.md)와 [실행·복구](../runbook.md)를 따른다.

Closed-loop interpretability pipeline for the π₀.₅ backbone (PaliGemma
vision-language prefix + action expert) on LIBERO simulation suites.

The default commands in this doc reproduce a **5-trial public minimal
sweep** (10 tasks × 5 trials × 3 layers per capture target: AE
{0,5,17}, PG {0,11,16}). Paper-faithful runs use 50 trials and four
layers per target (AE {0,5,11,17}, PG {0,5,11,16}); the included
sweep is a runnable demonstration of the full pipeline, not the
full paper coverage. PG layer 17 is excluded from intervention
sweeps because its post-MLP residual is the final PaliGemma output;
no later action-expert block consumes that edited prefix state, so
these interventions are structural no-ops.

## Pipeline sanity check (5-trial scale)

This is the public minimal sweep (LIBERO-Spatial, 10 tasks × 5
trials = 50 episodes per condition, top-3 features per ranking),
not the paper's 50-trial run. The goal is to verify this codebase
reproduces the qualitative claims of the paper's Table 3
(`tab:causal-results-summary`) at this sample size; small numerical
gaps are expected.

### Hard zero-out (α = 0)

| Layer | source | Baseline | Δ event-aligned | Δ window-mean | Δ task-mean | Δ random-alive |
|---|---|---:|---:|---:|---:|---:|
| AE l00 | paper     | 96.4% | **−96.4** | **−96.4** | **−96.4** | −0.7  |
|        | this repo | 96.0% | **−96.0** | **−96.0** | **−96.0** | −30.0 |
| AE l05 | paper     | 96.4% | **−95.6** | **−96.2** | **−96.2** | −23.4 |
|        | this repo | 98.0% | **−98.0** | **−98.0** | **−98.0** | −78.7 |
| AE l17 | paper     | 96.4% | **−81.2** | **−96.4** | **−96.4** | −7.1  |
|        | this repo |100.0% | **−99.3** | **−100.0**| **−100.0**| −2.0  |
| PG l00 | paper     | 96.8% | −2.2 | −1.4 | −0.9 | −0.7 |
|        | this repo | 98.0% | −0.7 | −0.7 | +0.7 | +0.7 |
| PG l11 | paper     | 96.8% | −2.7 | −2.2 | −2.8 | −1.0 |
|        | this repo | 94.0% | +5.3 | +3.3 | +4.7 | +4.0 |
| PG l16 | paper     | 96.7% | −0.7 | +0.0 | −1.5 | −0.3 |
|        | this repo | 98.0% | +0.7 | −0.7 | −0.7 | +0.0 |

The AE-shallow random-alive deviation is a sample-size artifact: at 5
trials × 10 tasks the alive feature pool is only 75–84 features (≈8%
of the dictionary), so the random sample picks task-relevant
features more often than at the paper's 50-trial scale.

### Soft α-sweep on AE l17 (α = 0.50, only α shared by both grids)

| source    | Δ event-aligned | Δ window-mean | Δ task-mean | Δ random-alive |
|---|---:|---:|---:|---:|
| paper     | −54.6 | −88.6 | −89.3 | +0.0 |
| this repo | −55.3 | −86.7 | −88.7 | −2.0 |

## Installation

openpi runs JAX inference in one process and a LIBERO sim policy in
another, communicating over an `openpi-client` websocket. The two
sides need different Python versions, so two envs are required.

Both envs use [uv](https://docs.astral.sh/uv/) (openpi's chosen
package manager — `pip` does not read `uv.lock`, so it would resolve
git-pinned deps like `lerobot` to incompatible PyPI latests). Install
once if it is not on `PATH`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Define one env var pointing at this repo's checkout. All
installation and pipeline commands below reference `$EVENT_SAE_ROOT`:

```bash
export EVENT_SAE_ROOT=/path/to/event-sae   # adjust to the actual checkout
```

### Step 1: Clone the openpi fork

The openpi fork at `xc-j/openpi-event-sae` is vanilla openpi
plus the in-source SAE collection + intervention hooks needed by the
JAX path. Clone into `external/`:

```bash
cd "$EVENT_SAE_ROOT/external"
git clone https://github.com/xc-j/openpi-event-sae.git
cd openpi-event-sae
git submodule update --init --recursive
```

### Step 2: Main env (py3.11, JAX + torch 2.7.1)

From the fork root, sync the locked dependencies and install the
openpi fork in editable mode:

```bash
cd "$EVENT_SAE_ROOT/external/openpi-event-sae"
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Run scripts in this env with `uv run` from the fork root — no
manual activation needed.

Sanity check:

```bash
cd "$EVENT_SAE_ROOT/external/openpi-event-sae"
uv run python -c "import jax, torch, lerobot.common; print('jax', jax.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
# expect: jax 0.5.3 torch 2.7.1 cuda True
```

### Step 3: Shared external libraries (same as openvla)

Two libraries from `docs/openvla.md` Step 2 are also used here.
Clone under `external/` at the repo root (already gitignored), then
install into the openpi main env via `uv pip`. Skip the clone if
the openvla half already did it.

**(b) `dictionary_learning`** — SAE training library:

```bash
cd "$EVENT_SAE_ROOT/external"
git clone https://github.com/saprmarks/dictionary_learning.git
cd "$EVENT_SAE_ROOT/external/openpi-event-sae"
uv pip install -e ../dictionary_learning
```

**(c) AWE** — kinematic keyframe extraction (fork with packaging
fixes; see the fork's NOTICE):

```bash
cd "$EVENT_SAE_ROOT/external"
git clone https://github.com/xc-j/awe.git
cd "$EVENT_SAE_ROOT/external/openpi-event-sae"
uv pip install -e ../awe

# AWE imports robosuite.utils.transform_utils for quaternion math; this
# pulls robosuite + mujoco (~500 MB on disk) but no GL is exercised.
uv pip install robosuite==1.4.0
```

**(d) Gemini SDK** — event-cluster annotation:

```bash
uv pip install google-genai
```

### Step 4: LIBERO sim env (py3.8, robosuite + torch 1.11+cu113)

The LIBERO sim has incompatible deps with the main env, so it gets a
separate venv. Recipe from `external/openpi-event-sae/examples/libero/README.md`:

```bash
cd "$EVENT_SAE_ROOT/external/openpi-event-sae"

uv venv --python 3.8 .venv-libero
source .venv-libero/bin/activate

uv pip sync \
    examples/libero/requirements.txt \
    third_party/libero/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    --index-strategy=unsafe-best-match

uv pip install -e packages/openpi-client
uv pip install -e third_party/libero

# Required when running LIBERO sim scripts:
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
```

Sanity check:

```bash
python -c "import torch, robosuite, libero; print('torch', torch.__version__, 'robosuite', robosuite.__version__)"
# expect: torch 1.11.0+cu113 robosuite 1.4.1
```

Unless marked as **LIBERO sim client**, commands in the rest of
this doc run in the openpi main env (Step 2).

## Usage

The pipeline mirrors [docs/openvla.md](openvla.md) Phase 1–4 (collect →
train SAE → keyframes → events → score → rank → intervene), with
openpi-specific activation collection going through the JAX
`io_callback` hook in `external/openpi-event-sae/src/openpi/sae_collection/`.

Examples below assume the dense-collection run dir
`logs/openpi/sae_collection/$RUN` (under this repo) and the
LIBERO-Spatial / action-expert capture target. PaliGemma uses the
same flow with the companion run dir (capture target swap is
server-side).

The server has to run with `external/openpi-event-sae` as its cwd
(for uv) while writing outputs back under this repo. Server-side
paths use absolute `$EVENT_SAE_ROOT/...`; client and other pipeline
commands are run from `$EVENT_SAE_ROOT`, so relative `logs/...`
paths resolve there.

## Phase 1 — SAE training

### (a) Collect openpi activations during a LIBERO rollout

The server (openpi main env, JAX) and the LIBERO sim client
(LIBERO sim env, Python 3.8) run in separate processes. The server
is suite-agnostic — the suite (`libero_spatial`,
`libero_object`, `libero_goal`, `libero_10`) is picked client-side
by the `--config` YAML. Action-expert and PaliGemma run as two
separate sessions, one capture target per server.

**Terminal 1 — server** (openpi main env). Run from the fork root
with `uv run`, or activate `external/openpi-event-sae/.venv`
manually beforehand:

```bash
cd "$EVENT_SAE_ROOT/external/openpi-event-sae"
uv run python "$EVENT_SAE_ROOT/scripts/openpi/serve_policy.py" \
    --env libero \
    --mode dense \
    --capture-target action_expert \
    --layer-indices 0,5,17 \
    --output-root "$EVENT_SAE_ROOT/logs/openpi/sae_collection" \
    --run-name libero_spatial_ae_l0_5_17 \
    --port 8000
```

For PaliGemma, swap `--capture-target paligemma --layer-indices
0,11,16` (and pick a different `--run-name`).

**Terminal 2 — LIBERO sim client** (LIBERO sim env). After the
server prints `Enabled SAE collection: ...`, activate the sim venv
and run the client from this repo root. The `--config` chooses the
suite:

```bash
cd "$EVENT_SAE_ROOT"
source external/openpi-event-sae/.venv-libero/bin/activate
export PYTHONPATH=$PYTHONPATH:$EVENT_SAE_ROOT/external/openpi-event-sae/third_party/libero

python scripts/openpi/eval_libero.py \
    --config configs/examples/openpi/eval_libero_spatial.yaml \
    --override server.port=8000 \
    --override env.num_trials_per_task=5 \
    --override sae_collect.enabled=true \
    --override sae_collect.capture_target=action_expert \
    --override "sae_collect.layer_idxs=[0,5,17]" \
    --libero-root external/openpi-event-sae/third_party/libero
```

Output: `logs/openpi/sae_collection/<run_name>/` with
`sae_activations/post_mlp_residual{,__paligemma}/layer_NN_shard_*.pt`,
`activation_index.jsonl`, `trajectory_records.jsonl`, `videos/`, plus
`run_metadata.json` and `success.csv`.

### (b) Train one BatchTopK SAE per (target, layer)

One python invocation per `(capture_target, layer)` pair, e.g. for
action-expert layer 17:

```bash
python scripts/train_sae.py \
    --config configs/examples/openpi/train_sae_libero_spatial.yaml \
    --save-dir logs/openpi/sae/libero_spatial_action_expert_l17 \
    --override data_dir=logs/openpi/sae_collection/<ae_run_name>/sae_activations/post_mlp_residual \
    --override layer_idx=17 \
    --override activation_dim=1024 \
    --override dict_size=1024 \
    --override submodule_name=post_mlp_residual \
    --override run_tag=libero_spatial_action_expert_l17
```

Repeat for the other five `(target, layer)` pairs (AE l00/l05/l17 +
PG l00/l11/l16). PaliGemma uses `activation_dim=dict_size=2048` and
`submodule_name=post_mlp_residual__paligemma`.

Output: `logs/openpi/sae/libero_spatial_<target>_l<NN>/trainer_0/ae.pt`
(+ `config.json` and intermediate checkpoints).

Or skip step (b) and use the paper's six pre-trained SAEs
(three AE layers + three PG layers, BatchTopK k=64) from the
[Hugging Face Hub](https://huggingface.co/mr-cabbage/event-sae-openpi-libero).
For each `(target, layer)` pair you operate on, set `$SAE_CKPT` to
that pair's checkpoint — either the HF download or the local
training output:

```bash
# Pretrained, e.g. action_expert layer 17:
SAE_CKPT=$(hf download mr-cabbage/event-sae-openpi-libero action_expert_l17/ae.pt)

# Or locally trained:
SAE_CKPT=logs/openpi/sae/libero_spatial_action_expert_l17/trainer_0/ae.pt
```

Reset `$SAE_CKPT` for each pair as you sweep through them. All
subsequent commands in this doc that need a checkpoint reference
`--sae-checkpoint $SAE_CKPT`.

## Phase 2 — Kinematic keyframe extraction

Keyframes are computed from the rollout trajectory only, not from
the SAE activations, so a single keyframe set is reused for scoring
features from every layer of every capture target. Use either the
AE or the PG run's `trajectory_records.jsonl` — by convention the
AE collection (the first stream collected) supplies the events
used downstream. Substitute `$RUN` below with that collection's
run name (the `--run-name` chosen in step (a)).

### (c) Extract AWE kinematic keyframes from rollout trajectories

CPU-only.

```bash
RUN=<your_ae_collection_run_name>
python scripts/extract_keyframes.py \
    --trajectory-records-path logs/openpi/sae_collection/$RUN/trajectory_records.jsonl
```

Output: `waypoint_summary.json` under
`logs/openpi/keyframes/$RUN/dp_pos_only_err0p05/`.

## Phase 3 — Event clustering with VLM annotation

### (d) Render 5-frame bundles around each keyframe

```bash
python scripts/extract_keyframe_media.py \
    --waypoint-summary-path logs/openpi/keyframes/$RUN/dp_pos_only_err0p05/waypoint_summary.json
```

Outputs under `logs/openpi/events/$RUN/samples_5frames_stride2/`: PNG
frames + clip MP4s + `samples.jsonl`.

### (e) Build vision embeddings + state vectors per sample

Requires GPU.

```bash
python scripts/build_event_features.py \
    --samples-path logs/openpi/events/$RUN/samples_5frames_stride2/samples.jsonl
```

Output: `event_features.jsonl` next to `samples.jsonl`.

### (f) Task-local agglomerative clustering

CPU-only.

```bash
python scripts/cluster_events.py \
    --event-features-path logs/openpi/events/$RUN/samples_5frames_stride2/event_features.jsonl
```

Outputs under `clusters/` next to `event_features.jsonl`:
`cluster_assignments.jsonl`, `clusters.jsonl`, `summary.json`.

### (g) Annotate clusters with Gemini

```bash
export GEMINI_API_KEY=<your-key>
python scripts/annotate_clusters.py \
    --clusters-path logs/openpi/events/$RUN/samples_5frames_stride2/clusters/clusters.jsonl
```

Output: `gemini-2_5-flash_cluster_annotations.jsonl` next to
`clusters.jsonl`. Default model `gemini-2.5-flash` (override with
`--model`); the paper used a stronger Gemini model, and these
labels are descriptive only and do not affect feature ranking or
the intervention results.

## Feature ranking (bridge between Phase 3 and Phase 4)

Encode the saved dense activations through each trained SAE, score each
event cluster against the resulting sparse features, and produce a
ranked list of candidate features for Phase 4. Run all three steps
once per (capture target, layer) pair — six pairs for libero_spatial
(`{action_expert} × {0, 5, 17}` ∪ `{paligemma} × {0, 11, 16}`). All
six are scored against the same AE-derived event clusters (see
Phase 2 note).

### (h) Top-k SAE encoding (offline)

Apply each trained SAE to its own dense shards and write sparse top-k
shards. CPU- or GPU-friendly (auto-detects).

```bash
TARGET=action_expert     # or paligemma
LAYER=17                 # AE: {0,5,17}; PG: {0,11,16}
DENSE_RUN=<run name from step (a)>          # AE run for action_expert, PG run for paligemma
TAG=${TARGET}_l$(printf '%02d' ${LAYER})

python scripts/extract_topk.py \
    --dense-dir logs/openpi/sae_collection/${DENSE_RUN} \
    --sae-checkpoint $SAE_CKPT \
    --layer-idx ${LAYER} \
    --output-dir logs/openpi/topk/${TAG}
```

Output: per-pair `shard_*.pt` (sparse top-k rows) + `manifest.json`
under `logs/openpi/topk/<target>_l<NN>/`.

To skip step (h), use the **online top-k mode** in step (a):
re-launch the server with `--mode topk --sae-checkpoint $SAE_CKPT`
(and a single `--layer-indices NN`) and the rollout writes sparse
`token_topk_sparse_v1` shards directly into the run dir. Requires
an SAE checkpoint already available (from step (b) or the Hugging
Face Hub).

### (i) Event-feature score matrix

For each pair, score every SAE feature against the AE-derived event
clusters (max projection onto pulse / step-up / step-down templates
in a ±5-step window around each event, averaged across episodes).
CPU-only.

```bash
AE_RUN=<your_ae_collection_run_name>
EVT=logs/openpi/events/${AE_RUN}/samples_5frames_stride2

python scripts/score_cluster_features.py \
    --topk-run-dir logs/openpi/topk/${TAG} \
    --event-features-path ${EVT}/event_features.jsonl \
    --cluster-assignments-path ${EVT}/clusters/cluster_assignments.jsonl \
    --cluster-annotations-path ${EVT}/clusters/gemini-2_5-flash_cluster_annotations.jsonl \
    --prompt-records-path logs/openpi/sae_collection/${DENSE_RUN}/prompt_records.jsonl \
    --output-path logs/openpi/scores/${TAG}.pt
```

`--prompt-records-path` is required for paper-faithful
`matrix_task_mean` (per-task mean over **all** rollout timesteps in
the task, not only event-window timesteps). Omit it only if you
explicitly want the approximate event-only fallback.

Output: one `.pt` payload per pair — three matrices
(`matrix_raw`, `matrix_window_mean`, `matrix_task_mean`) plus
`row_keys`, `selected_events`, `templates`, `source`.

### (j) Build candidate feature lists

Surface the top-K features under four ranking strategies. CPU-only.
For π₀.₅ the paper uses K = 3. The script reads pre-computed matrices
from the score artifact; `--topk-run-dir` is only used by the
`random_alive` ranking (for the alive-feature scan).

```bash
python scripts/build_feature_rankings.py \
    --scores-pt logs/openpi/scores/${TAG}.pt \
    --topk-run-dir logs/openpi/topk/${TAG} \
    --output-dir logs/openpi/rankings/${TAG} \
    --top-k 3
```

Output: `event_aligned.jsonl`, `window_mean.jsonl`, `task_mean.jsonl`,
`random_alive.jsonl`, and `candidates.jsonl` (flat `4 × K` rows that
feed Phase 4 intervention).

## Phase 4 — Closed-loop intervention

Edit one SAE feature at inference time and check how the policy's
success rate changes. For selected feature `i`, feature-scaling
factor `α_f`, and reconstruction mix `α`:

    z'_i = α_f · z_i      # selected feature, scaled
    z'_j = z_j            # all other features unchanged (j ≠ i)
    x'   = x + α · (Dec(z') − Dec(z))

`α_f = 0` is the paper's hard zero-out (feature i is removed);
`α_f ∈ (0, 1)` is a soft suppression that the paper calls
"dose-response"; `α_f > 1` boosts; `α = 1` keeps the standard SAE
reconstruction. The server-side hook lives in
`external/openpi-event-sae/src/openpi/sae_collection/reconstruction.py`
and is enabled by `scripts/openpi/serve_policy.py --mode intervene`.

### (k) Run a single-feature intervention on LIBERO

Repeat the server-then-client pair once per `feature_id` in
`candidates.jsonl` — extract them with
`jq -r '.feature_id' candidates.jsonl`.

Start the server in intervention mode (one terminal, openpi main
env, run from the fork root):

```bash
cd "$EVENT_SAE_ROOT/external/openpi-event-sae"
uv run python "$EVENT_SAE_ROOT/scripts/openpi/serve_policy.py" \
    --env libero \
    --mode intervene \
    --sae-checkpoint $SAE_CKPT \
    --capture-target action_expert \
    --layer-idx 17 \
    --feature-indices 24830 \
    --feature-alpha 0.0 \
    --recon-alpha 1.0 \
    --port 8000
```

Run the LIBERO client (other terminal, LIBERO sim env). Disable
`sae_collect.enabled` so the client does no per-request context
plumbing — the server applies the intervention transparently:

```bash
cd "$EVENT_SAE_ROOT"
source external/openpi-event-sae/.venv-libero/bin/activate
export PYTHONPATH=$PYTHONPATH:$EVENT_SAE_ROOT/external/openpi-event-sae/third_party/libero

python scripts/openpi/eval_libero.py \
    --config configs/examples/openpi/eval_libero_spatial.yaml \
    --override server.port=8000 \
    --override env.num_trials_per_task=5 \
    --override sae_collect.enabled=false \
    --override "logging.root_dir=$EVENT_SAE_ROOT/logs/openpi/intervene/single_feature" \
    --libero-root external/openpi-event-sae/third_party/libero
```

Output: `logs/openpi/intervene/single_feature/EVAL-*/success.csv`
with one row per (episode, success). For each ranking, take the
mean of `SR_hook − SR_baseline` across its K features — this is
how much zeroing that ranking's features hurts the policy. Repeat
the server-then-client invocation once per `feature_id` in
`candidates.jsonl` to fill the full per-pair ΔSR table.

## Frozen environment snapshot

`environment-openpi.lock.yml` is a pinned record of the conda + pip
package versions on our working machine. It is **not a working
installer** — editable external repos (the openpi fork,
`dictionary_learning`, AWE) are not included. Use it to cross-check
versions when `uv sync` produces a different env than expected.
