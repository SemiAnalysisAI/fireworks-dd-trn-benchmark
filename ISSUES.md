# Issues encountered

Every error and blocker hit while running Qwen3.6-35B-A3B GRPO on MiMo coding
tasks with the Fireworks Dedicated Training API, the fw-ai/cookbook
`harbor_rl` recipes, Harbor and E2B, 2026-10-27 to 2026-10-28.

Runs: smoke run `mimo-qwen3p6-rl-001` (4 steps, 2026-10-27) and
`qwen3p6moe-mimorl-001` (20 steps, 2026-10-27/28). Full timelines are in
`fireworks-training-runs/<run>/run.md`; the write-up for Fireworks is
`fireworks-training-runs/qwen3p6moe-mimorl-001/fireworks-issues.md`.

Environment: cookbook `57968a2237b663be33078d2747a28463983ae134`, fireworks SDK
1.2.30, harbor 0.21.0, e2b 2.55.0, Python 3.12.

## Summary

| # | Issue | Owner | Impact | Status |
|---|---|---|---|---|
| 1 | B300 training shape's generator blocked by a serving-version blocklist | Fireworks | Run could not start on B300; moved to B200 | Open, report to Fireworks |
| 2 | Async loop hangs after the generator deployment returns 4xx | Fireworks cookbook | Trainer job left PENDING until a manual stop | Open, report |
| 3 | Generator returned HTTP 503 in two bursts mid-run | Fireworks | ~12 h of rollout time in failed calls; 31-42% of rollout time in 4 versions | Open, report |
| 4 | Trainer GPU time and generator request metrics not reported | Fireworks | MFU only as a client-side lower bound; no server-side throughput | Open, report |
| 5 | Evals block the next weight sync until their slowest trial finishes | Fireworks cookbook (design) | 67 min of stalled training, ~$88 | Mitigated: 3 attempts per eval |
| 6 | Watchdog/SIGINT stop skips the final save and promotion | Fireworks cookbook (design) | A budget stop loses the final model | Worked around: raised cap |
| 7 | Sidecar venv built on the task image's old python | Fireworks cookbook | E2B template builds fail for many MiMo tasks | Fixed in our `bake()` |
| 8 | Template prebuild treats failed templates as built | Fireworks cookbook | Relaunch skips broken templates; sandboxes fail with GPUs billing | Fixed in our `train()` |
| 9 | Concurrent trials race to build the same E2B template | Fireworks cookbook / E2B | Every trial failed ("build was cancelled") | Fixed: prebuild before fan-out |
| 10 | Renderer name `qwen3_6` has no TITO certification | Fireworks cookbook | Every rollout failed in the sidecar | Fixed: `qwen3_6_preserved` |
| 11 | No per-step metrics without `COOKBOOK_METRICS_FILE` | Fireworks cookbook | Smoke run kept no metrics ledger | Fixed: env var required in AGENTS.md |
| 12 | Model promotion timed out (smoke run) | Fireworks | Smoke model not retained | Fixed: set `output_model_id` |
| 13 | E2B Hobby tier rejects Harbor's 24 h sandbox timeout | E2B / Harbor | Every sandbox create failed | Fixed: E2B Pro |
| 14 | E2B template build "internal error" | E2B | 2 tasks cannot be used | Replaced tasks |
| 15 | E2B base provisioning fails on some images' apt state | E2B / dataset | 3 tasks cannot be used | Replaced tasks |
| 16 | Sandboxes lost or killed mid-rollout | E2B | 5 rollouts failed | Accepted (0.5%) |
| 17 | Model output truncated mid tool call | Config / model | 30% of trajectories discarded at 2,048 tokens; 15% of rollouts at 8,192 | Mitigated: 8,192 tokens |
| 18 | Rollouts exhaust the 65,536-token context | Config / model | 121 rollouts (13%) end at the context limit | Accepted |
| 19 | Hugging Face rate limit on the full dataset snapshot | Hugging Face | Could not download all 2,698 tasks | Worked around: split tasks only |
| 20 | Local machine setup (read-only `HF_HOME`, no `caffeinate`) | Our environment | Setup errors | Worked around |
| 21 | Budget watchdog bugs (restart accounting, stall window) | Our tooling | ~$12 unaccounted; would have killed a healthy run | Fixed |

## Fireworks platform

### 1. B300 shape generator blocked by serving version (blocking)
- **Error** (`POST /v1/accounts/<acct>/deployments`, HTTP 400, 2026-10-27T22:06:59Z):
  `... Dynamo frontend session-affinity headless Service suffix exceeds the
  Kubernetes 63-character name limit ... fixed by ... 7.70.0 ... "7.56.0" is
  within blocked range [7.55.0, 7.70.0)`.
- **Cause**: the latest validated version of training shape
  `accounts/fireworks/trainingShapes/qwen3p6-35b-a3b-262k` links generator
  shape version `rft-qwen3p6-35b-a3b-full-rdma-b300-bf16-w1-p1/versions/f1dttnuc`,
  which resolves to serving 7.56.0. The newer `loskjzvh` resolves to 7.67.0,
  also blocked. The block is independent of deployment ID length (2-51
  characters tried), and no customer setting changes the serving image.
- **Impact**: the same shape worked in the smoke run that morning. The trainer
  job is created first, so it is left with no generator.
- **Workaround**: training shape `qwen3p6-35b-a3b-256k` (4x B200 trainer, 2x
  B200 generator, ~$78/hr list). Benchmark numbers are B200, not B300.
- **Check before launch, for free**: `POST /deployments?deploymentId=probe&validateOnly=true`
  with the shape's `deploymentShape`; a "hot load bucket url is required" 400
  means it passed the version check.
- **Ask**: publish a validated 262k shape whose generator is on serving 7.70.0
  or later, and reject blocked shapes before creating the trainer.

### 2. Async loop hangs after the deployment 4xx
- **Symptom**: after the 400 above, `async_rl_loop` did not exit; trainer job
  `training-api-service-5310faf5` sat in `JOB_STATE_PENDING` until we sent
  SIGINT (then it was deleted; ~$1).
- **Ask**: fail fast and release the trainer when deployment creation returns 4xx.

### 3. Generator HTTP 503 bursts
- **Symptom**: model calls returned `Transient HTTP 503` and were retried 7
  times, then failed (`tito_server_http_error`). 247 calls failed this way,
  with 2,001 retry log lines, in two windows: 00:45-00:59 and 02:00-02:19 UTC
  on 2026-10-28. Deployment `qwen3p6-35b-a3b-1793142132`.
- **Impact**: 43,800 s (~12 h) of training-rollout time spent in failed calls,
  11% of all rollout time; in rollouts started on model versions 6, 7, 13 and
  14 it was 35-50% of inference time. Only 43 of the 247 failed calls
  overlapped a weight sync, so hot-loading is not the main cause.
- **Status**: cause unknown. Report with the deployment ID and time windows.

### 4. Missing server-side telemetry
- `perf/trainer_fwd_bwd_time`, `perf/trainer_optim_step_time` and
  `perf/trainer_busy_walltime_pct` were never returned by the trainer, so MFU
  can only use client-measured run time (includes queueing): 4.6-7.3% is a
  lower bound.
- `sampler/*` (time to first token, prefill/decode queue, cache hit rate,
  generation tokens/s) is not logged for custom rollouts that go through the
  TITO sidecar. The per-turn server metrics are inside each trial's TITO
  artifact, but not in the metrics stream.
- `perf/weight_sync_capture_time` / `hotload_time` only exist for RDMA
  publication; weight sync took 106-186 s every step (more than fwd/bwd) with
  no breakdown.

### 5. Evals block weight sync (design)
- The loop cannot publish new weights while an eval is running, so each eval
  stalls training until its slowest trial ends (agent cap 2,400 s + verifier
  600 s). With 8 attempts per task the 4 mid-run evals stalled training for
  10-23 min each, 67 min in total (~$88 of GPU time), and eval used 30% of
  sandbox time.
- **Mitigation**: 3 attempts per task (`eval_completions_per_prompt=3`). The
  setting is read at startup, so it cannot be changed in a running job.

### 6. A stopped run loses the final model (design)
- `async_rl_loop` only does the final save and `promote_latest` after
  `run_training` returns normally. A SIGINT from the budget watchdog leaves
  only the last periodic DCP checkpoint. We raised the cap from $500 to $650
  rather than risk that; the run finished at ~$385.

## Fireworks cookbook (`training/examples/harbor_rl`)

### 7. Sidecar venv uses the task image's python
- **Where**: `tito/prepare_tasks.py`, `_SIDECAR_PYTHON_INSTALL`.
- **Error** (E2B template build, step 4): `AssertionError: 3.8.20` (task
  000371), `AssertionError: 3.7.17` (000144), and for 000411 (python 3.10,
  passes the `>= 3.10` assert) `No matching distribution found for numpy==2.4.6`.
- **Cause**: the venv is built with the first `python3` on PATH; many MiMo
  images ship an older `/usr/local/bin/python3`. The pins need python 3.11.
- **Fix**: `fix_sidecar_python()` in `bake()` uses the image's python3 if it
  is 3.11 or newer, otherwise apt's `/usr/bin/python3`, and asserts 3.11.

### 8. Prebuild treats failed templates as built
- **Where**: `tito/e2b_templates.py`. "Exists" is `alias_exists()`, which is
  true for a template whose build errored or never started.
- **Impact**: after one failed prebuild, a relaunch skips the broken templates
  and every sandbox for those tasks fails mid-run with GPUs billing. Also,
  `asyncio.gather` aborts the whole prebuild on the first failure while other
  builds keep running server-side.
- **Fix**: `train()` checks each template's real E2B `buildStatus` and aborts
  before creating a trainer if any is not ready.

### 9. Template build race (smoke run)
- **Error**: `BuildException: build was cancelled` (non-retryable) on every
  trial. Concurrent trials of one task raced to build the same
  content-addressed template.
- **Fix**: `prebuild_e2b_templates` before fan-out. Cost: ~$25.

### 10. Renderer certification (smoke run)
- **Error**: `ValueError: renderer 'qwen3_6' has no production TITO certification`
  on every rollout. `qwen3_6` builds fine in the supervised path but the TITO
  sidecar requires a certified renderer.
- **Fix**: `qwen3_6_preserved` (certification
  `qwen3.6-35b-a3b-preserved@995ad96e-candidate`). Cost: ~$15.

### 11. No metrics ledger by default (smoke run)
- Without `COOKBOOK_METRICS_FILE` (or W&B), the loop writes no per-step
  metrics. Now required in AGENTS.md, together with `COOKBOOK_TRACE_FILE`.

### 12. Promotion timeout (smoke run)
- Promotion of the smoke model timed out at 90 s and the in-run model was not
  kept after cleanup. Setting `output_model_id` fixed it for the 20-step run
  (`qwen3p6moe-mimorl-20step`, READY).

## E2B

### 13. Hobby tier vs Harbor's sandbox timeout (smoke run)
- **Error**: `400: Timeout cannot be greater than 1 hours` on every sandbox
  create. Harbor 0.21.0 hardcodes `timeout=86_400`; Hobby caps at 1 h and 20
  concurrent sandboxes.
- **Fix**: smoke run patched Harbor to 3,600 s and capped trial budgets; the
  20-step run used E2B Pro with stock Harbor. Cost: ~$10.

### 14. Template build "internal error"
- **Error**: `BuildException: An internal error occurred` while uncompressing
  the base image layer, identical on 3 attempts.
- **Tasks**: `format-code-task-000216` (12 GB image; builds
  `e0bc4050-2eb2-4253-b818-8b646915d178`, `56e2b87c-1983-4088-822a-dd0d808cb41c`,
  `a3fc64e8-86e8-44ab-960a-e4be7a97b875`) and replacement candidate
  `format-code-task-002127` (750 MB image, so not size;
  `99a01a5e-1e2e-4b75-8726-ef4ca5392c9c`).
- **Status**: cause unknown, E2B-side. Tasks replaced in `task-split.json`.

### 15. Base provisioning fails on the image's apt state
These fail in E2B's own provisioning step, before any of our Dockerfile layers.

| Task | Split | Build | Error |
|---|---|---|---|
| `format-code-task-000751` | train | `ceb1be77-7274-4fe2-afd9-b925b69dfe43` | Dead apt source (`storage.googleapis.com/dart-archive`, "does not have a Release file"), so `apt-get update` fails |
| `format-code-task-002974` | train | `552683de-6b33-4e5a-a5c7-b4c496eac9ac` | Debian 11 image with stale apt lists; `openssh-server`/`sudo` at superseded versions, 404 from security.debian.org |
| `format-code-task-001495` | eval | `70e703ae-ffd5-4fb7-922e-29e9bd93702c` | Same as 002974 |

Replaced in `task-split.json` (seeded `random.Random("42-replacements")`):
000216 -> 002521, 000751 -> 002039, 002974 -> 002696, 001495 -> 000876. This is
also worth flagging to the dataset maintainers.

### 16. Sandboxes lost mid-rollout
Five of 960 rollouts in the 20-step run failed on the sandbox side:
- `SandboxUnreachableException` ("peer closed connection without sending TLS
  close_notify"): 000154, 002482 (train), 001022 (eval).
- `TimeoutException` ("Request timed out"): 002144 (eval).
- Agent killed with exit 137 (SIGKILL, likely out of memory in the 8 GiB sandbox): 000459 (train).

## Model and configuration

### 17. Output truncated mid tool call
- **Error**: trajectory discarded with `tito_sdk_parse_error`, parser reason
  `tool_unparsed_boundary`, `finish_reason=length`: a turn hit its output
  limit in the middle of a tool call.
- **At `max_completion_tokens=2048`**: ~30% of trajectories discarded
  (49 of ~159 trials in 10 min; 7 of 8 eval tasks affected). Stopped (~$30)
  and relaunched at 8,192.
- **At 8,192**: 140 of 960 rollouts (87 train, 53 eval) still ended this way;
  81 of the 87 logged had `finish_reason=length` (the output limit shrinks as
  the context fills). Concentrated in a few tasks: 001022 and 002144 (18
  each), 001129, 000768, 001237, 002923.
- **Next**: raise `max_completion_tokens` to 12-16K.

### 18. Context budget exhausted
- 121 rollouts (62 train, 59 eval) used the full 65,536-token context
  (`context_budget_exhausted`). Training rollouts are still trained on
  (truncated); the max sequence length hits the limit in almost every model
  version.

## Our environment and tooling

### 19. Hugging Face rate limit
- The full `FineEnvs/MiMo-V2.6-RL-harbor-code` snapshot hit HTTP 429. Only the
  88 split tasks were downloaded; `data/tasks.jsonl` lists all 2,698.

### 20. Local machine
- System `HF_HOME=/data/models` is read-only: set
  `HF_HOME=$HOME/.cache/huggingface` and unset `HF_HUB_ENABLE_HF_TRANSFER`.
- No `caffeinate` on Linux: launch with `setsid nohup`.

### 21. Budget watchdog
- A restarted watchdog counted only Fireworks resources created after its own
  start, so ~$12 of spend went unaccounted. Fixed with `--since` (count from
  the run's launch time).
- The 60-minute stall rule would have killed a healthy run while an eval held
  weight sync. Raised to 90 minutes.
