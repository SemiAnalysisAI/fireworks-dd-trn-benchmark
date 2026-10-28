# Run qwen3p6moe-mimorl-001

created: 2026-10-27
predecessor: mimo-qwen3p6-rl-001 (smoke, 4 steps, mean reward 0.60)
status: approved, launching

## Plan
- account: kimbo-semianalysis
- method: agentic RL (GRPO, server-side) via dedicated async_rl_loop
- base_model: accounts/fireworks/models/qwen3p6-35b-a3b (READY at preflight)
- training_shape: accounts/fireworks/trainingShapes/qwen3p6-35b-a3b-262k (4x B300 trainer,
  generator rft-qwen3p6-35b-a3b-full-rdma-b300-bf16-w1-p1/versions/f1dttnuc, 1x B300; same as smoke)
- renderer: qwen3_6_preserved (TITO cert qwen3.6-35b-a3b-preserved@995ad96e-candidate, re-verified)
- data: FineEnvs/MiMo-V2.6-RL-harbor-code, committed task-split.json (seed 42): 80 train / 8 eval.
  Only the 88 split tasks were downloaded (full snapshot hit HF 429 rate limits); bake unaffected.
- scale: 20 optimizer steps = 80 train tasks / 4 groups per step, epochs 1, 8 completions per
  prompt (32 rollouts/step), lr 1e-6 full-param, max_seq_len 65536, max_completion_tokens 2048,
  max_head_offpolicy_versions 1, DCP checkpoint at step 10, holdout eval every 5 steps (+start/end)
- sandboxes: E2B Pro (24h sessions, 100 concurrent), 2 vCPU / 8 GiB per task, max 64 concurrent
  trials, agent cap 2400s / verifier cap 600s (mimo-trial-config.yaml)
- output_model_id: qwen3p6moe-mimorl-20step (name verified free at preflight)
- metrics: local only (no W&B). COOKBOOK_METRICS_FILE + COOKBOOK_TRACE_FILE set; trials dir
  retained; budget_watchdog.py samples E2B concurrency every 60s.
- environment: cookbook @ 57968a2237b663be33078d2747a28463983ae134 (= upstream HEAD at setup),
  venv python 3.12, fireworks 1.2.30, harbor 0.21.0 (stock 86400s e2b timeout), e2b 2.55.0.
  HF_HOME overridden to ~/.cache/huggingface (system /data/models is read-only).
  Launched with setsid+nohup on a Linux server (no caffeinate; machine does not sleep).

## Cost
- Fireworks: $75/hr list while up (trainer $60 + generator $15). Estimate 3-5h -> $225-375.
  Worst case (trials hitting the 2400s cap) ~13h -> ~$1,000, hence the cap.
  **Hard cap $500**: watchdog SIGINTs at $475 accrued, leaving ~$25 for teardown.
- E2B: $0.23/sandbox-hr (2 vCPU x $0.0504 + 8 GiB x $0.0162). ~1,150 sandbox sessions
  (640 train + 320 eval + ~10% retries + 88 template prebuilds) x ~10 min -> **~$45 estimate**.
  **Cap 1.5x = $68**: watchdog SIGINTs at $65 accrued. Excludes the $150/mo Pro base fee.
- Stall rule: once trials start, no new optimizer step for 60 min -> stop (60 rather than the
  40 first proposed: one legit step can approach ~55 min when a trial runs to the agent+verifier caps).

## Approval
2026-10-27, user, verbatim:
> 1. do max $500 for fireworks, and sth reasonable, 1.5x your estimated cost for e2b
> 2. yes permission to stop
> 3. local, no w&b
> 4. qwen3p6moe-mimorl-001 and  qwen3p6moe-mimorl-20step
> 5. yes

(1 = cost caps, 2 = agent may stop a stalled/over-budget run without asking, 3 = no W&B,
4 = run id / output model name, 5 = account kimbo-semianalysis confirmed.)

## Launch
launched_at: 2026-10-27T21:19:41Z
train_pid: 1871408 (setsid/nohup; runs/qwen3p6moe-mimorl-001/train.pid)
watchdog_pid: 1871436 (budget_watchdog.py --fw-stop-usd 475 --e2b-stop-usd 65 --stall-min 60)
launch_log: runs/qwen3p6moe-mimorl-001/launch.log
metrics: runs/qwen3p6moe-mimorl-001/metrics.jsonl; trace: client-phase-trace.json; trials: trials/
status: launching (E2B template prebuild for 88 tasks first; no GPU billing until it finishes)

## Incident 1: E2B template build internal error (no GPU spend)
21:21:05Z: during prebuild_e2b_templates (before any Fireworks resource was created), the
build for format-code-task-000216 failed twice with E2B BuildException "An internal error
occurred" while uncompressing the 12 GB base layer of
xiaomimimo/mimo-v2.6-rl-oss@sha256:181a746c... (build ids e0bc4050-2eb2-4253-b818-8b646915d178,
56e2b87c-1983-4088-822a-dd0d808cb41c). asyncio.gather aborted the whole prebuild; train pid
exited 21:21:45Z. Watchdog saw no Fireworks resources -> $0 GPU. 15 other templates were
registered but had no build started. Diagnosing with a 2-task test build (000216 + 000036).
Diagnosis (21:24-21:40Z): not transient. Of the 7 builds that finished before the client
died, 3 ready / 4 error. Three errors were deterministic in the cookbook's sidecar layer
(tito/prepare_tasks.py _SIDECAR_PYTHON_INSTALL): it builds the venv over the first python3
on PATH, and many MiMo images ship an older /usr/local/bin/python3:
  000371: AssertionError 3.8.20; 000144: AssertionError 3.7.17;
  000411: python 3.10 passes the >=3.10 assert but numpy==2.4.6 needs >=3.11.
Smoke run's 5 tasks happened to have >=3.11. 000216's "internal error" while uncompressing a
12 GB base layer is separate (possibly E2B disk limit), retest after the fix.
Second bug: prebuild's "existed" = alias_exists(), true for errored/never-built templates,
so a relaunch would skip them and every sandbox of those tasks would fail.
Fixes (train_mimo_qwen3p6.py): bake() rewrites the sidecar venv to use image python3 if
>=3.11 else apt /usr/bin/python3 (task python3 stays first on PATH); train() checks real
E2B buildStatus after prebuild and aborts before GPU spend if any template is not ready.
Rebaked (content hash changes -> fresh aliases). Prebuilding all 88 templates standalone.

## Template prebuild with fix (21:31-21:55Z)
84/88 ready (incl. 000371/000144/000411, which failed before the sidecar fix). 4 tasks fail
in E2B's own base provisioning step (before any of our Dockerfile layers), so no Dockerfile
fix applies; user decided (verbatim): "Replace them with other tasks / Note down the failed
tasks names and what reasons they failed".

| task | split | image digest | E2B build id | reason |
|---|---|---|---|---|
| format-code-task-000216 | train | sha256:181a746c9ab1... | a3fc64e8-86e8-44ab-960a-e4be7a97b875 | 12 GB base image; "An internal error occurred" while uncompressing the 12 GB layer (failed identically 3 times) |
| format-code-task-000751 | train | sha256:54dfdba6a0f4... | ceb1be77-7274-4fe2-afd9-b925b69dfe43 | image has a dead apt source (storage.googleapis.com/dart-archive, 404 "does not have a Release file"), so E2B provisioning's apt-get update fails |
| format-code-task-002974 | train | sha256:867119366b1f... | 552683de-6b33-4e5a-a5c7-b4c496eac9ac | Debian 11 image with stale apt lists; E2B provisioning installs openssh-server/sudo at superseded versions -> 404 from security.debian.org |
| format-code-task-001495 | eval | sha256:0aa1ad50466b... | 70e703ae-ffd5-4fb7-922e-29e9bd93702c | same as 002974 (stale Debian 11 apt lists, openssh-server/sudo 404) |

Replacement method: pool = dataset tasks (data/tasks.jsonl) not in the split, shuffled with
random.Random("42-replacements"); bake + template-build candidates in that order; first 3 ready
-> train, next ready -> eval. Recorded in task-split.json under "excluded".
Correction: 000216's "internal error" is not explained by image size: replacement candidate
002127 (750 MB image) failed the same way (build 99a01a5e-1e2e-4b75-8726-ef4ca5392c9c). Cause
unknown, E2B-side.

## Task replacement (21:58-22:06Z)
Candidates in seeded order: 002521 ready, 002039 ready, 002127 FAIL (internal error),
002696 ready, 000876 ready, (002523, 000585, 002141 ready, unused).
train: 000216->002521, 000751->002039, 002974->002696; eval: 001495->000876.
task-split.json updated (80/8, "excluded" + "replacement_method" recorded). Rebaked; all 88
templates existed=True and E2B buildStatus ready. Pre-relaunch: no deployments, no trainer
jobs, 0 E2B sandboxes. Relaunching under the same approved plan, caps and watchdog.
relaunched_at: 2026-10-27T22:05:42Z train_pid: 1919627 watchdog_pid: 1919623

## Incident 2: generator deployment blocked by Fireworks (22:06Z), BLOCKED
Templates OK (0 built, 88 existed). Trainer job training-api-service-5310faf5 created
(PENDING), then deployment create -> HTTP 400: "Dynamo frontend session-affinity headless
Service suffix exceeds the Kubernetes 63-character name limit ... fixed ... in 7.70.0 ...
"7.56.0" is within blocked range [7.55.0, 7.70.0)". Process hung after the error; agent sent
SIGINT 22:07:27Z (stop permission); trainer deleted ("Resources released; no further
billing"), exit 22:08:52Z. Spend ~$0-2 (trainer never left PENDING).
Probes (deployments?validateOnly=true, nothing created; verified deployments list empty):
- 262k shape generator f1dttnuc -> 7.56.0 blocked; newer version loskjzvh (2026-10-23,
  latestValidated) -> 7.67.0 blocked. Block is independent of deployment ID length (2-51 chars).
- No user-settable serving image: SDK sends no image tag; extra_values/extra_args need superuser.
- Alternatives passing the version check: qwen3p6-35b-a3b-256k (full-param, numerics validated,
  trainer 4x B200 @ trainer image 0.638.579, generator qxnjq1bd/x3iipdll 2x B200; ~$78/hr list
  at $13/GPU-hr) and qwen3p6-35b-a3b-262k-lora (LoRA trainer; not the approved method).
Awaiting user decision: wait for Fireworks to bump the 262k shape vs switch to B200 shape.

## Decision: switch to B200 shape (22:20Z)
User chose (AskUserQuestion, verbatim option): "Switch to B200 shape now" -- described as
"qwen3p6-35b-a3b-256k: full-param, numerics validated, 4x B200 trainer + 2x B200 generator,
~$78/hr list (vs $75). Same $500 cap. Not smoke-tested: older trainer image (0.638.579) against
current cookbook, and results are B200 numbers, not B300."
User also asked to note the issue for Fireworks: fireworks-issues.md (this directory).
Changes: SETTINGS training_shape_id -> accounts/fireworks/trainingShapes/qwen3p6-35b-a3b-256k
(latest validated version wv47m1sb, trainer image 0.638.579; generator qxnjq1bd/x3iipdll 2x B200).
Cost: $78/hr list ($13/GPU-hr B200 x 6). Estimate 3.5-5.5h (B200 slower than B300) -> ~$270-430.
Cap unchanged: watchdog SIGINT at $475 (rates --trainer-usd-hr 52 --gen-usd-hr 26). E2B cap $65.
relaunched (B200) at: 2026-10-27T22:32:42Z train_pid: 1958436 watchdog_pid: 1958433
trainer_job: accounts/kimbo-semianalysis/rlorTrainerJobs/training-api-service-15e04a0f
deployment: qwen3p6-35b-a3b-1793140441 (CREATING at 22:34:02Z; B200 generator accepted)
22:38Z trainer RUNNING; 22:39Z generator READY; 22:46:58Z initial weight sync done (433.5s, hotload 91s); 22:47:44Z first trials (stall clock armed, 60 min)

## Incident 3: per-call token cap truncation (B200 attempt, 22:47-22:59Z)
B200 shape worked end to end (trainer RUNNING 22:38, generator READY 22:39, initial weight sync
433.5s, first trials 22:47:44). But max_completion_tokens=2048 truncated the model mid tool-call:
by 22:57, 49 trajectories discarded (tito_sdk_parse_error / tool_unparsed_boundary; 48 with
finish_reason=length) of ~159 trials; producer rows_rejected 5 of 13 train rows; 7 of 8 eval tasks
affected (38 eval discards). No optimizer step yet. User chose (verbatim option): "Stop, raise to
8192, relaunch (Recommended)". SIGINT 22:58:48Z, clean teardown (deployment scaled to zero, trainer
deleted) by 22:59Z. Attempt cost: Fireworks ~$30, E2B ~$1.9 (watchdog).
Change: max_completion_tokens 2048 -> 8192 (max_seq_len unchanged 65536). Templates unaffected
(88 existed=True, all ready). Cap accounting is cumulative: watchdog relaunched with
--fw-stop-usd 444 (475 - 31 spent) and --e2b-stop-usd 63.
relaunched (B200, 8192 tok) at: 2026-10-27T23:00:57Z train_pid: 1989766 watchdog_pid: 1989763
trainer_job: training-api-service-400f9e21; deployment: qwen3p6-35b-a3b-1793142132 (23:02:55Z)
23:13:58Z first trials. 23:24Z check: 3 trajectory discards / 63 finished trials (~5%, vs ~30% at 2048), rows_rejected 0/8, rows_completed 5; fw $26.6 e2b $2.2 this attempt.
first optimizer step: 23:35:03Z (21 min after first trials); fw $41 e2b $3.0 this attempt
step 3 at 23:52:08Z; fw $63 e2b $5.5 this attempt

## Watchdog restart (00:38:14Z): eval-aware stall window
Step 5 at 00:07Z, then the step-5 eval blocked weight publishing (producer in_flight 0,
staleness capacity 0; at step 0 eval weight_sync_wait was 620s). Eval trials can run 40 min agent
+ 10 min verifier, so the 60-min stall rule would have killed a progressing run at ~01:07.
Restarted watchdog (old pid 1989789 -> new pid 2088830) with --stall-min 90 --resume and carried-over
spend (--initial-fw-usd 121.88 --initial-e2b-usd 13.57); caps unchanged (fw stop 444, e2b 63).

## Cap raised to $650 total (00:46:20Z)
Evals block weight publishing (~30-60 min each); 20 steps + final eval projected ~$560-600 total.
A watchdog (SIGINT) stop skips the final save + promotion (async_rl_loop.py: final ckpt.save /
promote_latest only after run_training returns normally), leaving only the step-10 DCP checkpoint.
User chose (verbatim option): "Raise cap to $650 total (Recommended)". Watchdog restarted:
--fw-stop-usd 594 (650 - 31 prior attempts - 25 teardown margin), e2b 63 unchanged, stall 90,
carried over fw=$121.88 e2b=$13.71.
Watchdog bug: restarted watchdogs filtered resources by their own start time, so 00:38-00:48Z
Fireworks spend was not accrued (~$12). Fixed (--since); restarted with --since
2026-10-27T23:00:00Z and initial fw=$135.04 (121.88 + gap at $78/hr), e2b=$13.83.

## Eval shots (00:55Z)
User asked: "Just do 3 shot for evals / If you can change that on the fly". Not changeable in the
running process (eval fn built at startup); a restart before the step-10 DCP checkpoint loses all
progress, and after it saves only ~$25-40 on two evals vs ~$20 + untested resume. This run keeps
8-shot evals. SETTINGS gained eval_completions_per_prompt=3 for future runs (file edit does not
affect the running process).
step 10 at ~01:16:49Z (DCP 'step-10' save started); fw $171 e2b $18.5 this attempt
EVAL step 10: reward 0.34 (53/64 completed)
EVAL step 15: reward 0.255 (55/64 completed)

## Outcome: SUCCESS (2026-10-28 03:35Z)
20/20 optimizer steps (03:11:34Z step 20). DCP checkpoints step-10, step-20 saved; sampler
checkpoint step-20-base-6e34ae03 promoted -> accounts/kimbo-semianalysis/models/qwen3p6moe-mimorl-20step
(state READY, kind HF_BASE_MODEL, verified via GET at 03:35Z; operation wp2jpb29).
Holdout eval (8 tasks x 8, reward over completed trajectories):
  step 0 0.277 (47/64) | step 5 0.263 (57/64) | step 10 0.340 (53/64) | step 15 0.255 (55/64) | step 20 0.302 (53/64)
  -> +0.025 vs baseline, within noise for this eval size.
Teardown verified: no deployments, trainer training-api-service-400f9e21 gone, 0 E2B sandboxes.
Cost (watchdog list-price estimate): final attempt Fireworks $349, E2B $35.8.
Run total: Fireworks ~$380 (attempts: $0 + ~$1 + ~$30 + $349), E2B ~$37.7; under the $650 cap
(and the original $500). E2B Pro base fee not included.
status: complete

## Follow-up (2026-10-28): eval attempts per task
This run evaluated 8 attempts per task (pass@8 0.500 -> 0.625, avg@8 0.203 -> 0.266, step 0 -> 20).
Evals stalled training 67 min, used 30% of sandbox time and ~$123 of ~$385. Future runs use
3 attempts per task (`eval_completions_per_prompt=3`, already the SETTINGS default) and report pass@3/avg@3.
