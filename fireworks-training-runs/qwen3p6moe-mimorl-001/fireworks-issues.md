# Issues to report to Fireworks (from run qwen3p6moe-mimorl-001, 2026-10-27)

Account: `kimbo-semianalysis`. Cookbook: fw-ai/cookbook @ `57968a2237b663be33078d2747a28463983ae134`
(training/, fireworks SDK 1.2.30, harbor 0.21.0, e2b 2.55.0).

---

## 1. Validated RL shape `qwen3p6-35b-a3b-262k` cannot create its generator (blocking)

**Summary.** The latest validated version of training shape
`accounts/fireworks/trainingShapes/qwen3p6-35b-a3b-262k` (version `o5kla0at`, trainer image
`0.638.678`) links generator deployment shape version
`accounts/fireworks/deploymentShapes/rft-qwen3p6-35b-a3b-full-rdma-b300-bf16-w1-p1/versions/f1dttnuc`.
Since sometime on 2026-10-27 (between ~14:00 and 22:06 UTC) deployment creation for it fails, because the server now
blocks the serving version it resolves to. The trainer job is created first, so the user is
left with a trainer and no generator.

**Error** (`POST /v1/accounts/kimbo-semianalysis/deployments?deploymentId=qwen3p6-35b-a3b-1793138818`, HTTP 400, 2026-10-27T22:06:59Z):

```
Dynamo frontend session-affinity headless Service suffix exceeds the Kubernetes 63-character
name limit with 39-40 character Helm releases; introduced by
https://github.com/fw-ai/fireworks/pull/62965 and fixed by
https://github.com/fw-ai/fireworks/pull/66505 in 7.70.0. No perf-agent config_map.json entries
use this range; mapping PR not applicable. (mapping.json PR: NA): "7.56.0" is within blocked
range [7.55.0, 7.70.0)
```

**Evidence**
- The same shape and flow created deployment `qwen3p6-35b-a3b-1793124108` successfully
  earlier the same day (~14:00 local, smoke run `mimo-qwen3p6-rl-001`), so the block is new.
- `validateOnly=true` probes (nothing created):
  - `.../rft-qwen3p6-35b-a3b-full-rdma-b300-bf16-w1-p1/versions/f1dttnuc` -> blocked, `"7.56.0"`
  - `.../rft-qwen3p6-35b-a3b-full-rdma-b300-bf16-w1-p1/versions/loskjzvh` (created 2026-10-23,
    `latestValidated: true`) -> blocked, `"7.67.0"`
  - Blocked for every deployment ID length tried (2, 7, 19, 31, 41, 51 chars), although the
    message describes a 39-40 character Helm release condition.
- No customer-side workaround: the SDK sends no serving image tag (training shape snapshot has
  no `deploymentImageTag`), and `extraValues` / `extraArgs` require a superuser key.
- Collateral: the cookbook's `async_rl_loop` hung after the 400 with trainer job
  `accounts/kimbo-semianalysis/rlorTrainerJobs/training-api-service-5310faf5` in
  `JOB_STATE_PENDING`; it only exited on SIGINT (trainer then deleted, ~$1). It should fail fast
  and release the trainer.

**Ask**
1. Publish a validated `qwen3p6-35b-a3b-262k` training shape version whose generator resolves
   to serving >= 7.70.0 (or move `f1dttnuc` / `loskjzvh` onto >= 7.70.0).
2. When a server-side blocklist invalidates a validated shape, surface it at shape resolution /
   trainer creation, before a trainer job is provisioned.
3. Make the async loop fail fast (and clean up the trainer) when deployment creation returns 4xx.

**Workaround we used:** training shape `qwen3p6-35b-a3b-256k` (4x B200 trainer, generator
`deploymentShapes/qxnjq1bd/versions/x3iipdll` 2x B200), which passes the version check. This
changes our benchmark hardware from B300 to B200.

---

## 2. Cookbook: sidecar venv uses the task image's python3, which breaks many MiMo tasks

**Where:** `training/examples/harbor_rl/tito/prepare_tasks.py`, `_SIDECAR_PYTHON_INSTALL`.

The layer installs apt `python3`, then builds `/opt/fireworks-tito` with the first `python3` on
PATH. Many `FineEnvs/MiMo-V2.6-RL-harbor-code` images ship an older `/usr/local/bin/python3`
that shadows it. Also, the assert (`>= 3.10`) is looser than the pins (`numpy==2.4.6` needs
>= 3.11). E2B template builds fail at step 4:
- `format-code-task-000371`: `AssertionError: 3.8.20`
- `format-code-task-000144`: `AssertionError: 3.7.17`
- `format-code-task-000411`: python 3.10 passes the assert, then
  `No matching distribution found for numpy==2.4.6`

**Our fix** (in `train_mimo_qwen3p6.py` `bake()`): use the image `python3` if it is >= 3.11, else
`/usr/bin/python3` (apt, installed by the same layer); assert >= 3.11. All affected tasks then
built. Suggested upstream: same logic, or a uv-managed CPython for the sidecar (the mimoagent
harness already does this for its own venv).

---

## 3. Cookbook: `prebuild_e2b_templates` treats failed templates as built

**Where:** `training/examples/harbor_rl/tito/e2b_templates.py`. "exists" is
`environment._does_template_exist()` -> `AsyncTemplate.alias_exists()`, which is true for a
template whose build errored or never started. After one failed prebuild, a relaunch reports
`existed=True` for the broken templates, skips them, and every sandbox for those tasks fails
mid-run with GPUs billing. Also, `asyncio.gather` aborts the whole prebuild on the first
failure while the other builds keep running server-side.

**Our fix:** after prebuild, check the actual E2B `buildStatus == "ready"` for every template
and abort before creating a trainer. Suggested upstream: check build status (rebuild if not
ready) and use `return_exceptions=True` to report all failures at once.

---

## 4. (FYI, E2B-side, not Fireworks) MiMo images that cannot be provisioned on E2B

These fail in E2B's base provisioning step, before any cookbook layer. They were replaced in our
split: `000216` and candidate `002127` (E2B "internal error"), `000751` (dead dart-archive apt
source), `002974` and `001495` (stale Debian 11 apt lists, openssh-server/sudo 404). Details
and build IDs are in `task-split.json` under `excluded`. This may be worth flagging to whoever
maintains the dataset's E2B compatibility.
