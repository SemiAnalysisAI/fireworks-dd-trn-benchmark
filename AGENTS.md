# Agent setup and operating guide

Instructions for coding agents (Claude Code, Cursor, etc.) working in this
repo. Everything here was learned the expensive way; read before launching.

## What this repo does

`train_mimo_qwen3p6.py` runs full-parameter GRPO on Qwen3.6-35B-A3B against
MiMo coding tasks. GPUs are remote (Fireworks Dedicated Training API);
sandboxes are remote (E2B); **this machine runs the orchestration loop**, so
it must stay awake for the whole run — launch under `caffeinate -i`.

Money: the dedicated shape bills ~$78/hr list (trainer 4x B200 + generator
2x B200, shape `qwen3p6-35b-a3b-256k`; the B300 shape `-262k` was ~$75/hr but
is blocked server-side, see pitfall 7) from spin-up to teardown. E2B bills sandbox time separately. Never
launch `train` without explicit user approval of a plan that includes cost.

## One-time setup

1. **Keys** — a gitignored `.env` in the repo root with:
   - `FIREWORKS_API_KEY` — use a scoped service-account key, not a personal
     admin key (`firectl api-key create --service-account <name>`).
   - `E2B_API_KEY` — needs the Pro tier for this config (24h sandbox
     sessions, 100 concurrent; Hobby caps at 1h/20 which this config exceeds).
   - `HF_TOKEN` — optional, speeds up the dataset snapshot.

   Load them per process: `set -a; source .env; set +a`. The scripts read
   `os.environ`, not `.env`.

2. **Cookbook checkout + venv** (the script imports the cookbook's `training`
   package; it is not standalone):

   ```bash
   git clone --depth 1 https://github.com/fw-ai/cookbook.git ~/.cache/fireworks-cookbook
   cd ~/.cache/fireworks-cookbook/training
   python3 -m venv .venv            # Python 3.11+ (3.14 works; pydantic-v1 warning is benign)
   .venv/bin/pip install -e .
   .venv/bin/pip install 'harbor[e2b]==0.21.0' 'e2b==2.55.0' huggingface_hub
   .venv/bin/pip install 'matplotlib==3.10.7'   # analysis/ plots only; adds packages, upgrades none
   ```

   Do not upgrade `e2b` independently — 2.56+ needs a different `pyqwest`
   than the cookbook pins. Run everything below with that venv's python.

3. **Data**:

   ```bash
   python train_mimo_qwen3p6.py fetch   # ~214MB, 21k small files; HF_TOKEN helps
   python train_mimo_qwen3p6.py bake    # writes task-split.json (if absent) + mimo-tasks-prepared/
   ```

   `bake` refuses to overwrite `mimo-tasks-prepared/`; `rm -rf` it to rebake.
   `task-split.json` is committed for reproducibility — the 8 eval tasks are
   held out of training by construction. Don't regenerate it casually; the
   split is only comparable across runs if it stays fixed.

## Launching a run

All knobs live in the `SETTINGS` block of `train_mimo_qwen3p6.py` — edit the
block, don't add CLI flags. Then:

```bash
set -a; source .env; set +a
export COOKBOOK_METRICS_FILE="$PWD/runs/<run-name>/metrics.jsonl"   # REQUIRED for per-step metrics
export COOKBOOK_TRACE_FILE="$PWD/runs/<run-name>/client-phase-trace.json"  # Perfetto timeline
caffeinate -i ~/.cache/fireworks-cookbook/training/.venv/bin/python -u \
    train_mimo_qwen3p6.py train > runs/<run-name>/launch.log 2>&1 &
```

Evals run 3 attempts per task (`eval_completions_per_prompt=3`); keep it at 3.
Run `qwen3p6moe-mimorl-001` used 8 and its evals stalled training for 67 min,
about 29% of its Fireworks spend.

Without `COOKBOOK_METRICS_FILE` (or W&B configured) the loop persists **no
per-step metrics** — only console logs and per-trial artifacts. This was
missed once; don't miss it again.

## Pitfalls already paid for (do not rediscover)

1. **E2B template build race.** Concurrent trials of one task race to build
   the same content-addressed template; E2B cancels duplicates
   (`BuildException: build was cancelled`, non-retryable). The script's
   `train` subcommand prebuilds all templates before fan-out
   (`prebuild_e2b_templates`) — keep it that way. Templates are per-E2B-account
   and content-addressed: a new account or changed Dockerfile rebuilds.

2. **E2B plan limits.** Harbor requests 24h sandboxes (hardcoded
   `timeout=86_400` in `harbor/environments/e2b.py`). Hobby-tier accounts
   (1h cap) reject every sandbox with `400: Timeout cannot be greater than
   1 hours`. On Hobby you must patch that constant down and cap trial budgets;
   on Pro, stock harbor works. Current assumption: Pro.

3. **TITO renderer certification.** The sidecar fail-closes on renderers
   without production TITO certification. For Qwen3.6-35B-A3B the certified
   name is `qwen3_6_preserved` — NOT `qwen3_6`, even though `qwen3_6` builds
   fine in the supervised path. To verify a renderer before launch:

   ```python
   from training.utils.tokenizers import load_tokenizer
   from renderer.tito.registry import get_tito_renderer_certification  # run from cookbook/training
   get_tito_renderer_certification("<name>", load_tokenizer("<hf-tokenizer>", None))
   ```

4. **zsh eats `$path`.** Don't use `path` as a shell loop variable; zsh ties
   it to `PATH` and every subsequent command "disappears".

5. **Sidecar python vs. old task images.** The cookbook builds the TITO sidecar
   venv over the image's first `python3`; many MiMo images ship 3.7-3.10 in
   `/usr/local/bin` and the sidecar pins need >= 3.11. `bake` rewrites that
   block (`fix_sidecar_python`) to fall back to apt `/usr/bin/python3`.

6. **"Template existed" can mean "template is broken".** Prebuild checks
   `alias_exists()`, which is true for errored or never-started builds. `train`
   now checks real E2B `buildStatus` and aborts before GPU spend. Some MiMo
   images can't be provisioned by E2B at all (dead apt sources, stale Debian 11
   lists, unexplained "internal error"); those are replaced in
   `task-split.json` (`excluded` records why).

7. **Generator serving version blocked.** Since 2026-10-27 Fireworks rejects
   deployments on serving [7.55.0, 7.70.0); the B300 `qwen3p6-35b-a3b-262k`
   generators resolve to 7.56.0/7.67.0. Probe a shape for free before launch:
   `POST /deployments?deploymentId=probe&validateOnly=true` with its
   `deploymentShape`; a "hot load bucket url is required" 400 means it passed
   the version check. Report: `fireworks-training-runs/qwen3p6moe-mimorl-001/fireworks-issues.md`.

## Monitoring a run

- Trainer/deployment state: read-only GETs on
  `https://api.fireworks.ai/v1/accounts/<acct>/rlorTrainerJobs/<id>` and
  `/deployments/<id>`.
- Per-step records land in `COOKBOOK_METRICS_FILE` (two streams: optimizer
  records keyed `rollout/step`, producer gauges keyed `producer/event` — never
  join them on the generic step field).
- Sequencing expectations: trainer RUNNING ~4 min, generator serving
  ~10–15 min, then first trials. First optimizer step can take 10–40 min.
  No step after that → gather evidence, stop, debug; don't let it idle-burn.
- Generator saturation signal: `sampler/prefill_queue_s_mean` climbing →
  consider `replica_count+1` (+$15/hr, needs user approval).

## After a run — always verify teardown

The loop tears down on exit (`cleanup_on_exit=True`) and the platform has a
1h trainer inactivity timeout as backstop, but verify: list deployments and
trainer jobs via the API and confirm nothing is RUNNING/CREATING. Orphaned
E2B sandboxes die at their session timeout, but kill them via the E2B
dashboard if a run was interrupted ungracefully.

Checkpoint retention: set `output_model_id` in SETTINGS (a run without it
loses the tuned model at cleanup) and confirm the model exists
(`/v1/accounts/<acct>/models/<output_model_id>`) before declaring success.

## After a run — report plots

```bash
~/.cache/fireworks-cookbook/training/.venv/bin/python analysis/report.py runs/<run-name>
```

Decodes every trial's TITO artifact (~90 s per 1k trials; cached in
`runs/<run-name>/analysis/trials.jsonl`, `--reextract` to redo), derives
per-step and per-model-version stats, and writes PNGs to `runs/<run-name>/plots/`:
eval pass@k/avg@k, step time + MFU, rollout time, sandbox time, per-rollout
min/median/max stats, and generator-vs-trainer throughput. Needs the run's
`metrics.jsonl` and `client-phase-trace.json` (both env vars above). The plot
spec, data requirements and definitions are in `analysis/PLOTS.md`; copy the
PNGs into `fireworks-training-runs/<run-id>/plots/` so they are versioned.

- MFU counts model FLOPs with the NeMo Automodel `qwen3_5_flops` convention
  (6x per GEMM, Gated DeltaNet conv + state update, causal full attention)
  against 4x B200 dense BF16; time is client-side run time (fwd/bwd request
  windows + optimizer), so it is a lower bound. Update `CFG` / `PEAK_FLOPS` in
  `analysis/analyze.py` if the model or trainer shape changes.
- Per-step sequence lengths are not logged, so the quadratic attention term
  uses the run's token-weighted mean sequence length.

## Run records

Every run gets a manifest in `fireworks-training-runs/<run-id>/run.md`: the
approved plan + cost, resource IDs, incidents, outcome. Append, don't rewrite
history. Get explicit user approval (with a cost line) before any launch, and
record the approval quote in the manifest.
