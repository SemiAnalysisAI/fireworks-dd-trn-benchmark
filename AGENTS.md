# Agent setup and operating guide

Instructions for coding agents (Claude Code, Cursor, etc.) working in this
repo. Everything here was learned the expensive way; read before launching.

## What this repo does

`train_mimo_qwen3p6.py` runs full-parameter GRPO on Qwen3.6-35B-A3B against
MiMo coding tasks. GPUs are remote (Fireworks Dedicated Training API);
sandboxes are remote (E2B); **this machine runs the orchestration loop**, so
it must stay awake for the whole run — launch under `caffeinate -i`.

Money: the dedicated shape bills ~$75/hr list (trainer 4x B300 + generator
1x B300) from spin-up to teardown. E2B bills sandbox time separately. Never
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

## Run records

Every run gets a manifest in `fireworks-training-runs/<run-id>/run.md`: the
approved plan + cost, resource IDs, incidents, outcome. Append, don't rewrite
history. Get explicit user approval (with a cost line) before any launch, and
record the approval quote in the manifest.
