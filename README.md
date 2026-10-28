# fireworks-dd-trn-benchmark

Agentic RL training of **Qwen3.6-35B-A3B** (full-parameter GRPO) on **MiMo
coding tasks**, using the Fireworks **Dedicated Training API** for
trainer/generator GPUs, **Harbor** for sandboxed agent rollouts, and **E2B**
for the sandboxes. Built on the recipes in
[fw-ai/cookbook](https://github.com/fw-ai/cookbook) (`training/examples/harbor_rl`).

Each rollout is one Harbor trial: a fresh E2B sandbox boots a digest-pinned
task image, Xiaomi's pinned `mimoagent` harness works a real repository issue,
and the task's hidden tests decide the reward (1/0). Rewards form GRPO groups;
the trainer (4x B200, full-param) updates and hot-loads weights into the
generator deployment (2x B200) between steps. The intended B300 shape
(`qwen3p6-35b-a3b-262k`) is blocked server-side for now; see AGENTS.md pitfall 7.

## Contents

| File | Purpose |
|---|---|
| `train_mimo_qwen3p6.py` | The whole run: `fetch` / `bake` / `train` subcommands, all knobs in the `SETTINGS` block |
| `mimo-trial-config.yaml` | Harbor trial budget caps (agent 2400s, verifier 600s) — step-time tail control |
| `task-split.json` | Seeded (42) train/eval split: 80 train, 8 held-out eval tasks |
| `AGENTS.md` | Setup + operating instructions for coding agents (start here) |
| `fireworks-training-runs/` | Run manifests: approved plans, incidents, costs, outcomes |
| `budget_watchdog.py` | Spend + stall watchdog that runs beside `train` and stops it at a limit |
| `analysis/` | Post-run report: `report.py runs/<run>` decodes trials and writes the run's plots; [`PLOTS.md`](analysis/PLOTS.md) is the shareable plot spec |

## Quickstart

See **[AGENTS.md](AGENTS.md)** — it is written for coding agents but is the
canonical setup doc for humans too. Short version:

```bash
# keys in .env: FIREWORKS_API_KEY, E2B_API_KEY, HF_TOKEN (optional)
# one-time: clone fw-ai/cookbook, build its venv, install harbor[e2b] pins
python train_mimo_qwen3p6.py fetch   # snapshot the HF dataset (2,698 tasks)
python train_mimo_qwen3p6.py bake    # seeded split + bake mimoagent harness
python train_mimo_qwen3p6.py train   # launch (reads SETTINGS; costs real money)
```

## Dataset

[`FineEnvs/MiMo-V2.6-RL-harbor-code`](https://huggingface.co/datasets/FineEnvs/MiMo-V2.6-RL-harbor-code)
— the Code domain of Xiaomi's MiMo-V2.6-RL-oss as 2,698 ready-made Harbor
tasks. Its reference harness is the same pinned `mimoagent` commit this repo's
rollouts bake in, so runs reproduce Xiaomi's original setup (published parity
baseline: mean reward ~0.556).

## Evaluation

The 8 held-out tasks are scored every `evaluation_every` steps plus at the start
and end, reported as pass@k (tasks solved at least once in k attempts) and
avg@k (mean reward over all k attempts, failed attempts = 0).

**Use 3 attempts per task (`eval_completions_per_prompt=3`), not 8.** Run
`qwen3p6moe-mimorl-001` used 8: its evals stalled training for 67 min (each
blocks the next weight sync until its slowest trial ends), used 30% of all
sandbox time, and cost ~$123 of a ~$385 run. 3 attempts is the default in
SETTINGS from that run on; its numbers are pass@8/avg@8 and are not directly
comparable to later pass@3/avg@3.

## Tokens and cost: `qwen3p6moe-mimorl-001`

20 GRPO steps (4 tasks x 8 rollouts per step) on the B200 shape. A sample's
tokens are its full sequence (prompt + model turns + tool outputs); rollouts
split into several training sequences are summed.

| | Samples | Tokens |
|---|---|---|
| Generated, training rollouts trained on | 550 | 26.43M |
| Generated, training rollouts discarded | 90 | 6.39M |
| Generated, eval rollouts (8 attempts x 8 tasks x 5 evals) | 320 | 17.25M |
| **Generated, total** | **960** | **50.06M** |
| **Trained on** (`train/target_tokens`, 20 steps) | 549 rollouts / 647 sequences | **26.40M** |
| **Generated + trained** | | **76.46M** |

Cost (watchdog list-price estimate, successful attempt only; E2B Pro base fee
excluded). E2B is split by sandbox-hours. Eval's Fireworks share is the time
evals kept the trainer and generator up without training (4 mid-run evals
stalled training 67.4 min; the final eval ran 19.2 min after step 20) at $78/hr.

| | Training | Eval | Total |
|---|---|---|---|
| Sandbox-hours | 109.4 | 46.9 | 156.3 |
| E2B | $25.04 | $10.73 | $35.77 |
| Fireworks | $236.57 | $112.60 | $349.17 |
| **Total** | **$261.61** | **$123.33** | **$384.94** |

| Cost per M tokens | Training only | Including eval |
|---|---|---|
| Per M generated tokens | $7.97 (32.81M) | $7.69 (50.06M) |
| Per M trained tokens | $9.91 (26.40M) | $14.58 (26.40M) |
| Per M generated + trained | $4.42 (59.21M) | $5.03 (76.46M) |

Eval alone: $123.33 for 17.25M generated tokens ($7.15 per M). Including the
three failed launch attempts, the run cost ~$418 (Fireworks ~$380, E2B ~$38).
Plots for the run are in
[`fireworks-training-runs/qwen3p6moe-mimorl-001/plots/`](fireworks-training-runs/qwen3p6moe-mimorl-001/plots/);
[`analysis/PLOTS.md`](analysis/PLOTS.md) explains how to generate them for any run.

## Status

- **`qwen3p6moe-mimorl-001`** (2026-10-28): 20 GRPO steps on 4x B200 trainer +
  2x B200 generator; model promoted as
  `accounts/kimbo-semianalysis/models/qwen3p6moe-mimorl-20step`. Held-out eval
  pass@8 0.500 -> 0.625, avg@8 0.203 -> 0.266 (step 0 -> 20), within noise for
  8 tasks. Trainer MFU during run time 4.6-7.3%; the trainer was idle ~90% of
  each step, waiting on rollouts, weight sync and evals. Full history, incidents
  and costs: `fireworks-training-runs/qwen3p6moe-mimorl-001/run.md`; issues to
  report to Fireworks: `fireworks-issues.md` in the same folder.
- Smoke run (2026-10-27): 4 GRPO steps end to end, mean reward 0.60 (12/20
  trials) — consistent with the parity baseline. See
  `fireworks-training-runs/mimo-qwen3p6-rl-001/run.md`.
