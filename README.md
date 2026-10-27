# fireworks-mimo-rl

Agentic RL training of **Qwen3.6-35B-A3B** (full-parameter GRPO) on **MiMo
coding tasks**, using the Fireworks **Dedicated Training API** for
trainer/generator GPUs, **Harbor** for sandboxed agent rollouts, and **E2B**
for the sandboxes. Built on the recipes in
[fw-ai/cookbook](https://github.com/fw-ai/cookbook) (`training/examples/harbor_rl`).

Each rollout is one Harbor trial: a fresh E2B sandbox boots a digest-pinned
task image, Xiaomi's pinned `mimoagent` harness works a real repository issue,
and the task's hidden tests decide the reward (1/0). Rewards form GRPO groups;
the trainer (4x B300, full-param) updates and hot-loads weights into the
generator deployment (1x B300) between steps.

## Contents

| File | Purpose |
|---|---|
| `train_mimo_qwen3p6.py` | The whole run: `fetch` / `bake` / `train` subcommands, all knobs in the `SETTINGS` block |
| `mimo-trial-config.yaml` | Harbor trial budget caps (agent 2400s, verifier 600s) — step-time tail control |
| `task-split.json` | Seeded (42) train/eval split: 80 train, 8 held-out eval tasks |
| `AGENTS.md` | Setup + operating instructions for coding agents (start here) |
| `fireworks-training-runs/` | Run manifests: approved plans, incidents, costs, outcomes |

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

## Status

Smoke run (2026-10-27): 4 GRPO steps end to end, mean reward 0.60 (12/20
trials) — consistent with the parity baseline. See
`fireworks-training-runs/mimo-qwen3p6-rl-001/run.md` for the full history,
including three integration incidents and their fixes.
