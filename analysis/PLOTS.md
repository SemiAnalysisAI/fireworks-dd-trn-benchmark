# RL run report plots

How to produce the six standard plots for an async RL run, and the exact spec
they follow so plots from different runs (and different stacks) look the same
and can be compared side by side.

Reference output, from run `qwen3p6moe-mimorl-001` (Qwen3.6-35B-A3B, GRPO, 20
steps):
[`fireworks-training-runs/qwen3p6moe-mimorl-001/plots/`](../fireworks-training-runs/qwen3p6moe-mimorl-001/plots/).

| File | What it answers |
|---|---|
| `eval-reward.png` | Is the policy getting better on held-out tasks? |
| `step-time-mfu.png` | Where does each optimizer step's wall time go, and how busy is the trainer while it runs? |
| `rollout-time.png` | Where does a rollout's time go: inference, sandbox, or weight sync? |
| `sandbox-time.png` | Where does sandbox time go, and how many sandboxes run at once? |
| `rollout-stats.png` | How long, how big and how many turns are rollouts from each policy version? |
| `throughput-producer-consumer.png` | Does the generator keep up with the trainer? |

## 1. Generate them (Fireworks cookbook stack)

Requirements:

- The fw-ai/cookbook `training` venv (it provides the fireworks SDK that
  decodes TITO trajectory artifacts), plus `pip install matplotlib==3.10.7`.
- The run was launched with both of these set, so the run directory holds
  `metrics.jsonl` and `client-phase-trace.json`:

  ```bash
  export COOKBOOK_METRICS_FILE="$PWD/runs/<run>/metrics.jsonl"
  export COOKBOOK_TRACE_FILE="$PWD/runs/<run>/client-phase-trace.json"
  ```

- Harbor trial directories in `runs/<run>/trials/`, each with `result.json` and
  `artifacts/tito/compact/trajectory.tito`.

Then:

```bash
~/.cache/fireworks-cookbook/training/.venv/bin/python analysis/report.py runs/<run>
```

It writes `runs/<run>/analysis/trials.jsonl` (one record per trial; about 90 s
per 1,000 trials, reused on later runs of the command, `--reextract` to redo),
`runs/<run>/analysis/analysis.json` (derived stats) and the PNGs in
`runs/<run>/plots/`.

Model-specific constants live at the top of `analysis/analyze.py`: `CFG` (model
architecture, for FLOPs) and `PEAK_FLOPS` (trainer GPUs). Change them when the
model or the trainer shape changes.

## 2. Data each plot needs (any stack)

If you are not on the cookbook stack, log these and reproduce the plots from
the spec in section 4.

**Per optimizer step**

| Field | Meaning | Cookbook source |
|---|---|---|
| step time | wall time from the previous weight publish to this one | `perf/step_time` |
| run time | union of open fwd/bwd request windows + optimizer time | `perf/forward_backward_pending_time` + `perf/optim_step_time` |
| weight sync time | saving + hot-loading the new weights on the generator | `perf/weight_update_time` |
| eval wait | time the loop waited for a running eval before it could sync weights | `eval/weight_sync_wait_time` (logged on the eval's step; counts in the next step) |
| checkpoint time | checkpoint save after the publish (counts in the next step) | `checkpoint/wall_time` |
| trained tokens | all tokens of all sequences in the fwd/bwd batch | `train/target_tokens` |
| publish time and optimizer end time | wall clock | trace spans `weight_sync` and `optimizer` (`cat=async`, `args.batch_id`) |

**Per rollout (trial)**

| Field | Meaning |
|---|---|
| start, end | trial wall clock; the sandbox lives for this whole window |
| is eval | eval rollouts are excluded from every per-version plot |
| trained | true if the rollout's trajectory went into a training batch (not discarded) |
| setup, verifier windows | sandbox creation + agent install, and hidden-test verification |
| agent execution window | the agent loop |
| model call windows | start/end of every model request, including ones that failed after retries |
| sequence length | final context: prompt + every model turn + every tool output |
| turns | number of model calls that returned a turn |
| sequence lengths of trained pieces | if the rollout was split into several training sequences, each one's length |
| reward | 0/1 from the verifier |

**Per eval**: the step it evaluated and, per held-out task, the reward of every
attempt (a failed attempt counts as 0).

## 3. Definitions

- **Model version** of a rollout: number of weight publishes before the rollout
  started. 0 is the base model; k is the policy after step k.
- **Inference time** of a rollout: sum of its model-call windows, failed calls
  included. **Failed share**: failed-call time / inference time.
- **Sandbox time**: rollout wall time minus inference time. **Tool execution**
  = agent execution window minus inference time; **verifier** = verifier
  window; **setup** = the rest (sandbox create, agent install, gaps between
  phases).
- **Max concurrent sandboxes** in a version: highest number of overlapping trial
  windows (train and eval) while that version was being served.
- **Rollout wait** in a step: step time minus run time, weight sync, eval wait
  and checkpoint time.
- **MFU** = model FLOPs / (run time x peak FLOP/s). Model FLOPs follow NeMo
  Automodel `qwen3_5_flops` (Megatron-Bridge PR #2925): 6 FLOPs per
  multiply-add for every GEMM (attention projections, Gated DeltaNet
  projections, routed + shared experts, LM head), plus the Gated DeltaNet conv
  and state update (4 x value heads x value_dim^2 per token), plus causal
  softmax attention in full-attention layers (6 x heads x head_dim x S^2 per
  layer per sequence). Router and MTP head are excluded. When per-step sequence
  lengths are not logged, use the run's token-weighted mean length for S. Peak
  for B200 dense BF16 is 2.25 PFLOP/s per GPU. Run time is client-measured, so
  MFU is a lower bound.
- **Generator throughput** for step k: tokens of trained rollouts that finished
  between publish k-1 and publish k, divided by that window. **Trainer
  throughput**: trained tokens / run time.
- **pass@k**: share of eval tasks solved at least once in k attempts. **avg@k**:
  mean reward over all k attempts of every task. Use k = 3.

## 4. Visual spec

### Shared style

| Element | Rule |
|---|---|
| Size | width 14-15 in (11 in for the eval plot), height 6.2-6.6 in for one panel, 8.5-10.5 in for two or three; 200 dpi PNG on white |
| Font | matplotlib default (DejaVu Sans), base 10 pt |
| Title | `<run id> · <plot name>`, 14 pt bold, left-aligned at x = 0.05. No subtitle, no footnote |
| Legend | one row above the plot area, left-aligned, no frame, 9.5 pt. Labels of one to three words, no parentheses |
| Axis labels | short: `Step`, `Model version`, `Step time`, `MFU`, `Mean time per rollout`, `Tokens/s`, `Minutes`. No explanations |
| Gridlines | horizontal only, 0.8 px, `#e1e0d9` |
| Spines | bottom only, `#c3c2b7`; no tick marks; tick labels `#898781` |
| Units on ticks | `5 min`, `40K`, `6%`; zero is written `0` |
| Stacked bars | width 0.62, 1 px white gap between segments. Each segment carries its share of the bar (`42%`, 7.5 pt) when it is at least 11 pt tall: white text on dark fills, `#0b0b0b` on light fills |
| Lines | 2 px with 6 pt round markers, white 1.5 px marker edge. Value labels 8 pt bold above the point on a white rounded box |
| Two y axes | only to overlay a line on bars. Both axes get round tops with the same number of ticks (4-7, whichever leaves the least headroom) so the right ticks sit on the left gridlines |

### Palette

| Role | Hex |
|---|---|
| Run / inference / trainer / latency | `#2a78d6` |
| Weight sync / generator / sequence length | `#1baf7a` (rollout plot: weight sync) |
| Weight sync (step plot), sandbox, turns, pass@k | `#eb6834` |
| Sandbox setup / tool execution / verifier | `#f6b393` / `#eb6834` / `#a8401a` |
| Waiting, per-task dots | `#b9bab5` |
| MFU and concurrency lines | `#0b0b0b` |
| Text / secondary text | `#0b0b0b` / `#52514e` |
| Failed-inference label | `#184f95` |

### Plot by plot

**`eval-reward.png`** · title `eval pass@k and avg@k`
- x: eval step (0, 5, 10, ...). y: score 0-1.05.
- pass@k line (orange) and avg@k line (blue), values labeled above (pass@k)
  and below (avg@k) each point, 3 decimals.
- Grey dots: each task's avg@k, jittered ±0.35 around the step.

**`step-time-mfu.png`** · title `step time and MFU`
- x: optimizer step. Left y: minutes.
- Stacked bar per step, bottom to top: Run (blue), Weight sync (orange),
  Rollout wait (grey). Eval wait and checkpoint time are left out of the bar.
- Right y: MFU line (black) with each point labeled `6.4%`.

**`rollout-time.png`** · title `rollout time per model version`
- x: model version (training rollouts only). y: minutes.
- Stacked bar, bottom to top: mean inference time per rollout (blue), mean
  sandbox time per rollout (orange), the weight sync that loaded that version
  (green; 0 for version 0).
- When failed share >= 10%, label above the bar: `35% inference\ncalls failed`
  in `#184f95`, bold 7.5 pt.

**`sandbox-time.png`** · title `sandbox time per model version`
- x: model version. Left y: minutes.
- Stacked bar of mean per rollout, bottom to top: Setup, Tool execution,
  Verifier (light to dark orange).
- Right y: Max concurrent sandboxes line (black), each point labeled.

**`rollout-stats.png`** · title `rollout stats per model version`
- Three panels sharing the x axis (model version), top to bottom: Latency
  (minutes, blue), Sequence length (`K` tokens, green), Turns (orange).
- In each panel: a pale bar from min to max (width 0.42, the panel colour at
  22% opacity), and the medians as a line with dots, each labeled (`6 min`,
  `52K`, `56`).
- Key at top right, 9 pt grey: `dot = median, bar = min to max`.

**`throughput-producer-consumer.png`** · title `generator vs trainer throughput`
- Top panel: grouped bars per step, Generator (green) and Trainer (blue), in
  tokens/s with `K` ticks; each bar labeled in thousands (`2.7`).
- Bottom panel: a cumulative flow diagram over wall-clock minutes since the
  first rollout.
  - Produced (green step line): running total of trained rollouts' tokens,
    added when each rollout finishes.
  - Consumed (blue step line): running total of trained tokens, jumping at each
    optimizer step's end; label steps 1, 5, 10, 15, 20.
  - Backlog: green fill at 13% opacity between the lines where produced is
    ahead.
  - Background: alternating `#f1f1ee` columns, one per served model version,
    labeled `v0`, `v1`, ... at the top when the column is wider than 4 min.
  - How to read it: slope is rate, the vertical gap is the backlog, and flat
    stretches on both lines are the whole loop stalled.

## 5. Token and cost summary (table, not a plot)

- **Generated tokens**: sum of sequence length over every rollout. Report the
  split into trained, discarded and eval rollouts.
- **Trained tokens**: sum of trained tokens over all steps.
- **Cost**: Fireworks GPU time plus E2B sandbox time for the successful
  attempt. Split E2B between training and eval by sandbox-hours. Charge eval
  the GPU time it kept the trainer and generator up without training: the
  summed eval wait, plus the final eval after the last step, at the shape's
  hourly rate.
- Report cost per million generated, trained, and generated + trained tokens,
  with and without eval.

The reference table for `qwen3p6moe-mimorl-001` is in the repo README under
"Tokens and cost".
