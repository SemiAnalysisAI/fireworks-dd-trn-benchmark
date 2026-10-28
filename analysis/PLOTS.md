# RL run report plots

A presentation spec for six plots that summarize an asynchronous RL training
run: a trainer that runs optimizer steps, a generator that serves the current
policy, and agent rollouts that call the generator and run tools in sandboxes.
It assumes the data below has been collected somehow; it does not care which
framework, provider or sandbox system produced it. Following it makes plots
from different runs and stacks look the same, so they can be compared side by
side.

Example output (one 20-step GRPO run):
[`fireworks-training-runs/qwen3p6moe-mimorl-001/plots/`](../fireworks-training-runs/qwen3p6moe-mimorl-001/plots/).

| File | Question it answers |
|---|---|
| `eval-pass-avg.png` | Is the policy getting better on held-out tasks? |
| `step-time-mfu.png` | Where does each optimizer step's wall time go, and how well is the trainer used while it runs? |
| `rollout-time.png` | Where does a rollout's time go: inference, sandbox, or weight sync? |
| `sandbox-time.png` | Where does sandbox time go, and how many sandboxes run at once? |
| `rollout-stats.png` | How long, how big and how many turns are rollouts from each policy version? |
| `throughput-producer-consumer.png` | Does the generator keep up with the trainer? |

## 1. Data to collect

**Per optimizer step**

| Field | Meaning |
|---|---|
| step time | wall time from the previous weight publish to this step's publish |
| run time | time the trainer was actually computing: forward/backward plus the optimizer update |
| weight sync time | time to move the new weights onto the generator |
| eval wait | time the step was blocked waiting for an evaluation to finish (0 if none) |
| checkpoint time | time the step was blocked on a checkpoint save (0 if none) |
| trained tokens | all tokens of all sequences in the step's training batch |
| consume time | wall clock when the step finished training (optimizer done) |
| publish time | wall clock when the step's weights became the generator's policy |

**Per rollout**

| Field | Meaning |
|---|---|
| start, end | wall clock; the rollout's sandbox lives for this whole window |
| is eval | eval rollouts are excluded from every per-version plot |
| trained | whether the rollout went into a training batch (not discarded) |
| setup time | sandbox creation plus anything installed before the agent starts |
| verifier time | scoring the result (e.g. running hidden tests) |
| agent time | the agent loop, from first to last action |
| inference time | sum of the rollout's model-call durations, including calls that ultimately failed |
| failed inference time | the part of inference time spent in calls that failed |
| sequence length | the rollout's full final context: prompt, every model turn and every tool output |
| trained tokens | tokens this rollout contributed to training (sum over pieces if it was split into several sequences) |
| turns | number of model calls that returned a turn |

**Per evaluation**: the step it evaluated, and for each held-out task the reward
of every attempt (a failed attempt counts as 0). Use 3 attempts per task.

**Per run**: model architecture (for FLOPs), trainer accelerator count and peak
FLOP/s, and the cost rates for GPU time and sandbox time.

## 2. Definitions

- **Model version** of a rollout: number of weight publishes before the rollout
  started. 0 is the base model; k is the policy after step k.
- **Sandbox time** of a rollout: wall time (end - start) minus inference time.
  **Tool execution** = agent time minus inference time. **Setup** = sandbox
  time minus tool execution minus verifier time (sandbox creation, installs and
  gaps between phases).
- **Failed share** of a version: failed inference time / inference time, over
  its rollouts.
- **Max concurrent sandboxes** of a version: highest number of overlapping
  rollout windows (training and eval) while that version was being served.
- **Rollout wait** of a step: step time minus run time, weight sync, eval wait
  and checkpoint time.
- **MFU**: model FLOPs / (run time x number of trainer accelerators x peak
  FLOP/s per accelerator). Count model FLOPs with the usual convention: 6 FLOPs
  per multiply-add of every matrix multiply that touches a token (forward +
  backward), using only the parameters a token activates (top-k experts for
  MoE), plus the attention term that grows with sequence length (causal
  softmax attention: 6 x heads x head_dim x S² per layer per sequence) and the
  per-token state math of any linear-attention layers. NVIDIA NeMo Automodel's
  `flops_utils` has these formulas for most current architectures. Leave out
  recomputation. If run time is measured from the client, say MFU is a lower
  bound.
- **Generator throughput** of step k: trained tokens of rollouts that finished
  between publish k-1 and publish k, divided by that window. **Trainer
  throughput**: trained tokens / run time.
- **pass@k**: share of eval tasks solved at least once in k attempts.
  **avg@k**: mean reward over all k attempts of all tasks.

## 3. Shared style

| Element | Rule |
|---|---|
| Size | width 14-15 in (11 in for the eval plot); height 6.2-6.6 in for one panel, 8.5-10.5 in for two or three; 200 dpi PNG on white |
| Font | one sans-serif family throughout, base 10 pt |
| Title | `<run id> · <plot name>`, 14 pt bold, left-aligned. No subtitle, no footnote |
| Legend | one row above the plot area, left-aligned, no frame, 9.5 pt. Labels of one to three words, no parentheses |
| Axis labels | short: `Step`, `Model version`, `Step time`, `MFU`, `Mean time per rollout`, `Tokens/s`, `Minutes`. No explanations |
| Gridlines | horizontal only, thin, `#e1e0d9` |
| Frame | bottom axis line only (`#c3c2b7`); no tick marks; tick labels `#898781` |
| Tick units | `5 min`, `40K`, `6%`; zero is written `0` |
| Stacked bars | width 0.62 of the slot, 1 px white gap between segments. Each segment shows its share of the bar (`42%`, 7.5 pt) when it is at least 11 pt tall: white text on dark fills, near-black on light fills |
| Lines | 2 px with 6 pt round markers and a white 1.5 px marker edge. Value labels 8 pt bold above the point on a white rounded box |
| Two y axes | only to overlay a line on bars. Give both axes round tops with the same number of ticks (4-7, whichever leaves the least headroom) so the right-hand ticks sit on the left-hand gridlines |

**Palette**

| Hex | Used for |
|---|---|
| `#2a78d6` blue | run time, inference, trainer, latency, avg@k |
| `#eb6834` orange | weight sync (step plot), sandbox, tool execution, turns, pass@k |
| `#1baf7a` green | weight sync (rollout plot), generator, sequence length |
| `#f6b393` / `#a8401a` | sandbox setup / verifier (light and dark orange) |
| `#b9bab5` grey | rollout wait, per-task dots |
| `#0b0b0b` black | MFU line, concurrency line, label text |
| `#184f95` dark blue | failed-inference label |

## 4. The plots

### `eval-pass-avg.png` · title `eval pass@k and avg@k`
- x: eval step (e.g. 0, 5, 10, ...). y: score, 0 to 1.05.
- pass@k line (orange), values labeled above each point; avg@k line (blue),
  values labeled below. Three decimals.
- Grey dots: each task's avg@k, jittered ±0.35 around its step.

### `step-time-mfu.png` · title `step time and MFU`
- x: optimizer step. Left y: minutes.
- One stacked bar per step, bottom to top: Run (blue), Weight sync (orange),
  Rollout wait (grey). Eval wait and checkpoint time are left out.
- Right y: MFU line (black), each point labeled (`6.4%`).

### `rollout-time.png` · title `rollout time per model version`
- x: model version (training rollouts only). y: minutes.
- One stacked bar per version, bottom to top: mean inference time per rollout
  (blue), mean sandbox time per rollout (orange), and the weight sync that
  loaded that version (green; 0 for the base model).
- When the version's failed share is 10% or more, label above the bar
  `35% inference` / `calls failed` (two lines) in dark blue, bold 7.5 pt.

### `sandbox-time.png` · title `sandbox time per model version`
- x: model version. Left y: minutes.
- One stacked bar of mean time per rollout, bottom to top: Setup, Tool
  execution, Verifier (light to dark orange).
- Right y: Max concurrent sandboxes line (black), each point labeled.

### `rollout-stats.png` · title `rollout stats per model version`
- Three panels sharing the x axis (model version), top to bottom: Latency
  (minutes, blue), Sequence length (`K` tokens, green), Turns (orange).
- In each panel, per version: a pale bar from min to max (width 0.42, panel
  colour at about 22% opacity), and the medians drawn as a line with dots, each
  dot labeled (`6 min`, `52K`, `56`).
- Key at the top right, 9 pt grey: `dot = median, bar = min to max`.

### `throughput-producer-consumer.png` · title `generator vs trainer throughput`
- **Top panel**: grouped bars per step, Generator (green) and Trainer (blue),
  in tokens/s with `K` ticks; each bar labeled in thousands (`2.7`).
- **Bottom panel**: a cumulative flow diagram over wall-clock minutes since the
  first rollout started.
  - Produced (green step line): running total of trained tokens, added when
    each trained rollout finishes.
  - Consumed (blue step line): running total of trained tokens, jumping at
    each step's consume time. Label a few steps (`step 1`, `step 5`, ...).
  - Backlog: green fill at about 13% opacity between the lines where Produced
    is ahead.
  - Background: alternating very light grey (`#f1f1ee`) columns, one per
    served model version (publish to publish), labeled `v0`, `v1`, ... at the
    top when the column is wide enough to fit the label.
  - Reading it: slope is rate, the vertical gap is the backlog, the horizontal
    gap is how long finished work waits to be trained, and both lines flat
    means the whole loop is stalled.

## 5. Token and cost summary (a table, not a plot)

- **Generated tokens**: sum of sequence length over every rollout, split into
  rollouts that were trained on, rollouts that were discarded, and eval
  rollouts.
- **Trained tokens**: sum of trained tokens over all steps.
- **Cost**: GPU time plus sandbox time for the run. Split sandbox cost between
  training and eval by sandbox-hours. Charge eval the GPU time it kept the
  trainer and generator up without training: the summed eval wait, plus any
  eval after the last step, at the hourly GPU rate.
- Report cost per million generated, trained, and generated + trained tokens,
  with and without eval.
