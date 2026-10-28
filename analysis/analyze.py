"""Derive per-step and per-model-version stats from one run's raw outputs.

Inputs: ``metrics.jsonl`` (COOKBOOK_METRICS_FILE), ``client-phase-trace.json``
(COOKBOOK_TRACE_FILE) and ``analysis/trials.jsonl`` (extract_trials.py).
Output: ``analysis/analysis.json``, consumed by plots.py.

Policy version of a rollout = number of weight publishes before its trial
started (0 = base model). Publish time = end of the step's ``weight_sync`` span.
"""

from __future__ import annotations

import json
from pathlib import Path

# Qwen3.6-35B-A3B text config (HF config.json, model_type qwen3_5_moe_text).
CFG = dict(
    hidden=2048, layers=40, full_layers=10, heads=16, kv_heads=2, head_dim=256, vocab=248320,
    lin_k_heads=16, lin_v_heads=32, lin_k_dim=128, lin_v_dim=128, conv=4,
    experts_per_tok=8, moe_ffn=512, shared_ffn=512,
)
PEAK_FLOPS = 4 * 2.25e15  # trainer: 4x B200, dense BF16


def flops_coefficients(c: dict = CFG) -> tuple[float, float]:
    """(FLOPs per trained token, FLOPs per squared sequence length) for fwd + bwd.

    Follows NeMo Automodel ``qwen3_5_flops`` / Megatron-Bridge PR #2925: every
    GEMM costs 6 x MACs; Gated DeltaNet layers add the conv and the chunked
    delta-rule state update (4 x v_heads x v_dim^2 per token); full-attention
    layers add the causal QK^T/PV term, quadratic in sequence length. Router
    and MTP head are excluded (MTP is not trained in RL).
    """
    hs, q_dim = c["hidden"], c["heads"] * c["head_dim"]
    gdn_layers = c["layers"] - c["full_layers"]
    full_proj = 6 * hs * q_dim * (2 * c["kv_heads"] / c["heads"] + 2 + 1)  # q (+gate), k, v, o
    qk, v, nv = c["lin_k_heads"] * c["lin_k_dim"], c["lin_v_heads"] * c["lin_v_dim"], c["lin_v_heads"]
    gdn = 6 * (hs * (2 * qk + 2 * v + 2 * nv) + c["conv"] * (2 * qk + v) + nv * c["lin_v_dim"] ** 2 * 4 + hs * v)
    mlp = 6 * c["layers"] * hs * 3 * (c["moe_ffn"] * c["experts_per_tok"] + c["shared_ffn"])
    vocab = 6 * hs * c["vocab"]
    per_token = c["full_layers"] * full_proj + gdn_layers * gdn + mlp + vocab
    per_seq_sq = c["full_layers"] * 6 * q_dim  # x S^2 per sequence (causal half already applied)
    return per_token, per_seq_sq


def _windows(events: list[dict], origin: float, name: str, category: str) -> dict[int, tuple[float, float]]:
    return {
        e["args"]["batch_id"]: (origin + e["ts"] / 1e6, origin + (e["ts"] + e["dur"]) / 1e6)
        for e in events
        if e.get("name") == name and e.get("cat") == category and "batch_id" in e.get("args", {})
    }


def analyze(run_dir: Path) -> Path:
    records = [json.loads(line) for line in open(run_dir / "metrics.jsonl")]
    trace = json.loads((run_dir / "client-phase-trace.json").read_text())
    trials = [json.loads(line) for line in open(run_dir / "analysis" / "trials.jsonl")]
    origin, events = trace["trace_origin_unix_ns"] / 1e9, trace["traceEvents"]

    # Optimizer records carry train/*; eval and checkpoint records land on the step that started them.
    steps = [r for r in records if "train/grad_norm" in r and "producer/event" not in r]
    eval_wait = {r["rollout/step"]: r["eval/weight_sync_wait_time"] for r in records if "eval/weight_sync_wait_time" in r}
    ckpt = {r["rollout/step"]: r["checkpoint/wall_time"] for r in records if "checkpoint/wall_time" in r}
    sync = _windows(events, origin, "weight_sync", "async")
    optimizer = _windows(events, origin, "optimizer", "async")
    publish = {k: end for k, (_, end) in sync.items()}
    run_start = min(t["started"] for t in trials)
    run_end = max(t["finished"] for t in trials)
    bounds = [run_start] + [publish[k] for k in sorted(publish)]  # version v serves [bounds[v], bounds[v+1])

    def version_at(ts: float) -> int:
        return sum(1 for k in publish if publish[k] <= ts)

    # ---- rollouts --------------------------------------------------------------------------------
    rollouts = []
    for t in trials:
        e2e = t["finished"] - t["started"]
        agent = (t["agent_execution"][1] or t["agent_execution"][0]) - t["agent_execution"][0] if t["agent_execution"][0] else 0.0
        verify = (t["verifier"][1] - t["verifier"][0]) if t["verifier"][0] and t["verifier"][1] else 0.0
        inference = sum(c[1] - c[0] for c in t["calls"])
        trained = not t["metadata"]["evaluation"] and (
            t["status"] == "completed" or t["terminal_reason"] == "context_budget_exhausted")
        dist = t["dist"]
        rollouts.append(dict(
            version=version_at(t["started"]), eval=t["metadata"]["evaluation"], trained=trained,
            start=t["started"], end=t["finished"], e2e=e2e,
            inference=inference, inference_failed=sum(c[1] - c[0] for c in t["calls"] if c[2] == "failed"),
            tool=max(0.0, agent - inference), verify=verify, setup=max(0.0, e2e - agent - verify),
            seqlen=dist.get("trajectory/final_context_tokens", [0, 0, 0, 0])[3],
            turns=dist.get("trajectory/policy_turns", [0, 0])[1],
            segments=t["segments"] if trained else [],
            prompt=dist.get("turn/prompt_tokens", [0, 0])[1], cached=dist.get("cache/cached_prompt_tokens", [0, 0])[1],
            sampled=dist.get("turn/completion_tokens", [0, 0])[1],
            terminal_reason=t["terminal_reason"],
        ))

    # sandbox concurrency: every trial (train or eval) holds one sandbox from start to finish
    deltas = sorted([(r["start"], 1) for r in rollouts] + [(r["end"], -1) for r in rollouts])
    level, live = [], 0
    for ts, d in deltas:
        live += d
        level.append((ts, live))

    def max_concurrency(a: float, b: float) -> int:
        before = [n for ts, n in level if ts < a]
        return max([before[-1] if before else 0] + [n for ts, n in level if a <= ts < b])

    # ---- FLOPs / MFU -----------------------------------------------------------------------------
    per_token, per_seq_sq = flops_coefficients()
    seqs = [s for r in rollouts for s in r["segments"]]
    # Per-step sequence lengths are not logged; use the run's token-weighted mean length for sum(S^2).
    s_eff = sum(s * s for s in seqs) / sum(seqs)

    produced = sorted((r["end"], sum(r["segments"]), r["version"]) for r in rollouts if r["trained"])
    step_rows = []
    for r in steps:
        s = r["rollout/step"]
        tokens = r["train/target_tokens"]
        flops = per_token * tokens + per_seq_sq * s_eff * tokens
        run_s = r["perf/forward_backward_pending_time"] + r["perf/optim_step_time"]
        a, b = bounds[s - 1], bounds[s]
        gen_tokens = sum(tok for ts, tok, _ in produced if a <= ts < b)
        step_rows.append(dict(
            step=s, step_time=r["perf/step_time"], run_time=run_s, tokens=tokens, flops=flops,
            mfu=flops / (run_s * PEAK_FLOPS), trainer_tps=tokens / run_s, gen_tokens=gen_tokens, gen_tps=gen_tokens / (b - a),
            weight_sync=r["perf/weight_update_time"], eval_wait=eval_wait.get(s - 1, 0.0), checkpoint=ckpt.get(s - 1, 0.0),
            consumed_at=optimizer[s][1], window=[a, b],
        ))

    versions = []
    for v in range(len(bounds)):
        rs = [r for r in rollouts if not r["eval"] and r["version"] == v]
        if not rs:
            continue
        a, b = bounds[v], bounds[v + 1] if v + 1 < len(bounds) else run_end
        versions.append(dict(
            version=v, n=len(rs), max_concurrency=max_concurrency(a, b),
            weight_sync=sync[v][1] - sync[v][0] if v in sync else 0.0,  # the sync that put this version on the generator
            **{k: sum(r[k] for r in rs) for k in ("e2e", "inference", "inference_failed", "tool", "setup", "verify")},
            **{f"{k}_each": [r[k] for r in rs] for k in ("e2e", "seqlen", "turns")},
        ))

    # eval: every attempt's reward per task (a failed attempt counts as 0)
    evals = []
    for r in records:
        if "eval/reward" not in r:
            continue
        tasks: dict[str, list[float]] = {}
        for t, row in zip(trials, rollouts):
            if row["eval"] and row["version"] == r["rollout/step"]:
                tasks.setdefault(t["metadata"]["task_name"], []).append(t["reward"] or 0.0)
        evals.append(dict(
            step=r["rollout/step"], reward=r["eval/reward"], completed=r["eval/completed_trajectories"],
            attempted=r["eval/attempted_trajectories"], tasks=dict(sorted(tasks.items())),
        ))

    out = run_dir / "analysis" / "analysis.json"
    out.write_text(json.dumps(dict(
        flops_per_token=per_token, flops_per_seq_sq=per_seq_sq, s_eff=s_eff, peak_flops=PEAK_FLOPS,
        run_start=run_start, run_end=run_end, publish=bounds[1:], steps=step_rows, versions=versions, evals=evals,
        produced=produced, rollouts=rollouts,
    )))
    return out
