"""Flatten every Harbor trial of a run into one JSON record per trial.

Reads ``<run>/trials/*/result.json`` (Harbor phase timestamps, reward) and the
TITO artifact ``artifacts/tito/compact/trajectory.tito`` (per-call windows,
per-turn token counts, sidecar metric distributions). Writes
``<run>/analysis/trials.jsonl``. Needs the fireworks SDK (cookbook venv).
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

from fireworks.training.sdk.tito._artifact import unpack_trajectory_artifact

ARTIFACT = "artifacts/tito/compact/trajectory.tito"
PHASES = ("environment_setup", "agent_setup", "agent_execution", "verifier")


def _ts(value: str | None) -> float | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() if value else None


def trial_record(trial: Path) -> dict:
    rec: dict = {"trial": trial.name}
    result = json.loads((trial / "result.json").read_text())
    rec.update(
        started=_ts(result.get("started_at")),
        finished=_ts(result.get("finished_at")),
        reward=((result.get("verifier_result") or {}).get("rewards") or {}).get("reward"),
        exception=(result.get("exception_info") or {}).get("exception_type"),
    )
    for phase in PHASES:
        window = result.get(phase) or {}
        rec[phase] = [_ts(window.get("started_at")), _ts(window.get("finished_at"))]
    artifact = unpack_trajectory_artifact((trial / ARTIFACT).read_bytes())
    rec.update(
        status=artifact.status,
        terminal_reason=artifact.terminal_reason,
        metadata=dict(artifact.metadata),
        # [start, end, outcome, attempts, error_code]; failed = every upstream attempt errored
        calls=[[c.started_at, c.ended_at, c.outcome, c.attempts, c.error_code] for c in artifact.calls],
        dist={k: [d.count, d.sum, d.min, d.max] for k, d in artifact.metrics.distributions.items()},
        # one trainer sequence per segment; its length is the last turn's prompt + completion
        segments=[
            len(s.turns[-1].exact_prompt_ids) + len(s.turns[-1].exact_completion_ids)
            for s in artifact.segments
            if s.turns
        ],
    )
    return rec


def extract(run_dir: Path, workers: int = 8) -> Path:
    trials = sorted(p for p in (run_dir / "trials").iterdir() if (p / ARTIFACT).exists() and (p / "result.json").exists())
    out = run_dir / "analysis" / "trials.jsonl"
    out.parent.mkdir(exist_ok=True)
    with ProcessPoolExecutor(workers) as pool, open(out, "w") as handle:
        for rec in pool.map(trial_record, trials, chunksize=8):
            handle.write(json.dumps(rec) + "\n")
    return out
