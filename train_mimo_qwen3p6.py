#!/usr/bin/env python3
"""Train Qwen3.6-35B-A3B on MiMo coding tasks with the dedicated async RL loop.

Minimal, non-extensible adaptation of
``training/examples/harbor_rl/train_opencode.py`` (fw-ai/cookbook): Harbor task
rows + the MiMoAgent harness (``harnesses.mimoagent.rollout.make_rollout_fn``)
+ ``recipes.async_rl_loop.main`` (server-side GRPO on the Dedicated Training
API). Every setting lives in SETTINGS below — edit the block, don't pass flags.

Dataset: ``FineEnvs/MiMo-V2.6-RL-harbor-code`` — 2,698 ready-made Harbor task
dirs (digest-pinned images, hidden tests mounted only at verify time, reward
1/0 from the task's own test command). Its reference harness is the same
pinned mimoagent commit this recipe bakes in.

Usage (needs the cookbook's ``training`` package importable — run with the
cookbook venv python from ``~/.cache/fireworks-cookbook/training``, and
FIREWORKS_API_KEY / E2B_API_KEY exported, e.g. ``set -a; source .env``)::

  python train_mimo_qwen3p6.py fetch    # snapshot the HF dataset
  python train_mimo_qwen3p6.py bake     # seeded train/eval split + harness bake
  python train_mimo_qwen3p6.py train    # launch with SETTINGS below

The seeded split is persisted to ``task_split_file`` on first ``bake`` and
reused afterwards; eval tasks never enter the training rows.
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

from training.examples.harbor_rl import common
from training.examples.harbor_rl.harnesses.mimoagent.constants import (
    MIMOAGENT_HARBOR_IMPORT_PATH,
    PINNED_MIMOAGENT_VERSION,
)
from training.examples.harbor_rl.harnesses.mimoagent.rollout import make_rollout_fn
from training.examples.harbor_rl.tito.e2b_templates import prebuild_e2b_templates
from training.examples.harbor_rl.tito.evaluate import make_fixed_evaluation
from training.examples.harbor_rl.tito.rollout import (
    DEFAULT_MAX_CONCURRENT_TRIALS,
    DEFAULT_ROLLOUT_RETRIES,
)
from training.examples.harbor_rl.tito.trial import (
    DEFAULT_HARNESS_TOOL_TIMEOUT_SECONDS,
)
from training.examples.harbor_rl.tito.trial import load_harbor_rows
from training.recipes.async_rl_loop import Config, RolloutFnFactory, main
from training.utils import WandBConfig

logger = logging.getLogger(__name__)

MIMO_CODE_DATASET = "FineEnvs/MiMo-V2.6-RL-harbor-code"

SETTINGS = SimpleNamespace(
    # --- model / renderer (TITO contract; renderer fails closed if wrong) ---
    base_model="accounts/fireworks/models/qwen3p6-35b-a3b",
    tokenizer_model="Qwen/Qwen3.6-35B-A3B",
    tokenizer_revision=None,
    renderer_name="qwen3_6_preserved",  # TITO-certified (qwen3.6-35b-a3b-preserved@995ad96e); plain "qwen3_6" has no production TITO certification
    tito_prompt_mode="full_history",  # "incremental" is experimental
    # --- dedicated resources (shape pins 4x B300 full-param trainer +
    #     1x B300 BF16 generator per replica; ~$75/hr list while up) ---
    training_shape_id="accounts/fireworks/trainingShapes/qwen3p6-35b-a3b-262k",
    deployment_shape=None,            # None -> generator resolved from the shape
    trainer_job_id=None,              # set to reattach an existing trainer
    deployment_id=None,               # set to reattach a deployment
    replica_count=1,                  # generator replicas (+$15/hr each)
    sample_timeout=600,               # s per sampling request
    weight_sync_timeout=600,          # s per sampler hot-load
    skip_cleanup=False,               # True keeps trainer+deployment after exit
    output_model_id="mimo-qwen3p6-rl-20step",  # keep the tuned model artifact
    # --- data / sandbox ---
    source_tasks_dir="./mimo-code/tasks",    # raw HF snapshot (2,698 tasks)
    harbor_dataset="./mimo-tasks-prepared",  # harness-baked task dir (bake output)
    task_split_file="./task-split.json",     # persisted seeded split (bake creates)
    train_task_count=80,              # 4 groups/step x 20 steps, epochs=1
    eval_task_count=8,                # held out; never in training rows
    task_seed=42,
    harbor_registry_path=None,
    harbor_environment="e2b",         # no local Docker on this machine
    harbor_trials_dir=Path("./runs/mimo-20step/trials"),  # keep per-trial records
    harbor_trial_config="./mimo-trial-config.yaml",  # caps budgets to fit 1h E2B sandbox
    e2b_request_timeout=900.0,        # s; survives concurrent sandbox-create waves
    mimo_arm="bash",                  # mimoagent arm: "bash" or "cc"
    # E2B templates MUST be prebuilt once before rollout fan-out: concurrent
    # trials of one task race to build the same content-addressed template and
    # E2B cancels the duplicates (non-retryable BuildException).
    template_concurrency=16,
    template_timeout_seconds=1800.0,
    # --- harness / trial limits ---
    max_concurrent_trials=64,         # E2B Pro allows 100 concurrent sandboxes
    harness_tool_timeout_seconds=DEFAULT_HARNESS_TOOL_TIMEOUT_SECONDS,
    rollout_retries=DEFAULT_ROLLOUT_RETRIES,                      # 3
    retry_include_exception=[],
    terminal_failure_reward=None,     # None: retry then discard, never score 0
    tito_debug=False,
    # --- RL scale: 20 steps ---
    epochs=1,
    completions_per_prompt=8,         # GRPO group size
    prompt_groups_per_step=4,         # 4 tasks/step -> 32 rollouts/step, one wave
    max_completion_tokens=2048,       # per model call
    max_seq_len=65536,                # trajectory context budget (shape max 262144)
    temperature=1.0,
    learning_rate=1e-6,               # full-param; LoRA would use ~1e-5
    lora_rank=0,                      # 0 = full-parameter (matches the shape)
    max_head_offpolicy_versions=1,    # next step's rollouts overlap training
    dcp_save_interval=10,             # mid-run checkpoint at step 10
    # --- evaluation / observability ---
    sampling_only=False,              # True scores deployment_id, no trainer
    evaluation_every=5,               # holdout eval cadence (plus start and end)
    evaluation_concurrency=32,
    run_dir=Path("./runs/mimo-20step"),
    wandb_entity="",                  # "" disables W&B; metrics.jsonl always written
    wandb_project="harbor-rl-mimo",
    wandb_run_name=None,
)


def fetch() -> None:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=MIMO_CODE_DATASET,
        repo_type="dataset",
        local_dir="./mimo-code",
        allow_patterns=["tasks/**", "registry.json", "manifest.json", "data/*"],
    )
    tasks = sorted(p.name for p in (Path(path) / "tasks").iterdir() if p.is_dir())
    print(
        f"snapshot of {MIMO_CODE_DATASET} at {path}: {len(tasks)} Harbor tasks; "
        "next, bake the harness with harnesses.mimoagent.prepare_tasks"
    )


def load_split(s: SimpleNamespace) -> dict[str, list[str]]:
    """Seeded train/eval task split, persisted on first bake and reused after."""
    split_path = Path(s.task_split_file)
    if split_path.exists():
        split = json.loads(split_path.read_text())
        if len(split["train"]) != s.train_task_count or len(split["eval"]) != s.eval_task_count:
            raise ValueError(
                f"{split_path} does not match SETTINGS counts; delete it to re-split"
            )
        if set(split["train"]) & set(split["eval"]):
            raise ValueError("train/eval split overlaps")
        return split
    names = sorted(p.name for p in Path(s.source_tasks_dir).iterdir() if p.is_dir())
    need = s.train_task_count + s.eval_task_count
    if len(names) < need:
        raise ValueError(f"need {need} tasks, found {len(names)} in {s.source_tasks_dir}")
    sampled = random.Random(s.task_seed).sample(names, need)
    split = {
        "seed": s.task_seed,
        "train": sorted(sampled[: s.train_task_count]),
        "eval": sorted(sampled[s.train_task_count :]),
    }
    split_path.write_text(json.dumps(split, indent=1))
    logger.info("Wrote %s: %d train / %d eval tasks", split_path, len(split["train"]), len(split["eval"]))
    return split


def bake(s: SimpleNamespace) -> None:
    from training.examples.harbor_rl.harnesses.mimoagent.prepare_tasks import prepare

    split = load_split(s)
    names = split["train"] + split["eval"]
    prepared = prepare(
        Path(s.source_tasks_dir), Path(s.harbor_dataset), task_names=names
    )
    print(f"baked mimoagent harness into {len(prepared)} task contexts in {s.harbor_dataset}")


def rollout_fn_factory(s: SimpleNamespace, run_dir: Path) -> RolloutFnFactory:
    return partial(
        make_rollout_fn,
        harbor=common.harbor_settings(
            s,
            run_dir=run_dir,
            environment=s.harbor_environment,
            trials_dir=s.harbor_trials_dir,
        ),
        arm=s.mimo_arm,
    )


def build_config(s: SimpleNamespace, *, run_dir: Path, row_count: int) -> Config:
    return Config(
        log_path=str(run_dir / "logs"),
        base_model=s.base_model,
        learning_rate=s.learning_rate,
        completions_per_prompt=s.completions_per_prompt,
        prompt_groups_per_step=s.prompt_groups_per_step,
        max_completion_tokens=s.max_completion_tokens,
        max_seq_len=s.max_seq_len,
        temperature=s.temperature,
        epochs=s.epochs,
        max_rows=row_count,
        lora_rank=s.lora_rank,
        max_head_offpolicy_versions=s.max_head_offpolicy_versions,
        dcp_save_interval=s.dcp_save_interval,
        weight_sync_timeout=s.weight_sync_timeout,
        cleanup_on_exit=not s.skip_cleanup,
        output_model_id=s.output_model_id,
        trainer=common.trainer_config(s),
        deployment=common.deploy_config(s),
        wandb=WandBConfig(
            entity=s.wandb_entity,
            project=s.wandb_project,
            run_name=s.wandb_run_name or f"harbor-rl-mimo-{int(time.time()) % 100000}",
        ),
    )


def train(s: SimpleNamespace) -> None:
    if s.harbor_environment == "e2b":
        common.require_env("E2B_API_KEY")
        common.set_e2b_request_timeout(s.e2b_request_timeout)
    run_dir = s.run_dir.expanduser().resolve()
    split = load_split(s)
    rows = load_harbor_rows(
        s.harbor_dataset,
        registry_path=s.harbor_registry_path,
        task_names=split["train"],
        n_tasks=s.train_task_count,
    )
    eval_rows = load_harbor_rows(
        s.harbor_dataset,
        registry_path=s.harbor_registry_path,
        task_names=split["eval"],
        n_tasks=s.eval_task_count,
    )
    if not rows or not eval_rows:
        raise ValueError(f"No Harbor tasks found for {s.harbor_dataset!r}")
    logger.info("Loaded %d train / %d eval Harbor tasks", len(rows), len(eval_rows))
    if s.harbor_environment == "e2b":
        import asyncio

        records = asyncio.run(
            prebuild_e2b_templates(
                rows + eval_rows,
                trials_dir=run_dir / "template-trials",
                agent_import_path=MIMOAGENT_HARBOR_IMPORT_PATH,
                agent_version=PINNED_MIMOAGENT_VERSION,
                agent_provider="openai",
                context_limit=s.max_seq_len,
                output_limit=s.max_completion_tokens,
                trial_config=s.harbor_trial_config,
                max_concurrency=s.template_concurrency,
                timeout_seconds=s.template_timeout_seconds,
                tool_timeout_seconds=s.harness_tool_timeout_seconds,
            )
        )
        logger.info(
            "E2B templates ready: %d built, %d already existed",
            sum(1 for r in records if not r.existed),
            sum(1 for r in records if r.existed),
        )
    factory = rollout_fn_factory(s, run_dir)

    if s.sampling_only:
        common.run_sampling_only(
            s,
            factory,
            rows,
            completions_per_prompt=s.completions_per_prompt,
            max_concurrency=s.evaluation_concurrency,
            result_path=run_dir / "sampling-result.json",
        )
        return

    evaluation_fn = make_fixed_evaluation(
        eval_rows,
        completions_per_prompt=s.completions_per_prompt,
        max_concurrency=s.evaluation_concurrency,
    )
    main(
        build_config(s, run_dir=run_dir, row_count=len(rows)),
        rollout_fn_factory=factory,
        evaluation_fn=evaluation_fn,
        evaluation_interval=s.evaluation_every,
        rows=rows,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "fetch":
        fetch()
    elif command == "bake":
        bake(SETTINGS)
    elif command == "train":
        train(SETTINGS)
    else:
        sys.exit("usage: train_mimo_qwen3p6.py {fetch|bake|train} — settings are edited in SETTINGS")
