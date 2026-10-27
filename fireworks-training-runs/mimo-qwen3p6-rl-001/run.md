# Run mimo-qwen3p6-rl-001

entry_skill: configure (via research handoff in-session)
session_id: 93b36757-f7d8-4f0e-af89-b9f2a1b83aa9
client_source: fireworks-training-skill/2.3.0
created: 2026-10-27
status: preflight-complete, awaiting launch approval

## Plan
- account: kimbo-semianalysis
- method: agentic RL (GRPO, server-side) via dedicated async_rl_loop
- surface: Dedicated Training API
- base_model: accounts/fireworks/models/qwen3p6-35b-a3b (READY, instruct)
- training_shape: accounts/fireworks/trainingShapes/qwen3p6-35b-a3b-262k
  version o5kla0at (validated, trainerImageTag 0.638.678, full-param POLICY_TRAINER,
  4x B300 x1 node, ctx 262144; linked generator rft-qwen3p6-35b-a3b-full-rdma-b300-bf16-w1-p1/f1dttnuc, 1x B300 BF16)
- tokenizer: Qwen/Qwen3.6-35B-A3B; renderer: qwen3_6 (resolved + built OK)
- dataset: FineEnvs/MiMo-V2.6-RL-harbor-code (HF, local snapshot ./mimo-code; adapter mimo_harbor 1.3.0, 2698 tasks)
- harness: mimoagent (pinned, matches dataset reference harness)
- sandboxes: E2B (E2B_API_KEY set; harbor[e2b]==0.21.0, e2b==2.55.0)
- script: train_mimo_qwen3p6.py (adapted from train_opencode.py)
- cookbook: ~/.cache/fireworks-cookbook @ 57968a2237b663be33078d2747a28463983ae134
- sdk: fireworks 1.2.30 (venv python 3.14; pydantic v1 warning, non-blocking)
- smoke params: max_rows 5, epochs 1, completions/prompt 4, groups/step 1,
  lr 1e-6, lora_rank 0 (full-param), max_seq_len 65536, max_completion_tokens 2048,
  replica_count 1, sync (offpolicy 0)
- pricing basis: fireworks.ai/pricing on 2026-10-26: B300 $15/GPU-hr on-demand
  -> trainer $60/hr + generator $15/hr = $75/hr while up (list, trainer replica 1, 1 gen replica)
- caution: model record rlTunable/rlLoraTunable both False (flags appear to gate
  managed/serverless RFT; dedicated shape exists+validated. Verify at create; fail fast if rejected.)
- monitoring: step/loss/grad-norm + reward per step; no-progress timeout 20 min -> debug
- teardown: cleanup_on_exit=True (trainer + deployment torn down on exit)

## Telemetry
firectl not installed -> local run state only; privacy notice shown in research turn.

## Approval
2026-10-27: user approved final plan ("lgtm, launch when you see fit") after metric
additions (Perfetto trace, trials-dir retention, MFU estimate post-proc, tool-call
time as mean). Launching smoke run: 5 tasks x 4 completions x 1 epoch, full-param,
shape qwen3p6-35b-a3b-262k, E2B sandboxes.
status: launching
trainer_job: accounts/kimbo-semianalysis/rlorTrainerJobs/training-api-service-ab6ce113
deployment: qwen3p6-35b-a3b-1793124108
launch_pid: 59046
launch_log: runs/mimo-smoke/launch.log
launched_at: 2026-10-27T14:01:48 local

## Incident 1 (resolved)
14:17-14:26: all E2B trials failed with non-retryable BuildException "build was
cancelled" — concurrent trials of one task raced to build the same
content-addressed E2B template. Fix: added prebuild_e2b_templates() step before
fan-out (same as train_pi.py). Stopped run via SIGINT at 14:26; cleanup confirmed
(deployment scaled to zero then deleted, trainer job deleted). ~25 min trainer +
~10 min generator billed (~$25 est). Relaunching with fix; plan otherwise unchanged.

## Incident 2 (resolved)
14:33-14:43 run 2: templates prebuilt OK (3 built, 2 existed) but every sandbox
create failed: E2B 400 "Timeout cannot be greater than 1 hours" — harbor 0.21.0
hardcodes timeout=86400 (24h, Pro tier); account is Hobby tier (1h cap, 20
concurrent). Loop ended cleanly: 0 steps, deployment scaled to zero + trainer
deleted automatically. ~8 min trainer billed (~$10 est).
Fix (user chose patch over Pro upgrade): (a) patched venv harbor e2b.py
timeout 86400->3600 (NOTE: reinstalling harbor undoes this), (b) added
mimo-trial-config.yaml capping agent 2400s / verifier 600s so trials fit the 1h
sandbox; slow trials (>40 min agent time) are killed+discarded by design.

## Incident 3 (resolved)
14:46-14:58 run 3: sandboxes created fine (both E2B fixes held) but every rollout
failed in-sidecar: ValueError "renderer 'qwen3_6' has no production TITO
certification". My pre-launch check validated build_renderer only; the TITO
sidecar requires the certification registry (renderer/tito/registry.py). The
certified name for this model is "qwen3_6_preserved"
(qwen3.6-35b-a3b-preserved@995ad96e-candidate); verified against the live
tokenizer fingerprint before relaunch. ~12 min trainer billed (~$15 est).

## Run 4: SUCCESS (smoke complete)
15:00-15:30: 4 optimizer steps completed (of 5 rows; 1 group not trained).
Rewards by task (4 trials each): t1=0,0,0,0 | t2=1,1,1,1 | t3=0,1,0,1 |
t4=0,1,0,1 | t7=0,1,1,1 -> mean 0.60 (12/20), in line with dataset's published
~0.556 parity baseline. Agent exec 30-358s (2400s cap not binding).
Checkpoints: DCP 'step-4' + sampler 'step-4-base' saved mid-run; promote timed
out at 90s; in-run model rlor-82ba3d605e90 NOT retained as account model after
cleanup (fine for smoke; scaled run must set output_model_id / promotion).
Teardown verified: no deployments, trainer deleted. Est. cost run 4 ~$36;
session total ~$85.
GAP for scaled run: COOKBOOK_METRICS_FILE was unset -> no per-step metrics
ledger; only console log + per-trial artifacts + Perfetto trace retained.
Set COOKBOOK_METRICS_FILE and WANDB_ENTITY next time.
status: smoke complete

## E2B account change (2026-10-27)
User switched to an E2B Pro account (24h sandboxes, 100 concurrent); new key
verified. Reverted the harbor e2b.py 1h patch to stock 86400s. Trial-config caps
(agent 2400s / verifier 600s) KEPT deliberately as step-time tail control.
New account has 0 templates -> full prebuild on first scaled run.
