#!/usr/bin/env python3
"""Budget + stall watchdog for one `train_mimo_qwen3p6.py train` process.

Runs beside the training process (stdlib only, system python is fine). Every
POLL_S it accrues spend from what is actually up, samples E2B sandbox
concurrency, and tracks optimizer progress from COOKBOOK_METRICS_FILE:

  fireworks $  = trainer jobs x --trainer-usd-hr + deployment replicas x --gen-usd-hr
                 (resources created after watchdog start; list price, so an
                 estimate, not the invoice)
  e2b $        = sum over running sandboxes of cpu x VCPU_USD_S + GiB x GIB_USD_S

Stop conditions -> SIGINT to the training pid (the loop's @with_signals unwinds
and tears down trainer + deployment), SIGTERM after GRACE_S, SIGKILL + API
DELETE of this run's trainer/deployment after a second GRACE_S:
  - fireworks spend >= --fw-stop-usd
  - e2b spend >= --e2b-stop-usd
  - trials have started but no new optimizer step for --stall-min minutes

Outputs (in --out-dir): watchdog.log (one line per event, also stdout),
watchdog-status.json (latest snapshot), e2b-concurrency.jsonl (per poll).
After the training pid exits it keeps polling until this run's Fireworks
resources are gone (or TEARDOWN_WAIT_S), then exits.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

POLL_S = 60
GRACE_S = 900
TEARDOWN_WAIT_S = 1800
VCPU_USD_S = 0.000014       # e2b.dev/pricing, 2026-10-27
GIB_USD_S = 0.0000045
TERMINAL = ("COMPLETED", "FAILED", "CANCELLED", "DELETED", "EXPIRED")


def now() -> float:
    return time.time()


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


def parse_ts(value: str | None) -> float:
    if not value:
        return 0.0
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def http(method: str, url: str, headers: dict[str, str]) -> object:
    request = urllib.request.Request(url, method=method, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read()
    return json.loads(body) if body else {}


class Watchdog:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.out = Path(args.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.started = now()
        # resources created before this are not this run's (a restarted watchdog passes --since)
        self.since = parse_ts(args.since) if args.since else self.started - 60
        self.fw_usd = args.initial_fw_usd
        self.e2b_usd = args.initial_e2b_usd
        self.last_step = -1
        # set when first trial appears; a restarted watchdog resumes the clock now
        self.last_progress = now() if args.resume else None
        self.stop_reason = None
        self.stop_sent_at = None
        self.escalation = 0
        self.fw_resources: dict[str, str] = {}  # name -> state, this run only
        self.fw_headers = {"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"}
        self.e2b_headers = {"X-API-KEY": os.environ["E2B_API_KEY"]}
        self.fw_base = f"https://api.fireworks.ai/v1/accounts/{args.account}"

    def log(self, message: str) -> None:
        line = f"{iso(now())} {message}"
        print(line, flush=True)
        with open(self.out / "watchdog.log", "a") as handle:
            handle.write(line + "\n")

    def pid_alive(self) -> bool:
        try:
            os.kill(self.args.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def poll_fireworks(self, dt: float) -> None:
        live: dict[str, str] = {}
        usd_hr = 0.0
        for kind, key in (("rlorTrainerJobs", "rlorTrainerJobs"), ("deployments", "deployments")):
            data = http("GET", f"{self.fw_base}/{kind}?pageSize=200", self.fw_headers)
            for item in data.get(key, []):
                if parse_ts(item.get("createTime")) < self.since:
                    continue
                state = str(item.get("state", "UNKNOWN"))
                if any(t in state for t in TERMINAL):
                    continue
                live[item["name"]] = state
                if kind == "rlorTrainerJobs":
                    usd_hr += self.args.trainer_usd_hr
                else:
                    replicas = item.get("replicaCount")
                    usd_hr += self.args.gen_usd_hr * (1 if replicas is None else max(int(replicas), 0))
        for name, state in live.items():
            if self.fw_resources.get(name) != state:
                self.log(f"fireworks {name} -> {state}")
        for name in set(self.fw_resources) - set(live):
            self.log(f"fireworks {name} -> gone")
        self.fw_resources = live
        self.fw_usd += usd_hr * dt / 3600

    def poll_e2b(self, dt: float) -> None:
        sandboxes = http("GET", "https://api.e2b.dev/sandboxes", self.e2b_headers)
        usd_s = sum(
            float(s.get("cpuCount") or 2) * VCPU_USD_S
            + float(s.get("memoryMB") or 8192) / 1024 * GIB_USD_S
            for s in sandboxes
        )
        self.e2b_usd += usd_s * dt
        with open(self.out / "e2b-concurrency.jsonl", "a") as handle:
            handle.write(json.dumps({"t": iso(now()), "running": len(sandboxes)}) + "\n")
        self.sandboxes = len(sandboxes)

    def poll_progress(self) -> None:
        trials = Path(self.args.trials_dir)
        if self.last_progress is None and trials.is_dir() and any(trials.iterdir()):
            self.last_progress = now()
            self.log("first trials started; stall clock armed")
        metrics = Path(self.args.metrics_file)
        if not metrics.exists():
            return
        step = self.last_step
        for line in metrics.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Optimizer records carry train/* (grad norm, loss, ...); producer
            # gauges carry producer/event; eval records carry neither train/*.
            if "producer/event" in record or not any(k.startswith("train/") for k in record):
                continue
            step = max(step, int(record.get("rollout/step", -1)))
        if step > self.last_step:
            self.last_step = step
            self.last_progress = now()
            self.log(f"optimizer step {step} | fw ${self.fw_usd:.0f} e2b ${self.e2b_usd:.1f}")

    def check_stop(self) -> None:
        if self.stop_reason is None:
            stall_s = self.args.stall_min * 60
            if self.fw_usd >= self.args.fw_stop_usd:
                self.stop_reason = f"fireworks spend ${self.fw_usd:.0f} >= ${self.args.fw_stop_usd:.0f}"
            elif self.e2b_usd >= self.args.e2b_stop_usd:
                self.stop_reason = f"e2b spend ${self.e2b_usd:.1f} >= ${self.args.e2b_stop_usd:.0f}"
            elif self.last_progress is not None and now() - self.last_progress > stall_s:
                self.stop_reason = f"stall: no optimizer step for {self.args.stall_min} min (last step {self.last_step})"
            if self.stop_reason and self.pid_alive():
                self.log(f"STOP {self.stop_reason} -> SIGINT pid {self.args.pid}")
                os.kill(self.args.pid, signal.SIGINT)
                self.stop_sent_at = now()
            return
        if not self.pid_alive() or now() - self.stop_sent_at < GRACE_S * (self.escalation + 1):
            return
        self.escalation += 1
        if self.escalation == 1:
            self.log(f"ESCALATE pid {self.args.pid} still alive -> SIGTERM")
            os.kill(self.args.pid, signal.SIGTERM)
        elif self.escalation == 2:
            self.log(f"ESCALATE pid {self.args.pid} still alive -> SIGKILL + API delete")
            os.kill(self.args.pid, signal.SIGKILL)
            for name in self.fw_resources:
                try:
                    http("DELETE", f"https://api.fireworks.ai/v1/{name}", self.fw_headers)
                    self.log(f"DELETE {name} ok")
                except Exception as exc:  # noqa: BLE001
                    self.log(f"DELETE {name} failed: {exc}")

    def write_status(self, phase: str) -> None:
        status = {
            "t": iso(now()),
            "phase": phase,
            "pid_alive": self.pid_alive(),
            "elapsed_min": round((now() - self.started) / 60, 1),
            "fireworks_usd": round(self.fw_usd, 2),
            "e2b_usd": round(self.e2b_usd, 2),
            "e2b_running": getattr(self, "sandboxes", None),
            "last_step": self.last_step,
            "min_since_progress": None if self.last_progress is None
            else round((now() - self.last_progress) / 60, 1),
            "fireworks_resources": self.fw_resources,
            "stop_reason": self.stop_reason,
        }
        (self.out / "watchdog-status.json").write_text(json.dumps(status, indent=1) + "\n")

    def run(self) -> None:
        a = self.args
        self.log(
            f"watchdog start pid={a.pid} fw_stop=${a.fw_stop_usd} e2b_stop=${a.e2b_stop_usd} stall={a.stall_min}min "
            f"rates trainer=${a.trainer_usd_hr}/hr gen=${a.gen_usd_hr}/hr"
        )
        last = now()
        exited_at = None
        while True:
            time.sleep(POLL_S)
            dt, last = now() - last, now()
            for poll in (lambda: self.poll_fireworks(dt), lambda: self.poll_e2b(dt), self.poll_progress):
                try:
                    poll()
                except Exception as exc:  # noqa: BLE001 - one failed poll must not kill the watchdog
                    self.log(f"poll error: {type(exc).__name__}: {exc}")
            if exited_at is None and not self.pid_alive():
                exited_at = now()
                self.log(f"training pid {a.pid} exited; waiting for Fireworks teardown")
            if exited_at is None:
                self.check_stop()
            self.write_status("running" if exited_at is None else "teardown")
            if exited_at is not None and (not self.fw_resources or now() - exited_at > TEARDOWN_WAIT_S):
                left = ", ".join(f"{n}={s}" for n, s in self.fw_resources.items()) or "none"
                self.log(
                    f"watchdog done | fw ${self.fw_usd:.0f} e2b ${self.e2b_usd:.1f} | "
                    f"last step {self.last_step} | fireworks resources left: {left}"
                )
                self.write_status("done")
                return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--metrics-file", required=True)
    parser.add_argument("--trials-dir", required=True)
    parser.add_argument("--fw-stop-usd", type=float, required=True)
    parser.add_argument("--e2b-stop-usd", type=float, required=True)
    parser.add_argument("--stall-min", type=float, default=60)
    parser.add_argument("--initial-fw-usd", type=float, default=0.0, help="spend already accrued (watchdog restart)")
    parser.add_argument("--initial-e2b-usd", type=float, default=0.0, help="spend already accrued (watchdog restart)")
    parser.add_argument("--since", help="count Fireworks resources created at/after this ISO time (watchdog restart)")
    parser.add_argument("--resume", action="store_true", help="run already in progress: arm the stall clock now")
    parser.add_argument("--trainer-usd-hr", type=float, required=True, help="list $/hr per trainer job")
    parser.add_argument("--gen-usd-hr", type=float, required=True, help="list $/hr per generator replica")
    Watchdog(parser.parse_args()).run()


if __name__ == "__main__":
    main()
