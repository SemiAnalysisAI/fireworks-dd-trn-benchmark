"""Render the run report PNGs from ``analysis/analysis.json`` into ``<run>/plots/``."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, BLUE_DARK, ORANGE, AQUA, NEUTRAL = "#2a78d6", "#184f95", "#eb6834", "#1baf7a", "#b9bab5"
ORANGE_L, ORANGE_D = "#f6b393", "#a8401a"
FAILED_SHARE = 0.10  # label versions whose inference time is >= 10% failed (HTTP 503) calls

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "savefig.facecolor": "white",
})


def style(*axes) -> None:
    for ax in axes:
        for side in ("top", "left", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
    axes[0].grid(axis="y", color=GRID, linewidth=0.8, zorder=0)


def figure(title: str, size: tuple[float, float], **kwargs):
    fig, axes = plt.subplots(figsize=size, dpi=200, **kwargs)
    fig.suptitle(title, x=0.05, y=0.975, ha="left", fontsize=14, fontweight="bold", color=INK)
    return fig, axes


def legend(fig, handles, ncol: int, y: float = 0.93) -> None:
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.045, y), frameon=False, fontsize=9.5, ncol=ncol)


def fmt(template: str) -> FuncFormatter:
    return FuncFormatter(lambda val, _: template.format(val) if val else "0")


def version_axis(ax, versions: list[dict]) -> None:
    ax.set_xticks([v["version"] for v in versions])
    ax.set_xlim(-0.6, len(versions) - 0.4)
    ax.set_xlabel("Model version")


def nice_top(value: float, ticks: int = 5) -> float:
    """Smallest round axis top >= value that splits into `ticks` equal round steps."""
    step = value / ticks
    magnitude = 10 ** np.floor(np.log10(step))
    return ticks * magnitude * next(m for m in (1, 2, 2.5, 5, 10) if m * magnitude >= step)


def align_twin(ax, ax2, left_max: float, right_max: float) -> None:
    """Give both y axes round tops with the same tick count (4-7, least headroom) so right ticks sit on the left gridlines."""
    ticks = min(range(4, 8), key=lambda n: nice_top(left_max, n) / left_max + nice_top(right_max, n) / right_max)
    left, right = nice_top(left_max, ticks), nice_top(right_max, ticks)
    ax.set_ylim(0, left)
    ax.set_yticks(np.linspace(0, left, ticks + 1))
    ax2.set_ylim(0, right)
    ax2.set_yticks(np.linspace(0, right, ticks + 1))


def pt_per_unit(fig, ax) -> float:
    fig.canvas.draw()
    lo, hi = ax.get_ylim()
    return ax.get_window_extent().height / fig.dpi * 72 / (hi - lo)


def stacked(ax, x: float, parts: list[tuple[float, str, str]], width: float, ppu: float) -> float:
    """Stack (value, color, label ink) segments at x; label each with its share of the bar where it fits."""
    total = sum(p[0] for p in parts)
    bottom = 0.0
    for value, color, ink in parts:
        if value <= 0:
            continue
        ax.bar(x, value, width, bottom=bottom, color=color, edgecolor="white", linewidth=1.0, zorder=3)
        if value * ppu >= 11:
            ax.text(x, bottom + value / 2, f"{100 * value / total:.0f}%", ha="center", va="center", fontsize=7.5, color=ink, zorder=4)
        bottom += value
    return bottom


def line_points(ax, xs, ys, color: str, label_fmt: str | None = None, dy: int = 9) -> None:
    ax.plot(xs, ys, color=color, lw=2, marker="o", ms=6, mec="white", mew=1.5, zorder=6)
    if label_fmt:
        for x, y in zip(xs, ys):
            ax.annotate(label_fmt.format(y), (x, y), textcoords="offset points", xytext=(0, dy), ha="center", fontsize=8,
                        color=INK, fontweight="bold", zorder=7, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


def step_time_mfu(a: dict, run: str, out: Path) -> None:
    steps = a["steps"]
    fig, ax = figure(f"{run} · step time and MFU", (14, 6.6))
    ax2 = ax.twinx()
    shown = [(s["step_time"] - s["eval_wait"] - s["checkpoint"]) / 60 for s in steps]
    align_twin(ax, ax2, max(shown) * 1.1, max(100 * s["mfu"] for s in steps) * 1.3)
    ppu = pt_per_unit(fig, ax)
    # Eval wait and checkpoint saves are left out: bars are step time minus both.
    parts = [("run_time", "Run", BLUE, "white"), ("weight_sync", "Weight sync", ORANGE, "white"),
             ("rollout_wait", "Rollout wait", NEUTRAL, INK)]
    for s in steps:
        s["rollout_wait"] = s["step_time"] - s["run_time"] - s["weight_sync"] - s["checkpoint"] - s["eval_wait"]
        stacked(ax, s["step"], [(s[k] / 60, c, i) for k, _, c, i in parts], 0.62, ppu)
    ax.yaxis.set_major_formatter(fmt("{:.0f} min"))
    ax.set_ylabel("Step time")
    ax.set_xticks([s["step"] for s in steps])
    ax.set_xlim(0.4, len(steps) + 0.6)
    ax.set_xlabel("Step")
    line_points(ax2, [s["step"] for s in steps], [100 * s["mfu"] for s in steps], INK, "{:.1f}%")
    ax2.yaxis.set_major_formatter(fmt("{:.0f}%"))
    ax2.set_ylabel("MFU")
    style(ax, ax2)
    legend(fig, [Patch(color=c, label=l) for _, l, c, _ in parts] + [
        Line2D([], [], color=INK, lw=2, marker="o", mec="white", label="MFU")], ncol=4)
    fig.subplots_adjust(left=0.06, right=0.95, top=0.87, bottom=0.09)
    fig.savefig(out / "step-time-mfu.png")
    plt.close(fig)


def rollout_time(a: dict, run: str, out: Path) -> None:
    versions = a["versions"]
    fig, ax = figure(f"{run} · rollout time per model version", (15, 6.6))
    ax.set_ylim(0, max(v["e2e"] / v["n"] + v["weight_sync"] for v in versions) / 60 * 1.12)
    ppu = pt_per_unit(fig, ax)
    for v in versions:
        parts = [(v["inference"] / v["n"] / 60, BLUE, "white"), ((v["e2e"] - v["inference"]) / v["n"] / 60, ORANGE, "white"),
                 (v["weight_sync"] / 60, AQUA, "white")]
        top = stacked(ax, v["version"], parts, 0.62, ppu)
        failed = v["inference_failed"] / v["inference"]
        if failed >= FAILED_SHARE:
            ax.text(v["version"], top + 0.15, f"{100 * failed:.0f}% inference\ncalls failed", ha="center", va="bottom",
                    fontsize=7.5, color=BLUE_DARK, fontweight="bold", linespacing=1.0)
    ax.yaxis.set_major_formatter(fmt("{:.0f} min"))
    ax.set_ylabel("Mean time per rollout")
    version_axis(ax, versions)
    style(ax)
    legend(fig, [Patch(color=BLUE, label="Inference"), Patch(color=ORANGE, label="Sandbox"),
                 Patch(color=AQUA, label="Weight sync")], ncol=3)
    fig.subplots_adjust(left=0.05, right=0.985, top=0.87, bottom=0.1)
    fig.savefig(out / "rollout-time.png")
    plt.close(fig)


def sandbox_time(a: dict, run: str, out: Path) -> None:
    versions = a["versions"]
    fig, ax = figure(f"{run} · sandbox time per model version", (15, 6.6))
    tops = [(v["setup"] + v["tool"] + v["verify"]) / v["n"] / 60 for v in versions]
    ax2 = ax.twinx()
    align_twin(ax, ax2, max(tops) * 1.15, max(v["max_concurrency"] for v in versions) * 1.12)
    ppu = pt_per_unit(fig, ax)
    parts = [("setup", "Setup", ORANGE_L, INK), ("tool", "Tool execution", ORANGE, "white"), ("verify", "Verifier", ORANGE_D, "white")]
    for v in versions:
        stacked(ax, v["version"], [(v[k] / v["n"] / 60, c, i) for k, _, c, i in parts], 0.62, ppu)
    ax.yaxis.set_major_formatter(fmt("{:.1f} min"))
    ax.set_ylabel("Mean time per rollout")
    version_axis(ax, versions)
    line_points(ax2, [v["version"] for v in versions], [v["max_concurrency"] for v in versions], INK, "{:.0f}")
    ax2.yaxis.set_major_formatter(fmt("{:.0f}"))
    ax2.set_ylabel("Max concurrent sandboxes")
    style(ax, ax2)
    legend(fig, [Patch(color=c, label=l) for _, l, c, _ in parts] + [
        Line2D([], [], color=INK, lw=2, marker="o", mec="white", label="Max concurrent sandboxes")], ncol=4)
    fig.subplots_adjust(left=0.06, right=0.955, top=0.87, bottom=0.1)
    fig.savefig(out / "sandbox-time.png")
    plt.close(fig)


def request_stats(a: dict, run: str, out: Path) -> None:
    """Small multiples: min-max range bar per version with the medians joined as a line, one panel per stat."""
    versions = a["versions"]
    xs = np.array([v["version"] for v in versions], dtype=float)
    fig, axes = figure(f"{run} · rollout stats per model version", (15, 8.5), nrows=3, sharex=True, gridspec_kw={"hspace": 0.15})
    stats = [("e2e_each", "Latency", 60, "{:.0f} min", BLUE), ("seqlen_each", "Sequence length", 1e3, "{:.0f}K", AQUA),
             ("turns_each", "Turns", 1, "{:.0f}", ORANGE)]
    for ax, (key, label, scale, template, color) in zip(axes, stats):
        values = [np.array(v[key]) / scale for v in versions]
        lo, hi = np.array([v.min() for v in values]), np.array([v.max() for v in values])
        ax.bar(xs, hi - lo, 0.42, bottom=lo, color=color, alpha=0.22, lw=0, zorder=2)
        line_points(ax, xs, [float(np.median(v)) for v in values], color, template)
        ax.set_ylim(0, max(hi) * 1.1)
        ax.yaxis.set_major_formatter(fmt(template))
        ax.set_ylabel(label)
        style(ax)
    version_axis(axes[-1], versions)
    fig.text(0.985, 0.95, "dot = median, bar = min to max", ha="right", va="top", fontsize=9, color=MUTED)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.92, bottom=0.08)
    fig.savefig(out / "rollout-stats.png")
    plt.close(fig)


def throughput(a: dict, run: str, out: Path) -> None:
    steps, t0 = a["steps"], a["run_start"]
    fig, (ax, bx) = figure(f"{run} · generator vs trainer throughput", (15, 10.5), nrows=2,
                           gridspec_kw={"height_ratios": [1, 1.25], "hspace": 0.35})
    xs = [s["step"] for s in steps]
    w = 0.38
    ax.bar([x - w / 2 for x in xs], [s["gen_tps"] / 1e3 for s in steps], w, color=AQUA, zorder=3)
    ax.bar([x + w / 2 for x in xs], [s["trainer_tps"] / 1e3 for s in steps], w, color=BLUE, zorder=3)
    for s in steps:
        for dx, val in ((-w / 2, s["gen_tps"]), (w / 2, s["trainer_tps"])):
            ax.text(s["step"] + dx, val / 1e3 + 0.3, f"{val / 1e3:.1f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_ylim(0, max(s["trainer_tps"] for s in steps) / 1e3 * 1.15)
    ax.yaxis.set_major_formatter(fmt("{:.0f}K"))
    ax.set_ylabel("Tokens/s")
    ax.set_xticks(xs)
    ax.set_xlim(0.4, len(xs) + 0.6)
    ax.set_xlabel("Step")
    style(ax)
    ax.legend(handles=[Patch(color=AQUA, label="Generator"), Patch(color=BLUE, label="Trainer")],
              loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=9.5, ncol=2)

    # Cumulative flow: produced (rollout completion, shaded by generating version) vs consumed (optimizer end).
    end = (a["run_end"] - t0) / 60
    top = nice_top(sum(s["tokens"] for s in steps) / 1e6 * 1.12)
    edges = [0.0] + [(p - t0) / 60 for p in a["publish"]] + [end]
    for v in range(len(edges) - 1):
        if v % 2:
            bx.axvspan(edges[v], edges[v + 1], color="#f1f1ee", lw=0, zorder=0)
        if edges[v + 1] - edges[v] > 4:
            bx.text((edges[v] + edges[v + 1]) / 2, top * 0.95, f"v{v}", ha="center", va="bottom", fontsize=7.5, color=MUTED)
    prod_x = [0.0] + [(p[0] - t0) / 60 for p in a["produced"]] + [end]
    prod_y = list(np.cumsum([0.0] + [p[1] / 1e6 for p in a["produced"]]))
    prod_y.append(prod_y[-1])
    bx.step(prod_x, prod_y, where="post", color=AQUA, lw=2.2, zorder=4)
    cons_x, cons_y, total = [0.0], [0.0], 0.0
    for s in steps:
        x = (s["consumed_at"] - t0) / 60
        cons_x += [x, x]
        cons_y += [total, total + s["tokens"] / 1e6]
        total += s["tokens"] / 1e6
    cons_x.append(end)
    cons_y.append(total)
    bx.plot(cons_x, cons_y, color=BLUE, lw=2, zorder=5)
    grid = np.linspace(0, end, 3000)
    prod = np.array([prod_y[np.searchsorted(prod_x, g, side="right") - 1] for g in grid])
    cons = np.array([cons_y[np.searchsorted(cons_x, g, side="right") - 1] for g in grid])
    bx.fill_between(grid, cons, prod, where=prod >= cons, color=AQUA, alpha=0.13, lw=0, zorder=2)
    for s in steps:
        if s["step"] % 5 == 0 or s["step"] == 1:
            bx.annotate(f"step {s['step']}", ((s["consumed_at"] - t0) / 60, cons_y[2 * s["step"]]), textcoords="offset points",
                        xytext=(4, -12), fontsize=7.5, color=INK2)
    bx.set_xlim(0, end)
    bx.set_ylim(0, top)
    bx.set_xlabel("Minutes")
    bx.set_ylabel("Cumulative tokens (M)")
    style(bx)
    bx.legend(handles=[Line2D([], [], color=AQUA, lw=2.2, label="Produced"),
                       Line2D([], [], color=BLUE, lw=2, label="Consumed"),
                       Patch(color=AQUA, alpha=0.25, label="Backlog")],
              loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=9.5, ncol=3)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.885, bottom=0.06)
    fig.savefig(out / "throughput-producer-consumer.png")
    plt.close(fig)


def eval_pass_avg(a: dict, run: str, out: Path) -> None:
    """pass@k = share of tasks solved at least once in k attempts; avg@k = mean reward over all k attempts."""
    evals = a["evals"]
    xs = [e["step"] for e in evals]
    k = len(next(iter(evals[0]["tasks"].values())))
    fig, ax = figure(f"{run} · eval pass@{k} and avg@{k}", (11, 6.2))
    rng = np.random.default_rng(0)
    for e in evals:
        values = [float(np.mean(r)) for r in e["tasks"].values()]
        ax.scatter(e["step"] + rng.uniform(-0.35, 0.35, len(values)), values, s=26, color=NEUTRAL, lw=0, zorder=2)
    line_points(ax, xs, [float(np.mean([max(r) >= 1 for r in e["tasks"].values()])) for e in evals], ORANGE, "{:.3f}")
    line_points(ax, xs, [float(np.mean([np.mean(r) for r in e["tasks"].values()])) for e in evals], BLUE, "{:.3f}", dy=-16)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xticks(xs)
    ax.set_xlim(-1.2, max(xs) + 1.2)
    ax.set_xlabel("Step")
    style(ax)
    legend(fig, [Line2D([], [], color=ORANGE, lw=2, marker="o", mec="white", label=f"pass@{k}"),
                 Line2D([], [], color=BLUE, lw=2, marker="o", mec="white", label=f"avg@{k}"),
                 Line2D([], [], color=NEUTRAL, marker="o", ls="", label=f"Per task avg@{k}")], ncol=3, y=0.925)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.11)
    fig.savefig(out / "eval-pass-avg.png")
    plt.close(fig)


CHARTS = (eval_pass_avg, step_time_mfu, rollout_time, sandbox_time, request_stats, throughput)


def render(run_dir: Path) -> list[Path]:
    a = json.loads((run_dir / "analysis" / "analysis.json").read_text())
    out = run_dir / "plots"
    out.mkdir(exist_ok=True)
    for stale in out.glob("*.png"):
        stale.unlink()
    for chart in CHARTS:
        chart(a, run_dir.name, out)
    return sorted(out.glob("*.png"))
