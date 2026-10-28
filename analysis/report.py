#!/usr/bin/env python3
"""Post-run report: trial extraction -> per-step/per-version stats -> PNG plots.

Run with the cookbook venv python (needs the fireworks SDK and matplotlib)::

  python analysis/report.py runs/<run-id>            # reuse analysis/trials.jsonl if present
  python analysis/report.py runs/<run-id> --reextract

Writes ``runs/<run-id>/analysis/{trials.jsonl,analysis.json}`` and
``runs/<run-id>/plots/*.png``. Needs the run's COOKBOOK_METRICS_FILE
(``metrics.jsonl``) and COOKBOOK_TRACE_FILE (``client-phase-trace.json``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from analyze import analyze  # noqa: E402
from extract_trials import extract  # noqa: E402
from plots import render  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--reextract", action="store_true", help="re-read every trial artifact (~90 s per 1k trials)")
    args = parser.parse_args()
    if args.reextract or not (args.run_dir / "analysis" / "trials.jsonl").exists():
        print("extracted", extract(args.run_dir))
    print("analyzed", analyze(args.run_dir))
    for path in render(args.run_dir):
        print("plot", path)


if __name__ == "__main__":
    main()
