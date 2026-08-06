"""CLI: generate an HTML report for a single run.

Usage:
    python -m eval.generate_run_report --run-dir runs/task_001_run26 \
        --eval runs/task_001_run26/eval_full_v5.json \
        --golden tasks/task_001_diuretic_fluid/golden.json \
        --output /nemo-workspace/inf-evolve/home/.cache/ui_host/reports/mimic_run26.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from eval.run_report_html import build_report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True, help="Run directory (e.g. runs/task_001_run26)")
    ap.add_argument("--eval", required=True, help="Path to eval JSON (e.g. runs/…/eval_full_v5.json)")
    ap.add_argument("--golden", default=None, help="Optional golden.json path for cross-ref")
    ap.add_argument("--output", required=True, help="Output HTML path")
    args = ap.parse_args(argv)

    html = build_report(args.run_dir, args.eval, args.golden)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"Wrote {out} ({len(html):,} chars)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
