"""CLI: scan runs/ and build an index HTML linking to each run's report.

Usage:
    python -m eval.generate_index --runs-dir runs --eval-suffix eval_full_v5.json \
        --report-base-url /reports --output /nemo-workspace/.../mimic_runs.html
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path


_CSS = """
:root{--bg:#0d1117;--card:#161b22;--border:#30363d;--text:#e6edf3;--muted:#8b949e;--green:#3fb950;--red:#f85149;--yellow:#d29922;--blue:#58a6ff;--purple:#bc8cff}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:14px;line-height:1.55;padding:24px;max-width:1400px;margin:0 auto}
h1{font-size:22px;margin-bottom:4px;color:var(--blue)}
.subtitle{color:var(--muted);font-size:13px;margin-bottom:20px}
table{width:100%;border-collapse:collapse;background:var(--card);border-radius:8px;overflow:hidden}
th{background:#1c2128;color:var(--muted);font-size:10.5px;text-transform:uppercase;letter-spacing:.5px;padding:10px;text-align:left}
td{padding:9px 10px;border-top:1px solid var(--border);font-size:13px;vertical-align:top}
tr:hover td{background:#1c2128}
a{color:var(--blue);text-decoration:none}
a:hover{text-decoration:underline}
.tag{display:inline-block;padding:2px 8px;border-radius:11px;font-size:10.5px;font-weight:500;white-space:nowrap}
.tag-green{background:#0d2818;color:var(--green);border:1px solid #1a4d2e}
.tag-red{background:#2d1117;color:var(--red);border:1px solid #4d1f28}
.tag-yellow{background:#2d2200;color:var(--yellow);border:1px solid #4d3a00}
.tag-blue{background:#0d2140;color:var(--blue);border:1px solid #1a3d5c}
.tag-muted{background:#1c2128;color:var(--muted);border:1px solid var(--border)}
.c-green{color:var(--green)}.c-red{color:var(--red)}.c-yellow{color:var(--yellow)}.c-blue{color:var(--blue)}.c-muted{color:var(--muted)}
.sf-score{font-family:monospace;font-weight:600}
"""


def _e(x) -> str:
    if x is None: return "—"
    return html.escape(str(x))


def _pct_cls(pct):
    if pct is None: return "c-muted"
    if pct >= 75: return "c-green"
    if pct >= 50: return "c-yellow"
    return "c-red"


def _grade_tag(g):
    if g is None: return '<span class="tag tag-muted">—</span>'
    try: gi = int(g)
    except (TypeError, ValueError): return f'<span class="tag tag-muted">{_e(g)}</span>'
    cls = "tag-red" if gi <= 1 else "tag-yellow" if gi == 2 else "tag-green"
    return f'<span class="tag {cls}">grade {gi}</span>'


def _status_tag(s):
    if s in ("answer","success"): return f'<span class="tag tag-green">{_e(s)}</span>'
    if s == "abstain": return '<span class="tag tag-red">abstain</span>'
    if s is None: return '<span class="tag tag-muted">—</span>'
    return f'<span class="tag tag-yellow">{_e(s)}</span>'


def _collect(runs_dir: Path, eval_suffix: str):
    rows = []
    for rd in sorted(runs_dir.iterdir()):
        if not rd.is_dir():
            continue
        ev = rd / eval_suffix
        if not ev.exists():
            # Fallback: try any eval_full*_v5.json (e.g. stubbed variants)
            suffix_version = eval_suffix.replace("eval_full", "eval_full*").replace(".json", ".json")
            candidates = sorted(rd.glob(suffix_version))
            # Prefer non-stubbed
            candidates.sort(key=lambda p: ("stubbed" in p.name, p.name))
            if not candidates:
                continue
            ev = candidates[0]
        try:
            ej = json.loads(ev.read_text())
        except Exception:
            continue
        trace_path = rd / "trace.json"
        tr = {}
        if trace_path.exists():
            try: tr = json.loads(trace_path.read_text())
            except: pass
        cfg = tr.get("config", {}) or {}
        summ = tr.get("summary", {}) or {}
        ans_path = rd / "answer.json"
        ans = {}
        if ans_path.exists():
            try: ans = json.loads(ans_path.read_text())
            except: pass

        # D7 sub-fields
        d7 = (ej.get("dimensions") or {}).get("D7") or {}
        fs7 = d7.get("field_scores") or {}
        gc = fs7.get("golden_correctness")
        gc_score = gc.get("score") if isinstance(gc, dict) else None

        # agent estimate
        pe = ((ans.get("result") or {}) or {}).get("primary_effect") or {} if isinstance(ans.get("result"), dict) else {}
        agent_est = pe.get("estimate")

        # pull golden estimate from D7 reason if present
        golden_est = None
        if gc and isinstance(gc, dict):
            reason_str = gc.get("reason","")
            m = re.search(r"golden\s+IPW\s*=\s*([+-]?\d*\.?\d+)", reason_str)
            if not m:
                m = re.search(r"golden\s+effect[^(]*\(([+-]?\d*\.?\d+)\)", reason_str)
            if m:
                try: golden_est = float(m.group(1))
                except: pass

        rows.append({
            "run_id": rd.name,
            "model": cfg.get("model"),
            "status": summ.get("status"),
            "steps": summ.get("total_steps"),
            "wall_s": (summ.get("total_wall_clock_ms") or 0) / 1000 if summ.get("total_wall_clock_ms") else None,
            "overall": ej.get("overall_score"),
            "max_score": ej.get("max_score", 200),
            "pct": ej.get("percentage"),
            "d7_grade": d7.get("grade"),
            "gc_score": gc_score,
            "agent_est": agent_est,
            "golden_est": golden_est,
            "error": ej.get("error"),
            "start": cfg.get("start_time"),
        })
    rows.sort(key=lambda r: r["run_id"])
    return rows


def _fmt_est(v):
    if v is None: return "—"
    if isinstance(v, (int, float)): return f"{v:+.4f}"
    return str(v)


def _render(rows, report_base_url: str) -> str:
    tr = []
    for r in rows:
        run_id = r["run_id"]
        url = f"{report_base_url.rstrip('/')}/mimic_{run_id.replace('task_001_','')}"
        pct = r.get("pct")
        ov = r.get("overall")
        max_s = r["max_score"]
        if isinstance(pct, (int, float)):
            overall_cell = (
                f'<span class="sf-score {_pct_cls(pct)}">{ov:.0f}/{max_s}</span>'
                f'<div class="c-muted" style="font-size:11px">{pct:.1f}%</div>'
            )
        else:
            overall_cell = f'<span class="tag tag-red">0/{max_s}</span>'
        wall = r.get("wall_s")
        wall_cell = f"{wall:.0f}s" if isinstance(wall, (int, float)) else "—"
        tr.append(
            "<tr>"
            f"<td><a href='{_e(url)}'>{_e(run_id)}</a></td>"
            f"<td>{_e(r.get('model'))}</td>"
            f"<td>{_status_tag(r.get('status'))}</td>"
            f"<td class='sf-score'>{_e(r.get('steps'))}</td>"
            f"<td class='sf-score'>{wall_cell}</td>"
            f"<td>{overall_cell}</td>"
            f"<td>{_grade_tag(r.get('d7_grade'))}</td>"
            f"<td class='sf-score'>{_fmt_est(r.get('agent_est'))}</td>"
            f"<td class='sf-score'>{_fmt_est(r.get('golden_est'))}</td>"
            f"<td class='sf-score'>{_e(r.get('gc_score'))}</td>"
            "</tr>"
        )
    return (
        "<!DOCTYPE html>\n<html lang='en'><head><meta charset='UTF-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1.0'>"
        "<title>MIMIC agent runs — index</title>"
        f"<style>{_CSS}</style></head><body>"
        "<h1>MIMIC agent runs — index</h1>"
        f"<div class='subtitle'>{len(rows)} runs · click a run_id for the full report</div>"
        "<table><thead><tr>"
        "<th>Run</th><th>Model</th><th>Status</th><th>Steps</th><th>Wall</th>"
        "<th>Overall</th><th>D7 grade</th><th>Agent ATE</th><th>Golden ATE</th><th>GC</th>"
        "</tr></thead><tbody>"
        + "".join(tr)
        + "</tbody></table></body></html>"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--eval-suffix", default="eval_full_v5.json", help="Filename (under each run dir) to read")
    ap.add_argument("--report-base-url", default="/reports", help="URL base for per-run links (e.g. https://exp.evolverealty.uk/reports)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)

    rows = _collect(Path(args.runs_dir), args.eval_suffix)
    html_str = _render(rows, args.report_base_url)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_str)
    print(f"Wrote {out} ({len(rows)} runs)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
