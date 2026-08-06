"""Per-run HTML report renderer.

Produces a self-contained HTML page combining:
  1. Summary cards (model, steps, cost, status, overall %, grade, golden-correctness)
  2. Answer vs golden (primary effect, CI, direction + match badge)
  3. D1-D8 scorecard table + per-dimension sub-field detail (collapsible)
  4. Agent timeline (step-by-step thought/code/observation, collapsible)
  5. Appendix: artifact paths and final_decision

Entry point: build_report(run_dir, eval_json_path) -> str
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


_CSS = """
:root{--bg:#0d1117;--card:#161b22;--border:#30363d;--text:#e6edf3;--muted:#8b949e;--green:#3fb950;--red:#f85149;--yellow:#d29922;--blue:#58a6ff;--purple:#bc8cff;--orange:#d18616}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:14px;line-height:1.55;padding:24px;max-width:1200px;margin:0 auto}
h1{font-size:22px;margin-bottom:4px;color:var(--blue)}
h2{font-size:17px;margin:28px 0 10px;color:var(--purple);border-bottom:1px solid var(--border);padding-bottom:6px}
h3{font-size:13px;margin:14px 0 6px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
.subtitle{color:var(--muted);font-size:13px;margin-bottom:20px}
.grid{display:grid;gap:10px;margin:12px 0}
.grid-2{grid-template-columns:repeat(2,1fr)}
.grid-3{grid-template-columns:repeat(3,1fr)}
.grid-4{grid-template-columns:repeat(4,1fr)}
.grid-5{grid-template-columns:repeat(5,1fr)}
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:14px}
.card-label{font-size:10.5px;text-transform:uppercase;letter-spacing:.5px;color:var(--muted);margin-bottom:4px}
.card-value{font-size:22px;font-weight:600}
.card-sub{font-size:11.5px;color:var(--muted);margin-top:2px}
.c-green{color:var(--green)}.c-red{color:var(--red)}.c-yellow{color:var(--yellow)}.c-blue{color:var(--blue)}.c-purple{color:var(--purple)}.c-orange{color:var(--orange)}.c-muted{color:var(--muted)}
table{width:100%;border-collapse:collapse;margin:10px 0;background:var(--card);border-radius:8px;overflow:hidden}
th{background:#1c2128;color:var(--muted);font-size:10.5px;text-transform:uppercase;letter-spacing:.5px;padding:9px 11px;text-align:left}
td{padding:7px 11px;border-top:1px solid var(--border);font-size:12.5px;vertical-align:top}
tr:hover td{background:#1c2128}
.code-box{background:#0d1117;border:1px solid var(--border);border-radius:6px;padding:10px;margin:6px 0;font-family:'SF Mono',Monaco,Consolas,monospace;font-size:11.5px;white-space:pre-wrap;word-break:break-word;max-height:360px;overflow-y:auto;line-height:1.45;color:var(--text)}
.code-box.stderr{border-color:#4d1f28;background:#1f1518}
.code-box.thought{border-color:#3a1a5c;background:#1a1325}
.code-box.code{border-color:#1a4d2e}
.tag{display:inline-block;padding:2px 8px;border-radius:11px;font-size:10.5px;font-weight:500;white-space:nowrap}
.tag-ok,.tag-green{background:#0d2818;color:var(--green);border:1px solid #1a4d2e}
.tag-fail,.tag-red{background:#2d1117;color:var(--red);border:1px solid #4d1f28}
.tag-warn,.tag-yellow{background:#2d2200;color:var(--yellow);border:1px solid #4d3a00}
.tag-info,.tag-blue{background:#0d2140;color:var(--blue);border:1px solid #1a3d5c}
.tag-purple{background:#1f0d30;color:var(--purple);border:1px solid #3a1a5c}
.tag-muted{background:#1c2128;color:var(--muted);border:1px solid var(--border)}
details{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:0;margin:10px 0}
details > summary{padding:10px 14px;cursor:pointer;font-weight:500;user-select:none;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
details > summary:hover{background:#1c2128}
details[open] > summary{border-bottom:1px solid var(--border)}
details .details-body{padding:10px 14px}
.insight{background:#1a1f2e;border-left:3px solid var(--blue);padding:10px 14px;margin:10px 0;border-radius:0 6px 6px 0;font-size:13px}
.insight-red{border-left-color:var(--red);background:#1f1518}
.insight-yellow{border-left-color:var(--yellow);background:#1f1c12}
.insight-green{border-left-color:var(--green);background:#0f1f15}
.step-row{display:grid;grid-template-columns:60px 110px 1fr auto;gap:12px;align-items:center}
.step-num{font-family:monospace;color:var(--muted);font-size:12px}
.step-meta{font-size:11px;color:var(--muted)}
.sf-score{font-family:monospace;font-weight:600}
.hbar{display:inline-block;height:14px;border-radius:2px;vertical-align:middle;margin-right:6px}
.notes{font-size:12px;color:var(--muted);margin-top:4px}
pre{white-space:pre-wrap;word-break:break-word}
.kv-table td:first-child{color:var(--muted);font-size:12px;width:160px;white-space:nowrap}
.kv-table td{padding:5px 10px}
a{color:var(--blue)}
@media(max-width:800px){
.grid-3,.grid-4,.grid-5{grid-template-columns:repeat(2,1fr)}
body{padding:12px;font-size:13px}
h1{font-size:18px}h2{font-size:15px}
.card-value{font-size:18px}
.step-row{grid-template-columns:40px 90px 1fr}
}
@media(max-width:480px){
.grid-2,.grid-3,.grid-4,.grid-5{grid-template-columns:1fr}
}
"""


def _e(x: Any) -> str:
    """HTML-escape any value (None -> '—')."""
    if x is None:
        return "—"
    return html.escape(str(x))


def _fmt_score(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        return f"{v:.2f}"
    return str(v)


def _score_tag(score: float | int | None, weight: float | int | None = None) -> str:
    if score is None:
        return '<span class="tag tag-muted">—</span>'
    if score < 0:
        return f'<span class="tag tag-muted">{score:.2f} (excluded)</span>'
    pct = score * 100 if score <= 1.0 else score
    if pct >= 80:
        cls = "tag-green"
    elif pct >= 50:
        cls = "tag-yellow"
    else:
        cls = "tag-red"
    return f'<span class="tag {cls}">{score:.2f}</span>'


def _grade_tag(g: Any) -> str:
    if g is None:
        return '<span class="tag tag-muted">—</span>'
    try:
        gi = int(g)
    except (TypeError, ValueError):
        return f'<span class="tag tag-muted">{_e(g)}</span>'
    cls = "tag-red" if gi <= 1 else "tag-yellow" if gi == 2 else "tag-green"
    return f'<span class="tag {cls}">grade {gi}</span>'


def _pct_color(pct: float | None) -> str:
    if pct is None:
        return "c-muted"
    if pct >= 75:
        return "c-green"
    if pct >= 50:
        return "c-yellow"
    return "c-red"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _head(title: str) -> str:
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="UTF-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1.0">'
        f"<title>{_e(title)}</title>"
        f"<style>{_CSS}</style></head><body>"
    )


def _render_summary_cards(
    run_id: str,
    run_dir: Path,
    answer: dict,
    trace: dict,
    eval_json: dict,
) -> str:
    cfg = (trace or {}).get("config", {}) or {}
    summ = (trace or {}).get("summary", {}) or {}
    model = cfg.get("model", "—")
    steps = summ.get("total_steps", "—")
    tokens = summ.get("total_tokens")
    tok_str = f"{tokens:,}" if isinstance(tokens, int) else "—"
    wall_ms = summ.get("total_wall_clock_ms")
    wall_str = f"{wall_ms/1000:.1f}s" if isinstance(wall_ms, int) else "—"
    # cost: rough estimate from tokens, gpt-5-mini ~ $0.40/$1.60 per 1M in/out
    p_tok = summ.get("total_prompt_tokens", 0) or 0
    c_tok = summ.get("total_completion_tokens", 0) or 0
    cost = p_tok * 0.4e-6 + c_tok * 1.6e-6 if "gpt-5-mini" in str(model).lower() else None
    cost_str = f"${cost:.2f}" if cost is not None else "—"
    status = summ.get("status", "unknown")
    conf = summ.get("confidence", "—")

    pct = eval_json.get("percentage")
    overall = eval_json.get("overall_score")
    max_s = eval_json.get("max_score", 200)
    dims = eval_json.get("dimensions", {}) or {}
    # D7 golden-correctness if available
    gc = None
    try:
        gc = dims.get("D7", {}).get("field_scores", {}).get("golden_correctness", {}).get("score")
    except Exception:
        pass

    err = eval_json.get("error")

    status_cls = "tag-red" if status == "abstain" else "tag-green" if status in ("answer","success") else "tag-yellow"
    overall_cls = _pct_color(pct)
    gc_cls = _score_tag(gc) if gc is not None else '<span class="tag tag-muted">n/a</span>'

    parts = [f'<div class="grid grid-5">']
    parts.append(f'<div class="card"><div class="card-label">Model</div><div class="card-value c-blue" style="font-size:15px">{_e(model)}</div><div class="card-sub">temp {cfg.get("temperature","—")} · max {cfg.get("max_steps","—")}</div></div>')
    parts.append(f'<div class="card"><div class="card-label">Wall clock</div><div class="card-value">{wall_str}</div><div class="card-sub">{_e(steps)} steps · {tok_str} tok · {cost_str}</div></div>')
    parts.append(f'<div class="card"><div class="card-label">Status</div><div class="card-value"><span class="tag {status_cls}">{_e(status)}</span></div><div class="card-sub">confidence: {_e(conf)}</div></div>')
    if overall is not None and max_s:
        parts.append(f'<div class="card"><div class="card-label">Overall</div><div class="card-value {overall_cls}">{overall:.0f}/{max_s}</div><div class="card-sub">{pct:.1f}%</div></div>')
    else:
        parts.append(f'<div class="card"><div class="card-label">Overall</div><div class="card-value c-red">0/{max_s}</div><div class="card-sub">not scored</div></div>')
    parts.append(f'<div class="card"><div class="card-label">D7 golden_correctness</div><div class="card-value">{gc_cls}</div><div class="card-sub">vs golden IPW ATE</div></div>')
    parts.append('</div>')

    if err:
        parts.append(f'<div class="insight insight-red"><strong>Hard-gate error:</strong> <code>{_e(err)}</code></div>')

    return "\n".join(parts)


def _render_answer_vs_golden(answer: dict, eval_json: dict, golden: dict | None) -> str:
    """Primary effect + golden ATE side-by-side."""
    result = (answer or {}).get("result") or {}
    pe = (result.get("primary_effect") if isinstance(result, dict) else None) or {}
    agent_est = pe.get("estimate")
    agent_dir = pe.get("direction")
    agent_ci_lo = pe.get("ci_95_lower") or pe.get("ci_lower")
    agent_ci_hi = pe.get("ci_95_upper") or pe.get("ci_upper")
    agent_measure = pe.get("measure") or result.get("measure")
    final = (answer or {}).get("final_decision") or {}

    # golden: from D7 golden_correctness reason if available, else from golden.json
    golden_ate = None
    golden_dir = None
    d7 = eval_json.get("dimensions", {}).get("D7", {}) or {}
    gc_field = d7.get("field_scores", {}).get("golden_correctness", {}) or {}
    reason = gc_field.get("reason", "")
    # try parse "golden IPW=+X.XXX" from reason string
    import re
    m = re.search(r"golden\s+IPW\s*=\s*([+-]?\d*\.?\d+)", reason)
    if not m:
        m = re.search(r"golden\s+effect[^(]*\(([+-]?\d*\.?\d+)\)", reason)
    if m:
        try: golden_ate = float(m.group(1))
        except: pass
    if golden is not None:
        gr = (golden.get("result_golden") or {}).get("primary_effect") or {}
        if golden_ate is None:
            golden_ate = gr.get("estimate")
        golden_dir = gr.get("direction") or "null" if (golden_ate is not None and abs(golden_ate) < 0.05) else gr.get("direction")

    rows = []
    def row(label, a, g):
        return f"<tr><td>{_e(label)}</td><td>{_e(a)}</td><td>{_e(g)}</td></tr>"
    rows.append(row("measure", agent_measure, "ATE (IPW Horvitz-Thompson)"))
    rows.append(row("estimate", f"{agent_est:+.4f}" if isinstance(agent_est,(int,float)) else agent_est,
                    f"{golden_ate:+.4f}" if isinstance(golden_ate,(int,float)) else golden_ate))
    if isinstance(agent_ci_lo,(int,float)) and isinstance(agent_ci_hi,(int,float)):
        rows.append(row("95% CI", f"[{agent_ci_lo:+.4f}, {agent_ci_hi:+.4f}]", "[-0.050, +0.069]"))
    rows.append(row("direction", agent_dir, golden_dir or "null"))

    gc_score = gc_field.get("score")
    match_badge = _score_tag(gc_score) if gc_score is not None else '<span class="tag tag-muted">not scored</span>'

    out = [f'<h2>2. Answer vs golden</h2>']
    out.append(f'<div class="insight"><strong>D7 golden_correctness:</strong> {match_badge} &nbsp;&nbsp;<span class="c-muted">{_e(reason)}</span></div>')
    out.append('<div class="table-wrap"><table><thead><tr><th>Field</th><th>Agent</th><th>Golden</th></tr></thead><tbody>')
    out.extend(rows)
    out.append('</tbody></table></div>')
    if final:
        if final.get("status") == "abstain":
            out.append(f'<div class="insight insight-yellow"><strong>Agent abstained.</strong> {_e(final.get("abstain_reason",""))}</div>')
        else:
            interp = pe.get("interpretation") or pe.get("narrative") or ""
            if interp:
                out.append(f'<div class="insight"><strong>Interpretation:</strong> {_e(interp)}</div>')
    return "\n".join(out)


DIM_NAMES = {
    "D1": "Estimand",
    "D2": "Cohort",
    "D3": "Temporal",
    "D4": "Variables",
    "D5": "Method",
    "D6": "Diagnostics",
    "D7": "Quality",
    "D8": "Reasoning",
}


def _render_scorecard(eval_json: dict) -> str:
    dims = eval_json.get("dimensions", {}) or {}
    if not dims:
        return '<h2>3. Dimension scorecard</h2><div class="insight insight-red">No dimension scores — hard-gate failed.</div>'
    rows = []
    for key in ["D1","D2","D3","D4","D5","D6","D7","D8"]:
        d = dims.get(key) or {}
        grade = d.get("grade")
        ws = d.get("weighted_score")
        mfs = d.get("mean_field_score")
        hg = d.get("hard_gate_failed")
        notes = (d.get("notes") or "").strip()
        hg_tag = '<span class="tag tag-red">hard-gate</span>' if hg else ""
        rows.append(
            f"<tr><td><strong>{key}</strong> {_e(DIM_NAMES.get(key,''))}</td>"
            f"<td>{_grade_tag(grade)}</td>"
            f"<td class='sf-score'>{_e(f'{ws:.1f}' if isinstance(ws,(int,float)) else ws)}</td>"
            f"<td class='sf-score'>{_fmt_score(mfs)}</td>"
            f"<td>{hg_tag}</td>"
            f"<td class='notes'>{_e(notes)[:180]}</td></tr>"
        )
    table = (
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Dimension</th><th>Grade</th><th>Weighted</th><th>Mean field</th><th>Gate</th><th>Notes</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )

    # per-dimension sub-field detail
    detail_blocks = []
    for key in ["D1","D2","D3","D4","D5","D6","D7","D8"]:
        d = dims.get(key) or {}
        fs = d.get("field_scores") or {}
        if not fs:
            continue
        sub_rows = []
        for fn, fv in fs.items():
            if isinstance(fv, dict):
                sc = fv.get("score")
                rsn = fv.get("reason", "") or fv.get("notes","")
                dq = " <span class='tag tag-muted'>disqualified</span>" if fv.get("disqualified") else ""
            else:
                sc = fv; rsn = ""; dq = ""
            sub_rows.append(
                f"<tr><td>{_e(fn)}</td><td>{_score_tag(sc)}{dq}</td>"
                f"<td class='notes'>{_e(rsn)}</td></tr>"
            )
        grade = d.get("grade"); ws = d.get("weighted_score"); mfs = d.get("mean_field_score")
        header_bits = [
            f"<strong>{key}</strong> {_e(DIM_NAMES.get(key,''))}",
            _grade_tag(grade),
            f"<span class='step-meta'>weighted {ws:.1f} · mean {mfs:.2f}</span>" if isinstance(ws,(int,float)) and isinstance(mfs,(int,float)) else "",
        ]
        detail_blocks.append(
            "<details><summary>" + " ".join(h for h in header_bits if h) + "</summary>"
            "<div class='details-body'>"
            "<table><thead><tr><th>Sub-field</th><th>Score</th><th>Reason / notes</th></tr></thead><tbody>"
            + "".join(sub_rows) +
            "</tbody></table>"
            + (f"<div class='notes' style='margin-top:8px'><strong>Dimension notes:</strong> {_e(d.get('notes',''))}</div>" if d.get("notes") else "")
            + "</div></details>"
        )
    return (
        "<h2>3. Dimension scorecard</h2>"
        + table
        + "<h3>Per-dimension sub-field detail (click to expand)</h3>"
        + "".join(detail_blocks)
    )


def _render_estimand_cohort(answer: dict) -> str:
    est = (answer or {}).get("estimand") or {}
    coh = (answer or {}).get("cohort") or {}
    if not est and not coh:
        return ""
    est_rows = []
    for k in ["population","treatment","comparator","outcome","time_zero","time_horizon","effect_type"]:
        if k in est:
            est_rows.append(f"<tr><td>{_e(k)}</td><td>{_e(est[k])[:500]}</td></tr>")
    coh_bits = []
    for k in ["n_treated","n_control","inclusion_criteria","exclusion_criteria"]:
        if k in coh:
            v = coh[k]
            if isinstance(v, list):
                v = " · ".join(str(x)[:120] for x in v[:6]) + (" …" if len(v) > 6 else "")
            coh_bits.append(f"<tr><td>{_e(k)}</td><td>{_e(v)[:600]}</td></tr>")
    out = ["<h2>1. Estimand & cohort (as declared by the agent)</h2>"]
    if est_rows:
        out.append('<h3>Estimand</h3><table class="kv-table"><tbody>' + "".join(est_rows) + "</tbody></table>")
    if coh_bits:
        out.append('<h3>Cohort</h3><table class="kv-table"><tbody>' + "".join(coh_bits) + "</tbody></table>")
    return "\n".join(out)


def _render_timeline(trace: dict, run_log_text: str | None) -> str:
    steps = (trace or {}).get("steps") or []
    if not steps:
        return ""
    # Group by step_num -> {thought, code, observation}
    grouped: dict[int, dict[str, Any]] = {}
    for s in steps:
        n = s.get("step")
        grouped.setdefault(n, {"step": n})[s.get("entry_type")] = s
    ordered = sorted(grouped.items())
    blocks = []
    for n, g in ordered:
        thought = g.get("thought")
        code = g.get("code")
        obs = g.get("observation")
        submit = g.get("submit")
        # time taken: from thought entry if present
        t_ms = (thought or {}).get("wall_clock_ms") or 0
        prompt = (thought or {}).get("prompt_tokens") or 0
        comp = (thought or {}).get("completion_tokens") or 0
        ts = (thought or {}).get("timestamp", "")

        summary_parts = [f'<span class="step-num">step {n}</span>']
        if thought:
            summary_parts.append(f'<span class="tag tag-purple">thought</span>')
        if code:
            summary_parts.append(f'<span class="tag tag-info">code</span>')
        if obs:
            # check if stderr had traceback
            content = (obs.get("content") or "")
            is_err = "Traceback" in content or "Error" in content[:1000]
            summary_parts.append(f'<span class="tag {"tag-red" if is_err else "tag-green"}">obs</span>')
        if submit:
            summary_parts.append(f'<span class="tag tag-yellow">submit</span>')
        meta_bits = []
        if ts: meta_bits.append(f'{_e(ts)}')
        if t_ms: meta_bits.append(f'{t_ms/1000:.1f}s')
        if prompt or comp: meta_bits.append(f'{prompt:,}p/{comp:,}c tok')
        summary_parts.append(f'<span class="step-meta">{" · ".join(meta_bits)}</span>')
        # Preview: first line of thought content
        first_line = ""
        if thought and thought.get("content"):
            for line in thought["content"].splitlines():
                l = line.strip()
                if l and not l.startswith("<code>") and not l.startswith("</code>"):
                    first_line = l[:140]
                    break
        if first_line:
            summary_parts.append(f'<span class="step-meta" style="flex:1">› {_e(first_line)}</span>')

        body_parts = []
        if thought:
            c = thought.get("content","")
            body_parts.append(f'<h3>Thought</h3><div class="code-box thought">{_e(c)}</div>')
        if code:
            c = code.get("content","")
            body_parts.append(f'<h3>Code</h3><div class="code-box code">{_e(c)}</div>')
        if obs:
            c = obs.get("content","")
            # split stdout / stderr if possible
            if "stderr:" in c and "stdout:" in c:
                parts = c.split("stderr:", 1)
                so = parts[0].replace("stdout:","").strip()
                se = parts[1].strip() if len(parts) > 1 else ""
                if so:
                    body_parts.append(f'<h3>stdout</h3><div class="code-box">{_e(so[:8000])}</div>')
                if se:
                    body_parts.append(f'<h3>stderr</h3><div class="code-box stderr">{_e(se[:8000])}</div>')
            else:
                body_parts.append(f'<h3>Observation</h3><div class="code-box">{_e(c[:8000])}</div>')
        if submit:
            body_parts.append(f'<h3>Submit</h3><div class="code-box thought">{_e((submit.get("content") or "")[:4000])}</div>')
        blocks.append(
            "<details><summary>" + " ".join(summary_parts) + "</summary>"
            "<div class='details-body'>" + "".join(body_parts) + "</div></details>"
        )
    return "<h2>4. Agent timeline</h2>" + "".join(blocks)


def _render_appendix(run_dir: Path, trace: dict, answer: dict) -> str:
    files = []
    for name in ["answer.json","trace.json","conversation.json","run.log","system_prompt.txt","analysis.py"]:
        p = run_dir / name
        if p.exists():
            sz = p.stat().st_size
            files.append(f"<li><code>{_e(p.name)}</code> <span class='step-meta'>({sz:,} B)</span></li>")
    return (
        "<h2>5. Appendix</h2>"
        "<h3>Run artifacts</h3>"
        f"<ul>{''.join(files)}</ul>"
    )


def build_report(run_dir: str | Path, eval_json_path: str | Path, golden_path: str | Path | None = None) -> str:
    run_dir = Path(run_dir)
    answer = _load_json(run_dir / "answer.json") or {}
    trace = _load_json(run_dir / "trace.json") or {}
    eval_json = _load_json(Path(eval_json_path)) or {}
    golden = _load_json(Path(golden_path)) if golden_path else None

    run_id = run_dir.name  # e.g. task_001_run26
    cfg = trace.get("config", {}) or {}
    model = cfg.get("model", "?")
    start = cfg.get("start_time","")
    end = cfg.get("end_time","")
    status = (trace.get("summary") or {}).get("status", "?")

    title = f"MIMIC {run_id} — agent run report"
    subtitle = f"{_e(model)} · {_e(start)} → {_e(end)} · status <span class='tag tag-muted'>{_e(status)}</span>"

    sections = [
        _head(title),
        f'<h1>{_e(title)}</h1>',
        f'<div class="subtitle">{subtitle}</div>',
        _render_summary_cards(run_id, run_dir, answer, trace, eval_json),
        _render_estimand_cohort(answer),
        _render_answer_vs_golden(answer, eval_json, golden),
        _render_scorecard(eval_json),
        _render_timeline(trace, None),
        _render_appendix(run_dir, trace, answer),
        "</body></html>",
    ]
    return "\n".join(s for s in sections if s)
