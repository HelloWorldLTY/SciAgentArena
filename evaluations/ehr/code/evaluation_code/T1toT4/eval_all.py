"""
MedAgentBench — Evaluation Script (T1–T4)

T1  stepwise_v2_20      FHIR Workflow           N=20   action-list F1
T2  stepwise_all_435    Marker Verification     N=435  action-list F1
    stepwise_curated20  Marker Verif. (curated) N=20   action-list F1
T3  rare_disease_20     Rare Disease            N=20   action-list F1
T4  mimic_drug          Drug Recommendation     N=10   fuzzy drug-name F1
T5  causal_002          Causal Inference        N= 1   D1-D8 weighted score

Pass threshold (T1–T3): F1 >= 0.5 per case
Pass threshold (T4):    F1 >= 0.5 per case (fuzzy SequenceMatcher >= 0.82)
Pass threshold (T5):    percentage >= 50% (overall weighted score)

Usage:
    python eval_all.py <model_dir>
    T1-T4 file names must match: {prefix}_{group}.jsonl  (prefix = any string)
    T5 file name: answer.json, *_causal_002.json, or causal_002.json
    T5 requires:
      OPENAI_API_KEY (or + OPENAI_BASE_URL for OpenRouter)
      MIMIC_MAIN      path to the mimic-main project root
      MIMIC_DATA_ROOT path to the MIMIC-IV demo data root
      T5_GOLDEN       (optional) path to golden.json; defaults to
                      $MIMIC_MAIN/tasks/task_002_diuretic_demo/golden.json
"""

from __future__ import annotations
import json, os, re, sys
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).parent
PASS_F1 = 0.5


def _gt(env_var: str) -> Path:
    """Resolve a ground-truth file path from an environment variable.

    Set each variable to the absolute path of the corresponding file:
      GT_STEPWISE_V2_20     – T1 ground truth
      GT_STEPWISE_ALL_435   – T2a ground truth
      GT_STEPWISE_CURATED20 – T2b ground truth
      GT_RARE_DISEASE_20    – T3 tasks file (GT embedded)
      GT_MIMIC_DRUG         – T4 tasks file (GT embedded)

    Raises EnvironmentError at call time if the variable is not set.
    """
    v = os.environ.get(env_var)
    if not v:
        raise EnvironmentError(f"Set environment variable {env_var} to the ground-truth file path")
    return Path(v)


GT: dict[str, str] = {
    "stepwise_v2_20":     "GT_STEPWISE_V2_20",
    "stepwise_all_435":   "GT_STEPWISE_ALL_435",
    "stepwise_curated20": "GT_STEPWISE_CURATED20",
    "rare_disease_20":    "GT_RARE_DISEASE_20",   # GT embedded in tasks file
    "mimic_drug":         "GT_MIMIC_DRUG",         # GT embedded in tasks file
}
# rare_disease uses "id" as the task key; all others use "task_id"
GT_ID_KEY = {
    "rare_disease_20": "id",
}


# ── T1 / T2 / T3 : Action-list F1 ──────────────────────────────────────────

_BRACKET = re.compile(
    r"\[(?:Step|STEP|HA|FHIR)\s*\d*\]\s*(?:Step\s+\d+\s*[:\-]\s*)?([a-zA-Z_][a-zA-Z0-9_]*)",
    re.IGNORECASE,
)

def extract_actions(text: str) -> list[str]:
    """Extract predicted FHIR action names from model response.

    Tries JSON {"steps": [{"action": ...}]} first, then falls back to
    bracket format [Step N] action_name.
    """
    text = (text or "").strip()

    # strip markdown code fences
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()

    obj: dict | list = {}
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m2 = re.search(r"\{[\s\S]*\}", text)
        if m2:
            try: obj = json.loads(m2.group(0))
            except json.JSONDecodeError: pass

    if isinstance(obj, dict) and isinstance(obj.get("steps"), list):
        actions = [x["action"] for x in obj["steps"] if isinstance(x, dict) and x.get("action")]
        if actions:
            return [a.lower().strip() for a in actions]

    if isinstance(obj, list):
        actions = [x["action"] for x in obj if isinstance(x, dict) and x.get("action")]
        if actions:
            return [a.lower().strip() for a in actions]

    # bracket fallback: [Step 1] fhir_search_observation …
    return [m.lower().strip() for m in _BRACKET.findall(text)]


def stepwise_score(gt_steps: list, response: str) -> dict:
    """Bidirectional substring matching → precision / recall / F1.

    p matches g iff g in p or p in g (tolerates minor prefix/suffix variation).
    """
    gt   = [str(s.get("action","")).lower() for s in gt_steps if s.get("action")]
    pred = extract_actions(response)

    if not gt:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "any_correct": 0.0}

    hit_gt   = sum(1 for g in gt   if any(g in p or p in g for p in pred))
    hit_pred = sum(1 for p in pred if any(g in p or p in g for g in gt)) if pred else 0

    recall    = hit_gt   / len(gt)
    precision = hit_pred / len(pred) if pred else 0.0
    f1 = 2*precision*recall / (precision+recall) if (precision+recall) else 0.0

    return {
        "precision":   round(precision, 4),
        "recall":      round(recall,    4),
        "f1":          round(f1,        4),
        "any_correct": 1.0 if hit_gt > 0 else 0.0,
        "n_gt":        len(gt),
        "n_pred":      len(pred),
    }


def run_stepwise(group: str, result_file: Path) -> None:
    gt_rows = json.loads(_gt(GT[group]).read_text())
    id_key = GT_ID_KEY.get(group, "task_id")
    gt_map = {}
    for row in gt_rows:
        steps = row.get("expected_steps", [])
        # normalise: bare strings → dicts
        if steps and isinstance(steps[0], str):
            steps = [{"action": s} for s in steps]
        gt_map[row[id_key]] = steps

    scores = []
    for line in result_file.read_text().splitlines():
        if not line.strip(): continue
        row      = json.loads(line)
        task_id  = row.get("task_id", row.get("id", ""))
        response = row.get("model_response", row.get("response", ""))
        gt_steps = gt_map.get(task_id, row.get("expected_steps", []))
        if gt_steps and isinstance(gt_steps[0], str):
            gt_steps = [{"action": s} for s in gt_steps]
        if not isinstance(gt_steps, list) or not gt_steps:
            continue  # skip rows without usable GT
        scores.append(stepwise_score(gt_steps, response))

    n      = len(scores)
    acc    = sum(1 for s in scores if s["f1"] >= PASS_F1)
    f1s    = [s["f1"]        for s in scores]
    precs  = [s["precision"] for s in scores]
    recs   = [s["recall"]    for s in scores]
    any_c  = sum(s["any_correct"] for s in scores)

    print(f"  {group:30s}  N={n:3d}  acc={acc}/{n} ({100*acc/n:5.1f}%)  "
          f"F1={mean(f1s):.3f}  P={mean(precs):.3f}  R={mean(recs):.3f}  "
          f"AnyCorrect={any_c}/{n}")


# ── T4 : Drug Recommendation (fuzzy name matching) ──────────────────────────

def _norm(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\([^)]*\)", " ", text).replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def drug_match(gt_name: str, pred_name: str, threshold: float = 0.82) -> bool:
    ng, np = _norm(gt_name), _norm(pred_name)
    if not ng or not np: return False
    if ng == np or ng in np or np in ng: return True
    return SequenceMatcher(None, ng, np).ratio() >= threshold

def gt_drug_names(meds: list) -> list[str]:
    """Extract bare drug names from GT list; entries may be 'DrugName — dose — route'."""
    return [re.split(r"\s+[—–-]\s+", e, 1)[0].strip() for e in meds if isinstance(e, str)]

def pred_drug_names(text: str) -> list[str]:
    """Extract candidate drug names from free-text model response.

    Strips <thinking> blocks, then parses numbered / bullet lists,
    filtering out boilerplate lines.
    """
    if not text: return []
    text = re.sub(r"<thinking>[\s\S]*?</thinking>", "", text, flags=re.I)

    numbered = re.findall(r"(?m)^\s*\d{1,2}[\.)]\s*(.+)", text)
    cands = [x.strip() for x in numbered] if numbered else (
        [x.strip() for x in re.findall(r"(?m)^[-*•]\s*(.+)", text)]
        or [l.strip() for l in text.split("\n") if l.strip()]
    )

    drugs = []
    for c in cands:
        c = re.sub(r"\*\*(.*?)\*\*", r"\1", c)          # strip bold markdown
        c = re.split(r"\s+[—–-]\s+", c, 1)[0]          # drop dose/route suffix
        c = re.split(r"\s*:\s+", c, 1)[0]               # drop after colon
        c = re.sub(r"^\d+[\.)]\s*", "", c).strip(" \t:-.") # strip leading number
        if not c or len(c) > 100: continue
        # skip obvious non-drug lines
        low = c.lower()
        if any(low.startswith(k) for k in ["medication","drug","prescription","recommendation",
                "based on","the patient","given the","note:","here","following"]):
            continue
        if re.search(r"\b(should|would|could|patient|history|given)\b", low): continue
        drugs.append(c)
    return drugs


def drug_score(gt_meds: list, response: str) -> dict:
    """Compute F1 with fuzzy drug-name matching (SequenceMatcher >= 0.82)."""
    gt   = gt_drug_names(gt_meds)
    pred = pred_drug_names(response)

    if not gt:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "any_correct": 0.0}

    tp = sum(1 for g in gt   if any(drug_match(g, p) for p in pred))
    fp = sum(1 for p in pred if not any(drug_match(g, p) for g in gt))
    fn = sum(1 for g in gt   if not any(drug_match(g, p) for p in pred))

    recall    = tp / len(gt)
    precision = tp / len(pred) if pred else 0.0
    f1 = 2*precision*recall / (precision+recall) if (precision+recall) else 0.0

    return {
        "precision":   round(precision, 4),
        "recall":      round(recall,    4),
        "f1":          round(f1,        4),
        "any_correct": 1.0 if tp > 0 else 0.0,
        "n_gt":        len(gt),
        "n_pred":      len(pred),
    }


def run_drug(result_file: Path) -> None:
    drug_gt_map = {
        r["task_id"]: r["ground_truth_medications"]
        for r in json.loads(_gt(GT["mimic_drug"]).read_text())
    }
    scores = []
    for line in result_file.read_text().splitlines():
        if not line.strip(): continue
        row      = json.loads(line)
        task_id  = row.get("task_id", "")
        gt_meds  = drug_gt_map.get(task_id, row.get("gt_medications", []))
        response = row.get("model_response", row.get("response", ""))
        s        = drug_score(gt_meds, response)
        scores.append(s)

    n     = len(scores)
    acc   = sum(1 for s in scores if s["f1"] >= PASS_F1)
    f1s   = [s["f1"]        for s in scores]
    precs = [s["precision"] for s in scores]
    recs  = [s["recall"]    for s in scores]
    any_c = sum(s["any_correct"] for s in scores)

    print(f"  {'mimic_drug':30s}  N={n:3d}  acc={acc}/{n} ({100*acc/n:5.1f}%)  "
          f"F1={mean(f1s):.3f}  P={mean(precs):.3f}  R={mean(recs):.3f}  "
          f"AnyCorrect={any_c}/{n}")


# ── T5 : Causal Inference (D1-D8, task_002_diuretic_demo) ────────────────────

_T5_DIMS = [f"D{i}" for i in range(1, 9)]


def run_t5(result_file: Path) -> None:
    """Invoke eval.runner_full for D1-D8 causal-inference scoring.

    Required environment variables:
      MIMIC_MAIN       path to the mimic-main project root
      MIMIC_DATA_ROOT  path to the MIMIC-IV demo data root
      OPENAI_API_KEY   for the LLM-based D1 field scorer
    Optional:
      T5_GOLDEN        override path to golden.json
      OPENAI_BASE_URL  use an OpenAI-compatible endpoint (e.g. OpenRouter)
    """
    import os
    import subprocess
    import tempfile

    mimic_main = os.environ.get("MIMIC_MAIN", "")
    mimic_data = os.environ.get("MIMIC_DATA_ROOT", "")
    if not mimic_main or not mimic_data:
        print(f"  {'causal_002':30s}  [SKIP] set MIMIC_MAIN and MIMIC_DATA_ROOT to enable T5")
        return

    mimic_main_p = Path(mimic_main)
    golden = os.environ.get("T5_GOLDEN") or str(
        mimic_main_p / "tasks/task_002_diuretic_demo/golden.json"
    )

    out_fd, out_path = tempfile.mkstemp(suffix=".json")
    os.close(out_fd)

    env = dict(os.environ)
    env["PYTHONPATH"] = mimic_main + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable, "-m", "eval.runner_full",
        "--answer",       str(result_file),
        "--golden",       golden,
        "--project-root", mimic_main,
        "--run-dir",      str(result_file.parent),
        "--data-root",    mimic_data,
        "--output",       out_path,
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=env, cwd=mimic_main
    )

    out_p = Path(out_path)
    if proc.returncode != 0 or not out_p.exists() or not out_p.stat().st_size:
        err = (proc.stderr or proc.stdout).strip()[:120]
        print(f"  {'causal_002':30s}  [RUNNER ERROR] {err}")
        out_p.unlink(missing_ok=True)
        return

    r = json.loads(out_p.read_text())
    out_p.unlink(missing_ok=True)

    dims  = r.get("dimensions", {})
    d_str = "  ".join(
        f"{d}={dims.get(d, {}).get('weighted_score', 0.0):.0f}" for d in _T5_DIMS
    )
    total = r.get("overall_score", 0.0)
    pct   = r.get("percentage",    0.0)
    passed = "PASS" if pct >= 50.0 else "FAIL"
    print(f"  {'causal_002':30s}  N=  1  {d_str}  Total={total:.0f}/200 ({pct:.1f}%)  [{passed}]")


# ── Entry point ──────────────────────────────────────────────────────────────

GROUPS = [
    ("stepwise_v2_20",     "stepwise"),   # T1
    ("stepwise_all_435",   "stepwise"),   # T2a
    ("stepwise_curated20", "stepwise"),   # T2b
    ("rare_disease_20",    "stepwise"),   # T3
    ("mimic_drug",         "drug"),       # T4
    ("causal_002",         "t5"),         # T5
]

def find_result(model_dir: Path, group: str) -> Path | None:
    """Find result file for a group in model_dir.

    T1-T4: searches <prefix>_{group}.jsonl or bare {group}.jsonl.
    T5 (causal_002): additionally searches *_causal_002.json, causal_002.json,
    and the bare answer.json produced by eval.runner_full.
    """
    matches = list(model_dir.glob(f"*_{group}.jsonl"))
    matches += list(model_dir.glob(f"{group}.jsonl"))
    matches += list(model_dir.glob(f"*_{group}.json"))
    matches += list(model_dir.glob(f"{group}.json"))
    if group == "causal_002":
        matches += list(model_dir.glob("answer.json"))
    return matches[0] if matches else None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval_all.py <model_dir>")
        sys.exit(1)
    model_dir = Path(sys.argv[1])
    print(f"\nModel dir: {model_dir}\n")

    for group, kind in GROUPS:
        f = find_result(model_dir, group)
        if f is None:
            print(f"  {group:30s}  [NOT FOUND]")
            continue
        if kind == "stepwise":
            run_stepwise(group, f)
        elif kind == "drug":
            run_drug(f)
        else:
            run_t5(f)
    print()
