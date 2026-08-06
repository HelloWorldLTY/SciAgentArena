#!/usr/bin/env python3
"""
TxAgent Benchmark Runner — all task0502.csv tasks
Model: mims-harvard/TxAgent-T1-Llama-3.1-8B

Handles all 6 task groups from task0502.csv:
  - stepwise_curated20  (20 instances, 3 difficulties)
  - stepwise_0404       (20 instances, 3 difficulties)
  - stepwise_all_435    (435 instances, 3 difficulties)
  - rare_disease_20     (20 instances, 3 difficulties)
  - stepwise_v2_20      (20 instances, 3 difficulties)
  - mimic_drug          (10 instances, complex)

Usage:
  python run_txagent_benchmark.py [--tasks all] [--max-new-tokens 1024]

Output: results saved to same directory as this script.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent.resolve()
BENCH_DIR  = SCRIPT_DIR.parent.parent / "MedAgentBench"  # …/medagentbench/MedAgentBench

TASK_FILES = {
    "stepwise_curated20": BENCH_DIR / "synthea_stepwise_benchmark/output/medagentbench_stepwise_curated20_tasks.json",
    "stepwise_0404":      BENCH_DIR / "synthea_stepwise_benchmark/output/medagentbench_0404_instantiated_tasks.json",
    "stepwise_all_435":   BENCH_DIR / "synthea_stepwise_benchmark/output/medagentbench_stepwise_tasks.json",
    "rare_disease_20":    BENCH_DIR / "synthea_stepwise_benchmark/output_rare_disease/medagentbench_rare_disease_tasks.json",
    "stepwise_v2_20":     BENCH_DIR / "synthea_stepwise_benchmark/output_v2/medagentbench_stepwise_v2_tasks.json",
    "mimic_drug":         BENCH_DIR / "mimic_drug_recommendation_tasks.json",
}

GT_FILES = {
    "stepwise_curated20": BENCH_DIR / "synthea_stepwise_benchmark/output/medagentbench_stepwise_curated20_groundtruth.json",
    "stepwise_0404":      None,   # GT embedded in task JSON
    "stepwise_all_435":   BENCH_DIR / "synthea_stepwise_benchmark/output/groundtruth.json",
    "rare_disease_20":    None,   # GT embedded in task JSON
    "stepwise_v2_20":     BENCH_DIR / "synthea_stepwise_benchmark/output_v2/medagentbench_stepwise_v2_groundtruth.json",
    "mimic_drug":         None,   # GT embedded in task JSON
}

MODEL_NAME     = "mims-harvard/TxAgent-T1-Llama-3.1-8B"
RAG_MODEL_NAME = "mims-harvard/ToolRAG-T1-GTE-Qwen2-1.5B"  # used if TxAgent class available

# ──────────────────────────────────────────────────────────────────────────────
# Model loading (prefer native TxAgent, fallback to HF transformers pipeline)
# ──────────────────────────────────────────────────────────────────────────────

_model = None
_tokenizer = None
_use_txagent_class = False
_txagent_instance = None


def load_model():
    global _model, _tokenizer, _use_txagent_class, _txagent_instance

    # Try native TxAgent first (needs vllm)
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent /
                               "EHR_bench/agent_deployments/TxAgent/src"))
        from txagent import TxAgent
        print(f"[INFO] Loading TxAgent class with model {MODEL_NAME} ...")
        agent = TxAgent(MODEL_NAME, RAG_MODEL_NAME, enable_summary=False)
        agent.init_model()
        _txagent_instance = agent
        _use_txagent_class = True
        print("[INFO] TxAgent class loaded successfully.")
        return
    except Exception as e:
        print(f"[WARN] TxAgent class unavailable ({e}), falling back to HF transformers.")

    # Fallback: HF transformers
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[INFO] torch version: {torch.__version__}")
    print(f"[INFO] torch.version.cuda: {torch.version.cuda}")
    print(f"[INFO] torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"[INFO] CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        target_device = {"" : "cuda:0"}
    else:
        print("[WARN] CUDA not available, using CPU (will be slow!)")
        target_device = "cpu"

    print(f"[INFO] Loading {MODEL_NAME} via transformers ...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=target_device,
    )
    _model.eval()
    dev = next(_model.parameters()).device
    print(f"[INFO] Model loaded on device: {dev}")
    if str(dev) == "cpu":
        print("[ERROR] Model is on CPU — inference will be extremely slow!")


def generate_response(prompt: str, max_new_tokens: int = 1024) -> str:
    """Generate a response from the loaded model."""
    if _use_txagent_class and _txagent_instance is not None:
        try:
            return _txagent_instance.run_multistep_agent(
                prompt,
                temperature=0.0,
                max_new_tokens=max_new_tokens,
                max_token=8192,
                max_round=15,
                call_agent=False,
            )
        except Exception as e:
            print(f"[WARN] TxAgent run_multistep_agent failed: {e}")
            return ""

    # HF pipeline path
    import torch
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful medical AI assistant with expertise in clinical pharmacology "
                "and EHR-based clinical workflows. You reason step by step and follow instructions precisely."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(text, return_tensors="pt").to(_model.device)
    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=_tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs.input_ids.shape[1]:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ──────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ──────────────────────────────────────────────────────────────────────────────

STEPWISE_SYSTEM = (
    "You are an EHR clinical workflow assistant. Given the clinical scenario below, "
    "produce a concise step-by-step action plan as strict JSON.\n\n"
    "Output format:\n"
    "{\n"
    '  "steps": [\n'
    '    {"step": 1, "action": "fhir_verb_resource", "detail": "..."}\n'
    "  ],\n"
    '  "final_answer": "..."\n'
    "}\n\n"
    "Rules:\n"
    "- Output JSON only, no markdown and no extra text.\n"
    "- Each action should be a concrete FHIR operation such as fhir_search_observation, "
    "fhir_create_medication_request, fhir_create_service_request, fhir_stop_medication, "
    "fhir_search_procedure, fhir_create_condition, verify_marker_condition, branch_decision, "
    "calculator, etc.\n"
    "- Use English only.\n"
    "- Include only clinically necessary steps.\n\n"
)

DRUG_SYSTEM = (
    "You are an expert clinical pharmacist and physician. "
    "Given patient information, diagnoses, and lab results, "
    "provide a numbered list of recommended medications. "
    "For each drug, give its name and a brief one-line justification. "
    "Be specific with drug names. List the most important/primary drugs first."
)


def build_stepwise_prompt(case: Dict[str, Any]) -> str:
    task_id   = case.get("id") or case.get("task_id", "")
    difficulty= case.get("difficulty", "")
    context   = case.get("context", "")
    instruction = case.get("instruction", "")
    return (
        STEPWISE_SYSTEM
        + f"Task ID: {task_id}\n"
        + f"Difficulty: {difficulty}\n"
        + f"Context: {context}\n"
        + f"Instruction: {instruction}"
    )


def build_drug_prompt(case: Dict[str, Any]) -> str:
    return (
        DRUG_SYSTEM + "\n\n"
        + f"Question: {case.get('question', '')}\n\n"
        + f"Patient Information:\n{case.get('patient_info', '')}\n\n"
        + f"Diagnoses:\n{case.get('diagnoses', '')}\n\n"
        + f"Key Labs:\n{case.get('key_labs', '')}\n"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

_JSON_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]*?)```")


def _parse_model_json(text: str) -> Any:
    text = text.strip()
    m = _JSON_BLOCK.search(text)
    if m:
        text = m.group(1).strip()
    # try full text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # try first {...} block
    m2 = re.search(r"\{[\s\S]*\}", text)
    if m2:
        try:
            return json.loads(m2.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def _extract_steps(parsed: Any) -> List[Dict[str, Any]]:
    if isinstance(parsed, dict) and isinstance(parsed.get("steps"), list):
        return [x for x in parsed["steps"] if isinstance(x, dict)]
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    return []


def score_stepwise(expected_steps: List[Dict], model_text: str) -> Dict[str, float]:
    """Compute precision/recall/F1 on FHIR action names."""
    expected_actions = [str(s.get("action", "")).lower() for s in expected_steps if s.get("action")]

    parsed = _parse_model_json(model_text)
    steps  = _extract_steps(parsed)
    produced_actions = [str(s.get("action", "")).lower() for s in steps]

    exp_count = len(expected_actions)
    pred_count = len(produced_actions)

    if exp_count == 0:
        recall = 1.0
    else:
        hit = sum(1 for a in expected_actions if any(a in c or c in a for c in produced_actions))
        recall = hit / exp_count

    if pred_count == 0:
        precision = 1.0 if exp_count == 0 else 0.0
    else:
        matched = sum(1 for c in produced_actions if any(a in c or c in a for a in expected_actions))
        precision = matched / pred_count

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "expected_count": exp_count,
        "produced_count": pred_count,
    }


def _normalize_drug(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _drug_match(gt: str, pred: str) -> bool:
    ngt = _normalize_drug(gt)
    npred = _normalize_drug(pred)
    if not ngt or not npred:
        return False
    if ngt == npred or ngt in npred or npred in ngt:
        return True
    return SequenceMatcher(None, ngt, npred).ratio() >= 0.82


def _extract_gt_drug_names(gt_medications: List[str]) -> List[str]:
    """Extract just the drug name from 'DrugName — justification' format."""
    names = []
    for entry in gt_medications:
        if isinstance(entry, str):
            name = re.split(r"\s+[—–-]\s+", entry, maxsplit=1)[0].strip()
            names.append(name)
    return names


def _extract_pred_drugs(text: str) -> List[str]:
    """Extract drug names from numbered list in model output."""
    if not text:
        return []
    text = re.sub(r"<thinking>[\s\S]*?</thinking>", "", text, flags=re.I)

    numbered = re.findall(r"(?m)^\s*\d{1,2}[\.)]\s*(.+)", text)
    if numbered:
        candidates = [x.strip() for x in numbered if x.strip()]
    else:
        bullets = re.findall(r"(?m)^[-*•]\s*(.+)", text)
        candidates = [x.strip() for x in bullets] if bullets else [
            ln.strip() for ln in text.split("\n") if ln.strip()
        ]

    drugs: List[str] = []
    for c in candidates:
        c = re.sub(r"\*\*(.*?)\*\*", r"\1", c)
        c = re.split(r"\s+[—–-]\s+", c, maxsplit=1)[0]
        c = re.split(r"\s*:\s+", c, maxsplit=1)[0]
        c = re.sub(r"^\d+[\.)]\s*", "", c).strip(" \t:-.")
        if not c or len(c) > 100:
            continue
        low = c.lower()
        if any(low.startswith(kw) for kw in [
            "medication", "drug", "prescription", "recommendation",
            "based on", "the patient", "given the", "note:", "here",
            "primary", "key", "these", "following",
        ]):
            continue
        if re.search(r"\b(should|would|could|patient|history|given)\b", low):
            continue
        drugs.append(c)
    return drugs


def score_drug(gt_medications: List[str], model_text: str) -> Dict[str, float]:
    gt_names   = _extract_gt_drug_names(gt_medications)
    pred_names = _extract_pred_drugs(model_text)

    if not gt_names:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "gt_count": 0, "pred_count": len(pred_names)}

    tp = sum(1 for g in gt_names if any(_drug_match(g, p) for p in pred_names))
    recall    = tp / len(gt_names)
    precision = (tp / len(pred_names)) if pred_names else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "gt_count":  len(gt_names),
        "pred_count": len(pred_names),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Data loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_stepwise_curated20() -> List[Dict]:
    tasks = json.loads(TASK_FILES["stepwise_curated20"].read_text())
    gt_list = json.loads(GT_FILES["stepwise_curated20"].read_text())
    gt_map = {g["task_id"]: g["expected_steps"] for g in gt_list}
    for t in tasks:
        t["_expected_steps"] = gt_map.get(t["id"], [])
    return tasks


def load_stepwise_0404() -> List[Dict]:
    tasks = json.loads(TASK_FILES["stepwise_0404"].read_text())
    for t in tasks:
        es = t.get("expected_steps", [])
        # Some 0404 tasks store expected_steps as an int (just a count) — treat as no GT
        t["_expected_steps"] = es if isinstance(es, list) else []
    return tasks


def load_stepwise_all_435() -> List[Dict]:
    tasks = json.loads(TASK_FILES["stepwise_all_435"].read_text())
    gt_list = json.loads(GT_FILES["stepwise_all_435"].read_text())
    gt_map = {g["task_id"]: g["expected_steps"] for g in gt_list}
    for t in tasks:
        t["_expected_steps"] = gt_map.get(t["id"], [])
    return tasks


def load_rare_disease_20() -> List[Dict]:
    tasks = json.loads(TASK_FILES["rare_disease_20"].read_text())
    for t in tasks:
        t["_expected_steps"] = t.get("expected_steps", [])
    return tasks


def load_stepwise_v2_20() -> List[Dict]:
    tasks = json.loads(TASK_FILES["stepwise_v2_20"].read_text())
    gt_list = json.loads(GT_FILES["stepwise_v2_20"].read_text())
    gt_map = {g["task_id"]: g["expected_steps"] for g in gt_list}
    # v2 task keys: task_index, task_id, difficulty, scenario, expected_steps
    for t in tasks:
        tid = t.get("task_id", t.get("id", ""))
        t["_expected_steps"] = gt_map.get(tid, t.get("expected_steps", []))
        # Normalize field names for prompt builder
        if "instruction" not in t:
            t["instruction"] = t.get("scenario", str(t))
        if "id" not in t:
            t["id"] = tid
    return tasks


def load_mimic_drug() -> List[Dict]:
    tasks = json.loads(TASK_FILES["mimic_drug"].read_text())
    for t in tasks:
        t["_gt_medications"] = t.get("ground_truth_medications", [])
    return tasks


# ──────────────────────────────────────────────────────────────────────────────
# Task runners
# ──────────────────────────────────────────────────────────────────────────────

def run_stepwise_group(
    name: str,
    tasks: List[Dict],
    max_new_tokens: int,
    out_dir: Path,
) -> Dict:
    out_path = out_dir / f"{name}.jsonl"
    print(f"\n{'='*60}")
    print(f"[TASK GROUP] {name}  ({len(tasks)} instances)")
    print(f"{'='*60}")

    rows = []
    scores_by_diff: Dict[str, List[float]] = {}

    for i, case in enumerate(tasks):
        task_id   = case.get("id") or case.get("task_id", f"task_{i}")
        difficulty= case.get("difficulty", "unknown")
        prompt    = build_stepwise_prompt(case)
        expected  = case.get("_expected_steps", [])

        print(f"  [{i+1}/{len(tasks)}] {task_id} ({difficulty}) ...", end=" ", flush=True)
        t0 = time.time()
        try:
            response = generate_response(prompt, max_new_tokens=max_new_tokens)
        except Exception as e:
            print(f"ERROR: {e}")
            response = ""
        elapsed = time.time() - t0

        metrics = score_stepwise(expected, response)
        print(f"F1={metrics['f1']:.3f}  ({elapsed:.1f}s)")

        row = {
            "task_id":    task_id,
            "difficulty": difficulty,
            "instruction": case.get("instruction", ""),
            "response":   response,
            "score":      metrics,
        }
        rows.append(row)

        scores_by_diff.setdefault(difficulty, []).append(metrics["f1"])

    # Write JSONL
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    # Aggregate scores
    summary = {
        "task_group":    name,
        "n_instances":   len(rows),
        "overall_f1":    round(sum(r["score"]["f1"] for r in rows) / len(rows), 4) if rows else 0,
        "overall_precision": round(sum(r["score"]["precision"] for r in rows) / len(rows), 4) if rows else 0,
        "overall_recall":    round(sum(r["score"]["recall"] for r in rows) / len(rows), 4) if rows else 0,
        "by_difficulty": {
            diff: {
                "n": len(f1s),
                "f1": round(sum(f1s) / len(f1s), 4),
            }
            for diff, f1s in scores_by_diff.items()
        },
    }
    scores_path = out_dir / f"{name}_scores.json"
    scores_path.write_text(json.dumps(summary, indent=2))

    print(f"\n  → Saved {len(rows)} rows to {out_path}")
    print(f"  → Overall F1: {summary['overall_f1']}")
    return summary


def run_mimic_drug(
    tasks: List[Dict],
    max_new_tokens: int,
    out_dir: Path,
) -> Dict:
    out_path = out_dir / "mimic_drug.jsonl"
    print(f"\n{'='*60}")
    print(f"[TASK GROUP] mimic_drug  ({len(tasks)} instances)")
    print(f"{'='*60}")

    rows = []
    f1_scores = []

    for i, case in enumerate(tasks):
        task_id = case.get("task_id", f"MIMIC_DRUG_{i+1:03d}")
        prompt  = build_drug_prompt(case)
        gt_meds = case.get("_gt_medications", [])

        print(f"  [{i+1}/{len(tasks)}] {task_id} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            response = generate_response(prompt, max_new_tokens=max_new_tokens)
        except Exception as e:
            print(f"ERROR: {e}")
            response = ""
        elapsed = time.time() - t0

        metrics = score_drug(gt_meds, response)
        f1_scores.append(metrics["f1"])
        print(f"F1={metrics['f1']:.3f}  ({elapsed:.1f}s)")

        rows.append({
            "task_id":  task_id,
            "question": case.get("question", ""),
            "response": response,
            "score":    metrics,
            "gt_medications": gt_meds,
        })

    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    summary = {
        "task_group":        "mimic_drug",
        "n_instances":       len(rows),
        "overall_f1":        round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0,
        "overall_precision": round(sum(r["score"]["precision"] for r in rows) / len(rows), 4) if rows else 0,
        "overall_recall":    round(sum(r["score"]["recall"] for r in rows) / len(rows), 4) if rows else 0,
    }
    scores_path = out_dir / "mimic_drug_scores.json"
    scores_path.write_text(json.dumps(summary, indent=2))

    print(f"\n  → Saved {len(rows)} rows to {out_path}")
    print(f"  → Overall F1: {summary['overall_f1']}")
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

ALL_TASKS = [
    "stepwise_curated20",
    "stepwise_0404",
    "stepwise_all_435",
    "rare_disease_20",
    "stepwise_v2_20",
    "mimic_drug",
]


def main():
    parser = argparse.ArgumentParser(description="TxAgent Benchmark Runner")
    parser.add_argument(
        "--tasks", nargs="+", default=["all"],
        choices=["all"] + ALL_TASKS,
        help="Task groups to run (default: all)"
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--out-dir", type=str,
        default=str(SCRIPT_DIR),
        help="Output directory (default: same dir as this script)"
    )
    args = parser.parse_args()

    tasks_to_run = ALL_TASKS if "all" in args.tasks else args.tasks
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Tasks to run: {tasks_to_run}")
    print(f"[INFO] Output dir: {out_dir}")
    print(f"[INFO] Max new tokens: {args.max_new_tokens}")

    # Load model once
    load_model()

    summaries = {}

    for task_name in tasks_to_run:
        try:
            if task_name == "stepwise_curated20":
                tasks = load_stepwise_curated20()
                s = run_stepwise_group(task_name, tasks, args.max_new_tokens, out_dir)
            elif task_name == "stepwise_0404":
                tasks = load_stepwise_0404()
                s = run_stepwise_group(task_name, tasks, args.max_new_tokens, out_dir)
            elif task_name == "stepwise_all_435":
                tasks = load_stepwise_all_435()
                s = run_stepwise_group(task_name, tasks, args.max_new_tokens, out_dir)
            elif task_name == "rare_disease_20":
                tasks = load_rare_disease_20()
                s = run_stepwise_group(task_name, tasks, args.max_new_tokens, out_dir)
            elif task_name == "stepwise_v2_20":
                tasks = load_stepwise_v2_20()
                s = run_stepwise_group(task_name, tasks, args.max_new_tokens, out_dir)
            elif task_name == "mimic_drug":
                tasks = load_mimic_drug()
                s = run_mimic_drug(tasks, args.max_new_tokens, out_dir)
            else:
                print(f"[WARN] Unknown task: {task_name}")
                continue
            summaries[task_name] = s
        except Exception as e:
            print(f"[ERROR] {task_name} failed: {e}")
            traceback.print_exc()
            summaries[task_name] = {"error": str(e)}

    # Write final summary
    summary_path = out_dir / "results_summary.json"
    final = {
        "model":     MODEL_NAME,
        "tasks_run": tasks_to_run,
        "results":   summaries,
    }
    summary_path.write_text(json.dumps(final, indent=2))
    print(f"\n{'='*60}")
    print(f"[DONE] Results summary written to {summary_path}")
    print(f"{'='*60}")
    for task, s in summaries.items():
        if "error" in s:
            print(f"  {task}: ERROR — {s['error']}")
        else:
            print(f"  {task}: F1={s.get('overall_f1', 'N/A')}")


if __name__ == "__main__":
    main()
