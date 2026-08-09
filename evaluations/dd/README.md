# Agent4Science — Molecular & Protein Benchmark

A sandbox benchmark for evaluating autonomous AI agents on small-molecule
discovery tasks. **Agents write and execute Python code; the framework runs that
code in a controlled environment and grades the output.** It covers **79 tasks
across five scientific categories**, all driven by a single CLI
(`evaluate.py`), a single task registry, and one uniform result schema:

```
executability  →  validity  →  correctness  →  strategic_success
```

You can either **bring your own agent script** and score it, or **generate one
with a hosted LLM** (OpenAI / Anthropic / Google) and score it — in both cases
the evaluator picks the right execution mode automatically.

---

## Setup

### 1. Virtual environment (uv)

The framework targets **Python 3.11–3.13**. Using [uv](https://docs.astral.sh/uv/):

```bash
# from the evaluations/dd/ directory
uv venv .venv --python 3.12
source .venv/Scripts/activate      # Windows (Git Bash); use .venv/bin/activate on macOS/Linux

# core dependencies (scoring + the C1/C2/C4/C5 categories)
uv pip install rdkit pandas numpy matplotlib nbformat nbclient ipykernel scipy scikit-learn networkx
```

Register a Jupyter kernel for the notebook categories (C2 and C5). `--sys-prefix`
scopes it to this venv and leaves your global kernels untouched:

```bash
python -m ipykernel install --sys-prefix --name python3 --display-name "Python 3 (a4s)"
```

Optional extras, installed only when you need them:

```bash
uv pip install pytdc                       # the 10 C3 (des_*) design tasks
uv pip install openai anthropic google-genai   # only the provider(s) you use for `generate`
```

> A conda alternative is provided in `environment.yml`.

### 2. API keys

**Scoring needs no API keys** — `evaluate.py run` executes agent scripts offline
(it sets `AGENT4S_NO_NETWORK=1` and tasks forbid network imports).

Keys are needed only to **generate** agent solutions with a hosted LLM
(`evaluate.py generate`). Copy `.env.example` to `.env` and fill in the
provider(s) you'll use; `.env` is auto-loaded from this directory at startup
(`setdefault` semantics — it never overrides a variable already in your shell):

| Provider (`--provider`) | Env var | Default model |
|---|---|---|
| `openai` | `OPENAI_API_KEY` | `gpt-5.2` |
| `openai` (Azure) | `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_API_KEY` (+ `_API_VERSION`, `_DEPLOYMENT`) | deployment name |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-4-6` |
| `google` | `GOOGLE_API_KEY` | `gemini-3-pro-preview` |

Azure is selected automatically when `AZURE_OPENAI_ENDPOINT` is set (the
*deployment* name is then sent in place of the model id). Run
`python evaluate.py doctor` to see which provider SDKs are installed and which
keys are detected.

---

## The five categories

| | Category | What it tests | Tasks | Mode |
|---|---|---|---:|---|
| **C1** | Chemical Data Preprocessing | Compute molecular properties & process chemical data — MW, LogP, TPSA, formal charge, similarity, substructure filtering, format conversion, target ID | 18 | batch |
| **C2** | Chemical Data Analysis | Analyze and visualize SAR/assay data — produce figures or named result variables (incl. 3 vision tasks) | 26 | notebook (+ batch) |
| **C3** | Molecule Optimization | Write a molecular optimizer that designs analogs under a *budgeted oracle* — QED, penalized logP, DRD2, MPO, scaffold/deco hop, rediscovery | 10 | design |
| **C4** | Chemical Safety Assessment | Flag safety/ADMET liabilities — toxicophores, hERG, PAINS, metabolic soft spots, reactive groups | 6 | batch |
| **C5** | Chemical Claim Validation | Recognize when data is untrustworthy and respond with calibrated, *"inconclusive"* judgments rather than confident wrong answers | 19 | notebook |
| | | | **79** | |

Each task is a self-contained JSON (prompt + input + constraints + scoring
logic). The agent receives the task's `input` and must produce output in the
mode the task declares.

---

## How evaluation works

A task's `constraints.output_format` decides which **runner** executes it; its
`scoring_logic.scorer` decides which oracle grades it:

| `output_format` | Runner | Agent must… |
|---|---|---|
| `json_stdout` | `batch_runner` | read `AGENT4S_INPUT_JSON` (env var), print a JSON result to **stdout** |
| `matplotlib_figure` | `notebook_runner` | use the injected `_TASK_INPUT`, leave a finished figure in a variable named `fig` |
| `notebook_variables` | `notebook_runner` | use `_TASK_INPUT`, set the named result **variables** the task asks for |
| `oracle_stream` | `design_runner` | import `scorers.oracle_budget_*`, call `oracle.score(smiles)` up to the budget |

Working reference agents for each mode are in **`examples/`**:
`agent_mw.py` (C1), `agent_sar.py` (C2 figure), `agent_c5_logp.py` (C5
variables), `agent_c4_tox.py` (C4), `agent_c3_qed.py` (C3 design).

---

## Running evaluations

All commands run with the venv's Python (`python` once activated).

```bash
# Check environment, providers, runners, and that every scorer imports
python evaluate.py doctor

# List all 79 tasks grouped by category (filterable)
python evaluate.py list
python evaluate.py list --category C2
python evaluate.py list --mode batch

# Score an existing agent script
python evaluate.py run tech_01_hard_mw examples/agent_mw.py

# Persist records with --out, then aggregate mean metrics by category
python evaluate.py run ana_01_sar examples/agent_sar.py --out results/ana_01_sar.json
python evaluate.py aggregate "results/*.json"
```

A `run` produces a record like:

```json
{
  "task_id": "tech_01_hard_mw", "category": "C1", "mode": "batch",
  "runner": "batch_runner", "scorer": "oracle_chem.score_numeric_property",
  "score": { "executability": 1.0, "validity": 1.0, "correctness": 0.48, ... }
}
```

---

## Generating agent solutions with an LLM

`generate` is the built-in agent driver: it builds a mode-aware prompt from the
task, calls a hosted LLM to write a solution script, and (optionally) scores it.

```bash
# Inspect the prompts a task would produce — no API call, no key needed
python evaluate.py generate tech_01_hard_mw --provider openai --dry-run

# Generate a solution and immediately run + score it
python evaluate.py generate tech_01_hard_mw --provider anthropic --run
python evaluate.py generate ana_01_tsne     --provider google --model gemini-3-pro-preview --run
```

Generated scripts are written to `results/generated/<task_id>.py` (or `--out`).
The system instruction adapts to the task's mode (stdout JSON / `fig` /
variables / oracle stream), so the produced script satisfies the runner's
contract. Providers: `openai` (incl. Azure), `anthropic`, `google`.

**Framework agents.** The original benchmark also evaluated tool-using agent
frameworks (ChemToolAgent, ChemCrow, ToolUniverse, Cactus, Delta, DrugAgent,
LIDDiA, …). Per-agent install/config guides live in **[`agent_setup/`](agent_setup/)**
— see its index for the full list. Their `.env` knobs (`CLAUDE_CODE_CMD`,
`BIOMNI_*`) are in `.env.example`.

To plug in **your own** agent, write a script that meets the contract for the
task's mode (see *How evaluation works* and the matching file in `examples/`),
then `python evaluate.py run <task_id> path/to/agent.py`.

---

## Layout

```
evaluations/dd/
├── evaluate.py            # unified CLI: list / doctor / run / aggregate / generate
├── registry.py            # task discovery + routing (also: python registry.py --json)
├── tasks_index.json       # generated manifest of all 79 tasks
├── .env.example           # API-key template (copy to .env)
├── agents/                # LLM agent layer: providers (openai/azure/anthropic/google) + prompts + dotenv
├── agent_setup/           # per-agent setup guides (ChemToolAgent, ToolUniverse, Cactus, …)
├── runners/               # batch_runner · notebook_runner · design_runner
├── scorers/               # oracle_chem/vis/interactive/regulator/validity + C3 design scorers + data/ (ground truth + C5 answer pool)
├── tasks_batch/           # batch task JSONs in per-category subfolders + data/
│   ├── C1_chemical_data_preprocessing/   # tech_*.json
│   ├── C3_molecule_optimization/         # des_*.json
│   └── C4_chemical_safety_assessment/    # reg_*.json
├── tasks_interactive/     # notebook task JSONs in per-category subfolders + data/ (incl. vision/)
│   ├── C2_chemical_data_analysis/        # ana_*.json
│   └── C5_chemical_claim_validation/     # val_*.json  (answer key lives in scorers/data/)
├── examples/              # one reference agent per category
└── environment.yml        # conda env (reference)
```

---

## Notes

- **C3 needs PyTDC** (`uv pip install pytdc`); flagged `[needs PyTDC]` by `list`.
  The other four categories run on the core dependencies alone.
- **Provider SDKs are optional** — install only the one(s) you use for
  `generate`; the rest of the framework imports and runs without them.
- The reference agents in `examples/` exercise each path but are not tuned to
  maximize score (e.g. C4's alert library is intentionally small).
- All Python is formatted with `black` (line length 88) and `isort`
  (black profile) for a uniform style across the codebase.
- For the full per-task manifest (category, mode, runner, scorer per task), see
  `tasks_index.json` or run `python registry.py --json`.
```
