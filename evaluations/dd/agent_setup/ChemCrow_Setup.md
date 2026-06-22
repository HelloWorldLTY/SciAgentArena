# Setting Up ChemCrow For `evaluate_official_chemcrow_agent4s.py`

## What This Script Actually Uses

`evaluate_official_chemcrow_agent4s.py` does all of the following:

- loads one Agent4Science task JSON
- imports the official `ChemCrow` agent class from the local ChemCrow repo
- asks the agent to solve the task directly
- extracts the final JSON answer from ChemCrow's output
- scores the answer with the local Agent4Science scorer

Important: this is not the Hugging Face Space and not a reimplementation. It uses the official local `chemcrow-public` repo, wrapped for Agent4Science evaluation.

## Which ChemCrow Repo It Expects

The wrapper expects the official ChemCrow checkout at:

- [`external/chemcrow-public/`](external/chemcrow-public)

and imports:

- `chemcrow.agents.ChemCrow`

from that repo.

## Prerequisites

- a dedicated Python virtual environment
- the official ChemCrow repo cloned under `external/chemcrow-public`
- `OPENAI_API_KEY`

This wrapper currently uses OpenAI-backed ChemCrow models by default.

## Recommended Setup

This repo already includes a helper script:

- [setup_chemcrow_env.sh](setup_chemcrow_env.sh)

Run:

```bash
cd <repo-root>
zsh setup_chemcrow_env.sh
```

That script will:

- create `.venv-chemcrow`
- install the official ChemCrow repo in editable mode
- install the runtime dependencies used by the bridge
- leave you with a dedicated interpreter at:
  - [`.venv-chemcrow/bin/python`](.venv-chemcrow/bin/python)

## Important Local Note

This setup relies on the local offline `molbloom` stub shipped in this repo:

- [`molbloom.py`](molbloom.py)

That is part of making ChemCrow install cleanly in this workspace.

## Manual Setup

If you want to reproduce the setup manually:

```bash
cd <repo-root>
python -m venv .venv-chemcrow
./.venv-chemcrow/bin/pip install -U pip "setuptools<81" wheel
./.venv-chemcrow/bin/pip install --no-deps -e external/chemcrow-public
./.venv-chemcrow/bin/pip install \
  ipython \
  python-dotenv \
  rdkit \
  synspace \
  "openai==0.27.8" \
  "paper-qa==1.1.1" \
  google-search-results \
  "langchain>=0.0.234,<=0.0.275" \
  "langchain_core==0.0.1" \
  nest_asyncio \
  tiktoken \
  rmrkl \
  streamlit \
  rxn4chemistry \
  duckduckgo-search \
  wikipedia \
  paperscraper
```

## Configure API Keys

Export your OpenAI key before running the benchmark:

```bash
export OPENAI_API_KEY="your-openai-key"
```

The wrapper can also read `OPENAI_API_KEY` from:

- [`.env`](.env)

## Sanity Check

```bash
cd <repo-root>
./.venv-chemcrow/bin/python -c "from chemcrow.agents import ChemCrow; print(ChemCrow.__name__)"
```

Expected:

```text
ChemCrow
```

## Run `evaluate_official_chemcrow_agent4s.py`

Example:

```bash
cd <repo-root>
./.venv-chemcrow/bin/python evaluate_official_chemcrow_agent4s.py \
  tasks_batch/tech_01_hard_logP.json \
  --model gpt-4o-mini \
  --pretty
```

What happens next:

- the wrapper imports the official ChemCrow agent
- it prompts ChemCrow to solve the Agent4Science task
- it extracts the final JSON answer
- it scores the answer locally
- it stores artifacts under `results/official_chemcrow_agent4s/`

## Output Locations

By default:

- [`results/official_chemcrow_agent4s/`](results/official_chemcrow_agent4s)

This includes agent output, raw stdout, stderr, and score JSON.

## Troubleshooting

### `OPENAI_API_KEY is not set`

Export the key in the same shell, or add it to the repo `.env`.

### `ModuleNotFoundError: No module named 'chemcrow'`

Use the interpreter inside `.venv-chemcrow`, not your base environment.

### ChemCrow installs but certain tasks still fail

That is often an agent behavior issue, not an installation issue. For example, some Agent4Science tasks surface ChemCrow refusal patterns, file-following issues, or missing optional parser dependencies in the benchmark environment.

## References

- Bridge script: [evaluate_official_chemcrow_agent4s.py](evaluate_official_chemcrow_agent4s.py)
- Setup helper: [setup_chemcrow_env.sh](setup_chemcrow_env.sh)
- Official local checkout: [external/chemcrow-public/](external/chemcrow-public)
- Official README: [external/chemcrow-public/README.md](external/chemcrow-public/README.md)
