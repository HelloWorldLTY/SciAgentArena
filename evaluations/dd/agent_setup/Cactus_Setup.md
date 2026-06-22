# Setting Up CACTUS For `evaluate_cactus_agent.py`

## What This Script Actually Uses

`evaluate_cactus_agent.py` does all of the following:

- loads one or more Agent4Science task JSON files
- imports the official `Cactus` agent class from the CACTUS repo
- asks that agent to answer the task directly
- extracts JSON from the answer
- scores the answer with the local Agent4Science scorer

Important: this is not a code-generation bridge. It is a direct adapter that runs the CACTUS agent itself against Agent4Science tasks.

## Which CACTUS Repo It Expects

The script imports:

- `cactus.agent.Cactus`

from:

- [`external/cactus/`](external/cactus)

So the local checkout must exist and be installed in the active environment.

## Prerequisites

- Python 3.10 to 3.12
- conda
- the official CACTUS repo cloned under `external/cactus`
- at least one provider API key that matches the model you plan to use

## Recommended Setup

This repo already includes a helper script:

- [setup_cactus_env.sh](setup_cactus_env.sh)

Run:

```bash
cd <repo-root>
bash setup_cactus_env.sh cactus-a4s 3.10
```

That script will:

- create a conda env
- clone CACTUS if needed
- install pinned LangChain versions compatible with CACTUS
- install the local CACTUS checkout in editable mode
- run a smoke import

## Manual Setup

If you want to set it up manually:

```bash
conda create -n cactus-a4s python=3.10 -y
conda activate cactus-a4s
python -m pip install --upgrade pip setuptools wheel
python -m pip install "numpy<2" pandas python-dotenv
python -m pip install \
  "langchain==0.3.4" \
  "langchain-community==0.3.3" \
  "langchain-core==0.3.12" \
  "langchain-anthropic==0.2.3" \
  "langchain-openai==0.2.3" \
  "langchain-google-genai==2.0.1"
python -m pip install -e external/cactus
```

Those LangChain pins matter. The wrapper explicitly warns that newer LangChain APIs can break CACTUS imports.

## Configure API Keys

Depending on `--model-type` and `--model-name`, export the right key:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
```

You only need the key for the provider you actually use.

## Sanity Check

```bash
conda activate cactus-a4s
cd <repo-root>
python -c "from cactus.agent import Cactus; print(Cactus.__name__)"
```

Expected:

```text
Cactus
```

## Run `evaluate_cactus_agent.py`

Example:

```bash
conda activate cactus-a4s
python evaluate_cactus_agent.py \
  tasks_batch/tech_01_hard_logP.json \
  --model-name claude-3-haiku-20240307 \
  --model-type api \
  --pretty
```

You can also run the formal-charge subset quickly:

```bash
python evaluate_cactus_agent.py \
  --formal-charge-only \
  --model-name claude-3-haiku-20240307 \
  --model-type api \
  --pretty
```

## Troubleshooting

### `Cactus/LangChain API mismatch`

Your LangChain versions are too new or otherwise incompatible. Reinstall the pinned versions from the setup script.

### Import errors from `cactus.*`

Make sure the editable install succeeded and that `external/cactus` exists.

### API failures

Confirm the correct provider key is exported for the selected model.

## References

- Adapter script: [evaluate_cactus_agent.py](evaluate_cactus_agent.py)
- Setup helper: [setup_cactus_env.sh](setup_cactus_env.sh)
- Official local checkout: [external/cactus/](external/cactus)
- Official README: [external/cactus/README.md](external/cactus/README.md)
