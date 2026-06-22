# Setting Up DrugAgent For `evaluate_official_drugagent_agent4s.py`

## What This Script Actually Uses

`evaluate_official_drugagent_agent4s.py` does all of the following:

- loads one Agent4Science task JSON
- imports the official DrugAgent LLM interface from the official repo
- asks that interface to generate one Python solution script
- saves the generated code and generation logs
- runs the generated script through `runners.batch_runner`

Important: this is not the full official DrugAgent runtime. It is an Agent4Science bridge around the official DrugAgent codebase.

## Which DrugAgent Repo It Expects

The script expects `--official-repo-dir` to point at the official DrugAgent checkout.

In this workspace, the default target is:

- [`external/drug-agent-official/`](external/drug-agent-official)

The bridge imports:

- `drugagent.LLM.complete_text`

from that repo, so the official repo directory must be importable from Python.

## Prerequisites

- Python 3.10 is the safest baseline
- a dedicated environment for DrugAgent
- the official DrugAgent repo cloned under `external/drug-agent-official`
- at least one model API key supported by the model you plan to use

The official repo README recommends:

```bash
conda create --name drugagent python=3.10
conda activate drugagent
pip install -r requirements.txt
```

## Install DrugAgent Dependencies

From the Agent4Science repo root:

```bash
conda create -n drugagent-a4s python=3.10 -y
conda activate drugagent-a4s
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r external/drug-agent-official/requirements.txt
```

If you also want the repo importable in editable mode:

```bash
python -m pip install -e external/drug-agent-official
```

That editable install is optional for the bridge, because the wrapper already inserts the repo root into `sys.path`.

## Configure API Keys

DrugAgent uses LiteLLM-style model IDs in the bridge wrapper, for example:

- `anthropic/claude-sonnet-4-6`
- `openai/gpt-4o-mini`

Export the matching provider key in your shell before running:

```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"
```

You only need the key for the provider that matches the `--model` you actually use.

## Sanity Check

Verify that the official DrugAgent LLM helper imports:

```bash
conda activate drugagent-a4s
cd <repo-root>
python -c "import sys; sys.path.insert(0, 'external/drug-agent-official'); from drugagent.LLM import complete_text; print(callable(complete_text))"
```

Expected:

```text
True
```

## Run `evaluate_official_drugagent_agent4s.py`

From the Agent4Science repo root:

```bash
conda activate drugagent-a4s
python evaluate_official_drugagent_agent4s.py \
  tasks_batch/tech_02_hard_indole.json \
  --model anthropic/claude-sonnet-4-6 \
  --official-repo-dir external/drug-agent-official \
  --pretty
```

What happens next:

- the script imports DrugAgent's official `complete_text`
- it asks DrugAgent to generate one solver script
- it saves that solver under `results/official_drugagent_agent4s/generated_code/`
- it saves LLM logs under `results/official_drugagent_agent4s/logs/`
- it runs the solver with `runners.batch_runner`

## Output Locations

By default:

- generated code: [`results/official_drugagent_agent4s/generated_code/`](results/official_drugagent_agent4s/generated_code)
- logs: [`results/official_drugagent_agent4s/logs/`](results/official_drugagent_agent4s/logs)

## Troubleshooting

### `Official DrugAgent repo not found`

Your `--official-repo-dir` is wrong. It must point to the DrugAgent repo root.

### Import errors inside `drugagent.*`

The environment is missing official repo dependencies. Re-run:

```bash
python -m pip install -r external/drug-agent-official/requirements.txt
```

### Model authentication errors

Make sure the API key for the selected model provider is exported in the shell.

### Heavy dependency issues

DrugAgent depends on both PyTorch and TensorFlow stack components. If installation gets messy, prefer a fresh dedicated conda environment rather than sharing one with other agents.

## References

- Bridge script: [evaluate_official_drugagent_agent4s.py](evaluate_official_drugagent_agent4s.py)
- Official local checkout: [external/drug-agent-official/](external/drug-agent-official)
- Official README: [external/drug-agent-official/README.md](external/drug-agent-official/README.md)
