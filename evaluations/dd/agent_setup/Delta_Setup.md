# Setting Up Delta For `evaluate_official_delta_agent4s.py`

## What This Script Actually Uses

`evaluate_official_delta_agent4s.py` does all of the following:

- loads one Agent4Science task JSON
- imports the official Delta `Agent` class from `src.agent`
- asks that agent to generate a Python solution
- saves the generated code
- runs the code through `runners.batch_runner`

Important: this is not the full official Delta pipeline. It uses the official Delta `Agent` class, but wraps it for Agent4Science task solving.

## Which Delta Repo It Expects

The script expects `--delta-dir` to point at a Delta repo root that contains:

- `src/`
- `pyproject.toml`
- `README.md`

In this workspace, you already have a local Delta checkout at [`delta/`](delta), and its Git remote points to the official repository:

- [deltawave-tech/delta](https://github.com/deltawave-tech/delta)

So for this repo, the default `--delta-dir delta` is already the right target.

## Prerequisites

- Python 3.11 or newer
- `uv`
- at least one model API key

Python 3.11 is the right baseline because Delta declares `requires-python = ">=3.11"` in [`delta/pyproject.toml`](delta/pyproject.toml:17).

## Install Delta Dependencies

Use the bundled Delta checkout:

```bash
cd delta
uv sync
source .venv/bin/activate
```

If `uv` is missing:

```bash
python3 -m pip install uv
```

## Configure API Keys

Delta's [`src/agent.py`](delta/src/agent.py:1) calls `load_dotenv()`, so a `.env` file in the Delta repo root is enough.

Create:

[`delta/.env`](delta/.env)

with:

```env
ANTHROPIC_API_KEY="your-anthropic-key"
OPENAI_API_KEY="your-openai-key"
```

You only need the provider key that matches the `--llm-provider` you plan to use.

## Sanity Check

Before running the benchmark wrapper, verify that the official Delta `Agent` imports correctly:

```bash
cd delta
source .venv/bin/activate
python -c "from src.agent import Agent; print(Agent.__name__)"
```

Expected:

```text
Agent
```

## Run `evaluate_official_delta_agent4s.py`

From the Agent4Science repo root:

```bash
python evaluate_official_delta_agent4s.py \
  tasks_batch/tech_04_hard_format.json \
  --llm-provider sonnet-4 \
  --delta-dir delta \
  --pretty
```

What happens next:

- the script imports `Agent` from Delta
- it asks Delta to generate one Python solver
- it saves that solver under `results/official_delta_agent4s/generated_code/`
- it runs the solver with `runners.batch_runner`
- it saves runner logs under `results/official_delta_agent4s/logs/`

## Output Locations

By default:

- generated code: [`results/official_delta_agent4s/generated_code/`](results/official_delta_agent4s/generated_code)
- runner logs: [`results/official_delta_agent4s/logs/`](results/official_delta_agent4s/logs)

## About `setup_delta_env.sh`

There is also a helper script at [`delta/setup_delta_env.sh`](delta/setup_delta_env.sh), but it contains a hard-coded Python path:

- `/opt/homebrew/Caskroom/miniconda/base/envs/agentd311/bin/python`

So the safer default is still:

```bash
cd delta
uv sync
source .venv/bin/activate
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'src.agent'`

Your `--delta-dir` is wrong. It must point to the Delta repo root, not to `delta/src` and not to the Agent4Science repo root.

### Authentication or provider errors

Make sure the API key for the selected `--llm-provider` is present in `delta/.env` or exported in the shell.

### Generated code is empty or invalid

That usually means the model response did not follow the bridge constraints closely enough. The wrapper already retries up to three times before giving up.

## References

- Setup target: [evaluate_official_delta_agent4s.py](evaluate_official_delta_agent4s.py)
- Local Delta checkout: [delta/](delta)
- Official Delta repo: [deltawave-tech/delta](https://github.com/deltawave-tech/delta)
