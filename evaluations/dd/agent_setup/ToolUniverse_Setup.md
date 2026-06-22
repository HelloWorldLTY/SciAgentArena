# Setting Up ToolUniverse For `evaluate_tooluniverse_agent4s.py`

## What This Script Actually Uses

`evaluate_tooluniverse_agent4s.py` does all of the following:

- loads one or more Agent4Science task JSON files
- initializes a local ToolUniverse workspace
- loads a curated set of Agent4Science-specific tool wrappers from:
  - [`tooluniverse_agent4s_tools.py`](tooluniverse_agent4s_tools.py)
- uses an LLM planner from OpenAI, Anthropic, or Gemini
- lets the planner call ToolUniverse tools
- scores the final answer with the local Agent4Science scorer

Important: this is not an official external agent repo. It is an Agent4Science-native tool-using benchmark harness built around the ToolUniverse package.

## Which Environment It Expects

Unlike the official-agent bridges, ToolUniverse does not require a separate external repo checkout.

It expects:

- the main Agent4Science repo
- the `tooluniverse` Python package
- one planner provider key

The script automatically creates and uses:

- [`.tooluniverse_home/`](.tooluniverse_home)
- [`.tooluniverse_cache/`](.tooluniverse_cache)

## Prerequisites

- Python 3.11 is a good default
- the main Agent4Science environment
- `tooluniverse`
- at least one provider key:
  - `OPENAI_API_KEY`
  - or `GOOGLE_API_KEY`
  - or `ANTHROPIC_API_KEY`

## Recommended Setup

If you are already using the main Agent4Science environment defined by:

- [environment.yml](environment.yml)

then install the extra runtime packages:

```bash
conda env create -f environment.yml
conda activate a4s-molprot
python -m pip install --upgrade pip setuptools wheel
python -m pip install tooluniverse openai anthropic google-genai python-dotenv
```

If you already have the environment, only the last line is usually needed.

## Configure API Keys

Pick one planner provider.

For OpenAI:

```bash
export OPENAI_API_KEY="your-openai-key"
```

For Gemini:

```bash
export GOOGLE_API_KEY="your-google-key"
```

For Anthropic:

```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
```

The script auto-selects a default provider in this order:

- OpenAI if `OPENAI_API_KEY` is present
- otherwise Gemini if `GOOGLE_API_KEY` is present
- otherwise Anthropic

## Sanity Check

```bash
conda activate a4s-molprot
cd <repo-root>
python -c "from tooluniverse import ToolUniverse; print(ToolUniverse.__name__)"
```

Expected:

```text
ToolUniverse
```

## Run `evaluate_tooluniverse_agent4s.py`

OpenAI example:

```bash
conda activate a4s-molprot
python evaluate_tooluniverse_agent4s.py \
  "tasks_batch/tech_02_hard_indole.json" \
  --provider openai \
  --model gpt-4o-mini \
  --summary
```

Gemini example:

```bash
python evaluate_tooluniverse_agent4s.py \
  "tasks_batch/tech_02_hard_indole.json" \
  --provider gemini \
  --model gemini-3-flash-preview \
  --summary
```

Anthropic example:

```bash
python evaluate_tooluniverse_agent4s.py \
  "tasks_batch/tech_02_hard_indole.json" \
  --provider anthropic \
  --model claude-sonnet-4-20250514 \
  --summary
```

## Output Locations

By default:

- [`results/tooluniverse_agent4s/`](results/tooluniverse_agent4s)

Each run stores task output, score JSON, and planner/tool artifacts there.

## Troubleshooting

### `OPENAI_API_KEY is not set` or similar

Export the API key for the planner provider you selected.

### `No files matched`

Your task path or glob pattern is wrong. The first positional argument is a path or glob, not a task ID.

### ToolUniverse import failures

Install or reinstall:

```bash
python -m pip install tooluniverse
```

### Planner output seems correct but score is low

This often means the tool planner returned normalized values that do not match the benchmark's strict representation requirements, especially on exact-string tasks.

## References

- Harness script: [evaluate_tooluniverse_agent4s.py](evaluate_tooluniverse_agent4s.py)
- Tool definitions: [tooluniverse_agent4s_tools.py](tooluniverse_agent4s_tools.py)
- Main environment: [environment.yml](environment.yml)
