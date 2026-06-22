# Setting Up Biomni (A1 agent)

## Is This Using the Official Repo?

Yes. This drives Biomni's official `A1` agent **inline** (the project venv has
the `biomni` package; the benchmark used 0.0.8). A thin `BiomniDriver` wraps
A1: it configures the backbone LLM, runs the agent with the data lake and tool
retriever disabled, and hands the produced script to the scorer.

## What the Driver Actually Does

- imports `from biomni.agent import A1` and `from biomni.config import default_config`
- configures the backbone LLM, source, temperature, timeout, and a local
  working `path`
- runs A1 with the **data lake and tool retriever disabled**, so it behaves as a
  general coding agent rather than pulling Biomni's biomedical data lake:
  - `A1(expected_data_lake_files=[])`, `agent.use_tool_retriever = False`
  - clears `module2api`, `data_lake_dict`, `library_content_dict`, know-how docs
- wraps the task prompt asking A1 to return the complete solution inside a single
  `<solution>...</solution>` block
- extracts that block, writes `agent_solution.py`; the harness then scores it via
  `runners.batch_runner`

## Prerequisites

- The **biomni** package installed in the project venv (`uv pip install biomni`;
  the benchmark used 0.0.8).
- An API key for the backbone LLM (default source Anthropic →
  `ANTHROPIC_API_KEY`).
- rdkit / numpy / scipy / networkx / pandas / scikit-learn / tdc in the same
  venv — the generated script needs them at scoring time.

## Configuration (env / knobs)

- `BIOMNI_LLM` — backbone model (default `claude-sonnet-4-20250514`)
- `BIOMNI_SOURCE` — provider (default `Anthropic`)
- `BIOMNI_TEMPERATURE` — sampling temperature (default `0`)
- `BIOMNI_PATH` — local Biomni working dir (default `<sandbox>/.biomni_local`)
- Driver also accepts `model`, `timeout_seconds` (default `1800`), and an
  optional `biomni_env_file` (a `.env` loaded with `setdefault`).

## Notes

- The data lake / tool retriever are disabled on purpose so Biomni acts as a
  code-writing agent on the Agent4Science task, not a bio-retrieval agent.
- A1 must return its solution in a `<solution>...</solution>` block (falls back
  to ```` ```python ```` / ```` ``` ```` fences); otherwise the run is recorded
  as `no_solution_script`.
- The driver itself is not bundled here; this guide covers the setup needed to
  run Biomni as the agent.
