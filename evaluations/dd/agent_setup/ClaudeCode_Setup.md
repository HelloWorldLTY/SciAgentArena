# Setting Up Claude Code as a Coding Agent

## Is This Using the Official Tool?

Yes. This drives the real `claude` CLI (Claude Code) **headlessly** as a
code-generation agent — it is not a reimplementation. A thin `ClaudeCodeDriver` wraps the CLI:
it builds the task prompt, runs `claude` headlessly, and hands the produced
script to the scorer.

## What the Driver Actually Does

- loads an Agent4Science task and builds the task prompt
- invokes `claude` headlessly in a per-task **sandbox** directory, roughly:

  ```
  claude -p <prompt> --output-format stream-json --bare --effort low \
         --dangerously-skip-permissions --verbose --include-hook-events \
         --tools Bash Edit Read Write Glob Grep LS \
         --allowedTools "Bash(python3:*)" Read Write Edit Glob Grep LS \
         --disallowedTools WebFetch WebSearch
  ```

- appends a system prompt telling Claude its **one deliverable** is to `Write`
  `agent_solution.py` (and explicitly NOT to run the oracle)
- captures the full stream-json NDJSON trace (assistant / tool_use / result
  events, `total_cost_usd`, `usage`, turn counts)
- the shared harness then executes that script via `runners.batch_runner`

So Claude Code writes the solution with Edit/Write and may use Bash `python3`
for quick local checks; the framework scores it afterward.

## Prerequisites

- The **Claude Code CLI** installed and on `PATH` (`claude --version`). Install
  per Anthropic's docs (npm `@anthropic-ai/claude-code`, or the native
  installer).
- Authentication for Claude Code: an Anthropic login or `ANTHROPIC_API_KEY` in
  the environment (Claude Code's own auth).
- The project venv (rdkit, numpy, scipy, networkx, pandas, scikit-learn, tdc) —
  the driver prepends `<venv>/bin` to `PATH` so any `python3` Claude runs uses
  that interpreter, and so the produced script can be scored.

## Configuration (env / knobs)

- `CLAUDE_CODE_CMD` — the CLI command (default `claude`); set to a full path or
  a wrapper script if needed.
- Driver options: `model` (`--model`), `effort` (default `low`),
  `timeout_seconds` (default `1800`), `max_budget_usd` (optional
  `--max-budget-usd`), `output_format` (`stream-json` | `json`), and the
  tool allow/deny lists.

## Notes

- Runs non-interactively: `--dangerously-skip-permissions` lets the agent
  `Write` without prompts **inside the sandbox** — keep it sandboxed.
- `WebFetch` / `WebSearch` are disabled to keep the agent offline, matching the
  task constraints.
- The driver itself is not bundled here; this guide covers the setup needed to
  run Claude Code as the agent.
