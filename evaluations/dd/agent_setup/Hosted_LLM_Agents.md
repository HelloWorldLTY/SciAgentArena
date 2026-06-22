# Hosted-LLM agents (OpenAI / Anthropic / Google)

The built-in agent driver. Given a task, `evaluate.py generate` builds a
mode-aware prompt, asks a hosted LLM to write a solution script, and (with
`--run`) executes + scores it through the shared runner.

## 1. Install the provider SDK(s) you'll use

```bash
uv pip install openai          # OpenAI / Azure OpenAI
uv pip install anthropic       # Anthropic
uv pip install google-genai    # Google Gemini
```

Only the provider you call is needed; the rest of the framework runs without any
of them.

## 2. Provide an API key

Copy `../.env.example` to `../.env` and fill in the relevant key(s). `.env` is
auto-loaded from the repo root (`setdefault` — never overrides your shell env).

| Provider (`--provider`) | Env var | Default model |
|---|---|---|
| `openai` | `OPENAI_API_KEY` | `gpt-5.2` |
| `openai` (Azure) | `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_API_KEY` (+ `_API_VERSION`, `_DEPLOYMENT`) | deployment name |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-4-6` |
| `google` | `GOOGLE_API_KEY` | `gemini-3-pro-preview` |

Azure is selected automatically when `AZURE_OPENAI_ENDPOINT` is set (the
deployment name is sent in place of the model id).

## 3. Generate and run

```bash
# Inspect the prompts only — no API call, no key required
python evaluate.py generate tech_01_hard_mw --provider openai --dry-run

# Generate a solution and immediately run + score it
python evaluate.py generate tech_01_hard_mw --provider anthropic --run
python evaluate.py generate ana_01_tsne --provider google --model gemini-3-pro-preview --run
```

Generated scripts are written to `results/generated/<task_id>.py` (or `--out`).
The system instruction adapts to the task's mode (stdout JSON / `fig` /
variables / oracle stream), so the script meets the runner's contract.

Check your configuration any time with:

```bash
python evaluate.py doctor      # shows each provider's SDK + key status
```
