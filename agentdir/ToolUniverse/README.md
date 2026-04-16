# ToolUniverse Agent

Drop in a natural-language prompt; ChatGPT + [ToolUniverse](https://zitniklab.hms.harvard.edu/ToolUniverse/)
plan with biomedical tools (Scanpy, AnnData, Scrublet, Harmony, SCVI, Scanorama, etc.),
and write back **two files** under `outputs/`:

- `<stem>.py` — the generated Python code (nothing else, ready to execute)
- `<stem>.txt` — the full run log (prompt, tool calls, raw model reply)

A multi-step pipeline runner is also included: list T1/T2/... prompts in one
file and get a single `onestep.py` plus per-task text logs.

## Install

```bash
# inside a clean env (tested on Python 3.10/3.11)
pip install -r requirements.txt
```

Required Python packages are listed in `requirements.txt` (`openai`,
`tooluniverse`). Targets that run the generated code downstream typically need
`scanpy`, `anndata`, etc.; install those in the environment where you execute
the generated `.py`.

## Configure credentials

Copy `.env.example` to `.env` and fill in the values. **Never commit `.env`.**

```bash
cp .env.example .env
# edit .env to set your Azure OpenAI or OpenAI key
```

Supported providers (auto-detected):

- **Azure OpenAI** — set `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`,
  `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION`.
- **OpenAI** — set `OPENAI_API_KEY` (optionally `OPENAI_MODEL`, default
  `gpt-4o`).

The scripts load `.env` from the agent directory automatically (no extra
dependency). You can also just `export` the variables in your shell.

## Quick start: one prompt → `.py` + `.txt`

```bash
python run_prompt.py --prompt-file prompts/example_hvg_pca.txt --name hvg_pca
```

This writes:

- `outputs/hvg_pca.py` — extracted Python
- `outputs/hvg_pca.txt` — prompt, tool calls in call order, raw model reply

Single-line prompt via flag:

```bash
python run_prompt.py --prompt 'Normalize counts and compute PCA on adata' --name normalize
```

From stdin (useful for piping):

```bash
cat my_prompt.txt | python run_prompt.py --name my_run
```

Useful flags:

| Flag | Default | Purpose |
|------|---------|---------|
| `--name STEM` | timestamp | output basename (shared by `.py` and `.txt`) |
| `--out-dir DIR` | `outputs` | directory to write outputs into |
| `--prompt-file FILE` | – | read prompt from a UTF-8 file |
| `--prompt TEXT` | – | inline prompt (single line) |
| `--tool-limit N` | `12` | how many tools to inject as function calls |
| `--tool-hint TEXT` | – | extra keywords for the tool finder |
| `--no-format-suffix` | off | do not append the "output only code" suffix |
| `--max-rounds N` | `15` | max tool-calling rounds before giving up |

## Multi-step pipeline → `onestep.py`

Write all steps in one file with `# TASK <label>` headers; for example
`prompts/spatial_t1_t11_onestep.txt`:

```
# TASK T1
You are an expert in bioinformatics, please write a code to read ...

# TASK T2
Generate concise, executable Python (Scanpy) code that annotates gene ...

# TASK SAVE
After these tasks, write code to save the resulting anndata object ...
```

Run:

```bash
python run_pipeline.py --prompt-file prompts/spatial_t1_t11_onestep.txt --name spatial
```

This writes:

- `outputs/spatial_onestep.py` — concatenated Python (one `# ========== Tn ==========` header per step)
- `outputs/spatial_output.txt` — per-step tool calls, raw reply, extracted code
- `outputs/spatial_tools_used.txt` — compact tool-call summary per step

Each step is generated independently, so later steps do not re-emit code
from earlier steps; just run the resulting `onestep.py` top-to-bottom.

## Example prompts included

- `prompts/example_read_adata.txt` — read an `.h5ad` given `path` and
  normalize obs keys to `sample` / `cell_type`.
- `prompts/example_hvg_pca.txt` — HVG + PCA on an existing `adata`.
- `prompts/scrna_t1_t10_onestep.txt` — full single-cell pipeline (T1–T10).
- `prompts/spatial_t1_t11_onestep.txt` — full spatial pipeline (T1–T11 + SAVE).

## How it works

1. Call `ToolUniverse.load_tools()` and use `Tool_Finder_Keyword` to pick a
   short, prompt-relevant toolset (no LLM required for retrieval).
2. Convert each tool spec into an OpenAI function-call schema.
3. Hand the prompt + tools to ChatGPT; on the first round we set
   `tool_choice="required"` so the model cannot skip tool grounding.
4. Loop: execute any tool the model calls, feed results back, repeat until
   the model emits the final reply.
5. Strip the triple-backtick fence and persist the code separately from the
   full log.

## Notes

- The generated code typically uses `import scanpy as sc` and assumes an
  `AnnData` named `adata`. For data-loading prompts, `path` is the input file
  variable.
- When an Azure deployment rejects `max_tokens`, the runner automatically
  retries with `max_completion_tokens` — no action needed.
- If you do not want the "output only code" suffix to be appended to your
  prompt, pass `--no-format-suffix`.

## References

- ToolUniverse building AI scientists guide: <https://zitniklab.hms.harvard.edu/ToolUniverse/guide/building_ai_scientists/chatgpt_api.html>
