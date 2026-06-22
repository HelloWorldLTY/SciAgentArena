# Agent setup guides

Per-agent setup instructions for the agents benchmarked by Agent4Science. The
unified evaluator (`../evaluate.py`) only requires an agent to produce a solution
script that satisfies a task's contract — these guides cover how to install and
configure each agent so it can do that.

## Hosted-LLM agents (built in)

GPT / Claude / Gemini are driven directly by `evaluate.py generate` — no extra
checkout, just an API key. See **[Hosted_LLM_Agents.md](Hosted_LLM_Agents.md)**
and `../.env.example`.

## Tool-using agent frameworks

Each guide covers the environment, dependencies, API keys, and the
`evaluate_<agent>.py` wrapper used to run that framework.

| Agent | Guide | Notes |
|---|---|---|
| ChemToolAgent | [ChemToolAgent_Setup.md](ChemToolAgent_Setup.md) | bridge to the official ChemAgent repo |
| ChemCrow | [ChemCrow_Setup.md](ChemCrow_Setup.md) | |
| ToolUniverse | [ToolUniverse_Setup.md](ToolUniverse_Setup.md) | Agent4Science-native tool harness |
| Cactus | [Cactus_Setup.md](Cactus_Setup.md) | |
| Delta | [Delta_Setup.md](Delta_Setup.md) | |
| DrugAgent | [DrugAgent_Setup.md](DrugAgent_Setup.md) | |
| LIDDiA | [LIDDiA_Setup.md](LIDDiA_Setup.md) | |
| Claude Code | [ClaudeCode_Setup.md](ClaudeCode_Setup.md) | drives the real `claude` CLI headless |
| Biomni | [Biomni_Setup.md](Biomni_Setup.md) | A1 agent, data lake / retriever disabled |

Claude Code and Biomni are driven by thin wrapper classes; their `.env` knobs
(`CLAUDE_CODE_CMD`, `BIOMNI_SOURCE/_LLM/_TEMPERATURE/_PATH`) are in
`../.env.example`.

## About these guides

The agent driver implementations are not bundled here — each guide covers the
environment, dependencies, and configuration needed to run that agent against
the benchmark. Paths in the guides are relative to the agent's own working
tree; adapt them to your checkout.
