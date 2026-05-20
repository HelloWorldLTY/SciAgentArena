# DELTA

**Category:** Chemistry  
**Repository:** https://github.com/deltawave-tech/delta

## Overview

DELTA is a multi-agent drug design pipeline that coordinates structure-based molecular docking (AutoDock), protein-ligand interaction analysis (PLIP), and LLM-driven design cycles. Three external services must be running before the pipeline starts.

## Requirements

- Python 3.x with `uv` package manager
- Haskell `cabal` build tool (for Gurnemanz service)
- Docker (for AutoDock service)
- OpenAI or Anthropic API key

## Installation

### 1. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or
pip install uv
```

### 2. Clone and Install Dependencies

```bash
git clone https://github.com/deltawave-tech/delta.git
cd delta
uv sync
source .venv/bin/activate
```

**Alternative (pip):**

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### 3. Configure API Keys

```bash
# Create .env in project root
cat > .env <<EOF
OPENAI_API_KEY="your-openai-key"
ANTHROPIC_API_KEY="your-anthropic-key"
EOF
```

## Starting Required Services

All three services must be running before executing the pipeline:

**1. Gurnemanz service:**
```bash
cd gurnemanz
cabal build
cabal run gurnemanz
```

**2. PLIP API service:**
```bash
uv run python plip/plip/plip_api.py
```

**3. AutoDock service:**
```bash
cd autodock
docker compose up --build
```

## Usage

```bash
uv run python src/demo/multi_agent_pipeline.py \
  --llm_provider sonnet-4 \
  --iteration_num 3
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--llm_provider` | `sonnet-4` | `sonnet-4`, `sonnet-3.7`, `o3`, `gpt4.1`, `gemini` |
| `--iteration_num` | `3` | Number of design/docking cycles |

Output is saved under the `runs/` directory with conversation logs and results.
