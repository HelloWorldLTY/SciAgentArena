# Medea

**Category:** Clinical  
**Repository:** https://github.com/mims-harvard/MEDEA

## Overview

Medea is a clinical research agent from Harvard MIMS built on the AgentLite framework. It supports three usage modes: full research planning + analysis + literature synthesis, research + analysis only, or literature reasoning only. It relies on MedeaDB (hosted on HuggingFace) for its knowledge base.

## Requirements

- Python 3.10
- `uv` package manager
- OpenRouter API key (or Azure / Google / Anthropic / NVIDIA credentials)
- `git-lfs` for downloading MedeaDB

## Installation

### 1. Clone and Set Up Environment

```bash
git clone https://github.com/mims-harvard/MEDEA.git
cd MEDEA

pip install uv
uv venv medea --python 3.10
source medea/bin/activate        # Windows: medea\Scripts\activate
```

### 2. Install Package

```bash
uv pip install -e .
uv pip install openai==1.82.1
```

### 3. Download MedeaDB

```bash
uv pip install -U huggingface_hub
huggingface-cli login

# Install git-lfs
brew install git-lfs          # macOS
# or
sudo apt-get install git-lfs  # Linux

git lfs install
git clone https://huggingface.co/datasets/mims-harvard/MedeaDB
```

## Configuration

```bash
cp .env.example .env
```

Edit `.env` with the following:

| Variable | Description |
|---|---|
| `MEDEADB_PATH` | Local path to the downloaded MedeaDB |
| `BACKBONE_LLM` | Model name (default: `gpt-4o`) |
| `SEED` | Reproducibility seed (default: `42`) |
| `OPENROUTER_API_KEY` | Primary API credentials |

### Alternative LLM Providers

Set the appropriate environment variables for:
- Azure OpenAI
- Google Gemini
- Anthropic Claude
- NVIDIA DeepSeek

## Usage

Three primary modes are available — see the `examples/` directory for detailed scripts:

1. **Full agent** — research planning, computational analysis, and literature synthesis
2. **Research + analysis only** — experiments without literature search
3. **Literature reasoning only** — paper search and synthesis without experiments
