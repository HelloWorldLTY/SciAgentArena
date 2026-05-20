# AutoBA

**Category:** Biomedicine  
**Repository:** https://github.com/JoshuaChou2018/AutoBA

## Overview

AutoBA is an AI agent for fully automated multi-omic bioinformatics analyses. It supports OpenAI models, local LLMs via Ollama, and Docker-based deployment.

## Requirements

- Python 3.10
- Conda / Mambaforge
- OpenAI API key (GPT-4 recommended) or a local Ollama model

## Installation

### Option 1: Conda / Mambaforge

```bash
# Install Mambaforge if not already available, then:
mamba install -c anaconda yaml==0.2.5 -y
pip install openai pyyaml transformers vllm

# Optional: RAG support
pip install llama-index llama-index-core llama-index-llms-ollama
```

### Option 2: Docker

```bash
docker pull joshuachou666/autoba:cuda12.2.2-cudnn8-devel-ubuntu22.04-autoba0.1.2
# Then activate the conda environment inside the container
```

### Option 3: Local LLM via Ollama

```bash
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.3.4 sh
```

Download a model (e.g., `ollama pull llama3.1`) and use `--model ollama_llama3.1` when running.

## Usage

### Basic Command

```bash
python app.py \
  --config ./examples/case1.1/config.yaml \
  --openai YOUR_OPENAI_API_KEY \
  --model gpt-4
```

### With Automatic Code Repair

```bash
python app.py --config ./examples/case1.1/config.yaml \
  --openai YOUR_OPENAI_API_KEY --model gpt-4 --execute True
```

### GUI

```bash
python gui.py
```

## Configuration File

The YAML config must specify:

```yaml
data_list:
  - /absolute/path/to/data_file
output_dir: /absolute/path/to/output
goal_description: "Describe the analysis goal here"
```

Use absolute paths. GPT-3.5 is not guaranteed to complete all tasks; GPT-4 is recommended.

## Supported Models

| Type | Examples |
|---|---|
| OpenAI (dynamic) | `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo` |
| OpenAI (snapshot) | `gpt-4-1106-preview` |
| Local via Ollama | `ollama_llama3.1`, `ollama_codellama` |
| Other | `codellama-7bi`, `deepseek` variants |
