# TxAgent

**Category:** Clinical  
**Repository:** https://github.com/mims-harvard/TxAgent

## Overview

TxAgent is a treatment recommendation agent from Harvard MIMS that uses a retrieval-augmented tool universe (ToolUniverse) to reason across drug databases and clinical guidelines.

## Requirements

- GPU with 80+ GB VRAM recommended (H100)
- Internet connection (ToolUniverse requires live access)
- HuggingFace account (for model downloads)

## Installation

### ToolUniverse (required dependency)

**From source:**
```bash
git clone https://github.com/mims-harvard/ToolUniverse.git
cd ToolUniverse
python -m pip install . --no-cache-dir
```

**Via pip:**
```bash
pip install tooluniverse
```

### TxAgent

**From source:**
```bash
git clone https://github.com/mims-harvard/TxAgent.git
cd TxAgent
python -m pip install . --no-cache-dir
```

**Via pip:**
```bash
pip install txagent
```

## Pretrained Models

Download from HuggingFace:

| Model | Role |
|---|---|
| `TxAgent-T1-Llama-3.1-8B` | Core language model |
| `ToolRAG-T1-GTE-Qwen2-1.5B` | Tool retrieval embedding model |

## Usage

### Basic Example

```bash
python run_example.py
```

### Interactive Gradio Interface

```bash
python run_txagent_app.py
```
