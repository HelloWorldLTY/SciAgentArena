# STELLA

**Category:** Biomedicine  
**Repository:** https://github.com/zaixizhang/STELLA

## Overview

STELLA is a biomedical research agent with a web interface, skill retrieval, and dynamic tool creation. It can also be used via the hosted version at [stella-agent.com](https://stella-agent.com/) without any local installation.

## Installation

### Option 1: Docker (Fastest Local Setup)

A pre-built Docker image is available on Google Drive. See `docker/README.md` in the repository for download and run instructions.

### Option 2: Conda (Recommended for Local)

```bash
git clone https://github.com/zaixizhang/STELLA.git
cd STELLA

conda create -n stella python=3.12 -y
conda activate stella

conda install -c conda-forge numpy pandas scikit-learn matplotlib seaborn -y
pip install -r requirements.txt
```

### Option 3: pip only

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```bash
echo "OPENROUTER_API_KEY=your_key_here" > .env
```

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes | Provides access to GPT, Claude, Gemini, etc. |
| `SERPAPI_API_KEY` | Optional | Web search capability |
| `PAPERQA_API_KEY` | Optional | Enhanced paper Q&A |

## Usage

### Web Interface

```bash
python stella_core.py
```

Open `http://localhost:7860` in your browser.

**CLI options:**

```bash
python stella_core.py --no_template           # disable skill retrieval
python stella_core.py --enable_tool_creation  # activate dynamic tool creation
python stella_core.py --port 8080             # custom port
```

### Programmatic Usage

```python
from stella_core import initialize_stella

manager_agent = initialize_stella(use_template=True)
result = manager_agent.run("Your research question here")
```

## Notes

- Large biomedical resource files are optional; download from Google Drive and extract to `resource/` for enhanced performance.
- The active LLM model can be changed by editing variables in `stella_core.py`.
