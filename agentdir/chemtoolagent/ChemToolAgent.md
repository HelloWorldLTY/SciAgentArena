# ChemToolAgent

**Category:** Chemistry  
**Repository:** https://github.com/OSU-NLP-Group/ChemToolAgent

## Overview

ChemToolAgent is a chemistry agent built at OSU that combines GPT-4o and Claude-3.5-Sonnet with property prediction checkpoints and a Jupyter-based Python server for tool execution.

## Requirements

- Python 3.9 (conda recommended)
- GPT-4o and/or Claude-3.5-Sonnet API keys
- Property prediction checkpoints from Zenodo
- Uni-Core library (manual install required)

## Installation

### 1. Create Environment & Install Dependencies

```bash
conda create -n chemagent python=3.9
conda activate chemagent
pip install -r requirements.txt
```

### 2. Install Uni-Core

Follow the instructions at the [Uni-Core repository](https://github.com/dptech-corp/Uni-Core) to build and install it manually.

### 3. Download Checkpoints

Download property prediction model checkpoints from [Zenodo](https://zenodo.org/) and place them in:

```
chemagent/tools/property_prediction/checkpoints/
```

### 4. Configure API Keys

Fill in your API keys in `api_keys.py` (see the file for the required format).

## Usage

### Start the Backend Server

```bash
cd python_server
./start_jupyter_server.sh 8888
```

### Run a Query

```python
from api_keys import api_keys
from chemagent import ChemAgent

agent = ChemAgent(model='gpt-4o-2024-08-06', api_keys=api_keys)
query = "What is the molecular weight of Caffeine?"
final_answer, tool_use_chain, conversation, conversation_with_icl = agent.run(query)
print(final_answer)
```

### Interactive Exploration

Open `playground.ipynb` in Jupyter for interactive usage.
