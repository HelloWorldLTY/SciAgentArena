"""runners — Execute agent code and collect outputs for scoring.

Three execution modes:

* **batch_runner**: runs an agent .py script as a subprocess. The agent reads
  task input from the AGENT4S_INPUT_JSON environment variable and prints its
  results as JSON to stdout. Used for stdout-scored tasks (Chemical Data
  Preprocessing, Chemical Safety Assessment).

* **notebook_runner**: builds an in-memory Jupyter notebook, injects the agent
  code plus a hidden inspector cell, and executes via nbclient. Used for tasks
  that produce matplotlib figures or named variables (Chemical Data Analysis,
  Chemical Claim Validation).

* **design_runner**: runs an optimizer agent whose budgeted oracle calls are
  recorded to a stream, then scored (Molecule Optimization).
"""
