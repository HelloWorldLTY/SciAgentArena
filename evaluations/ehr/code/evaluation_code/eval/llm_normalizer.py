"""LLM-based component extraction using GPT-4o-mini with hash-based caching."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from openai import OpenAI


CACHE_DIR = Path(__file__).resolve().parent.parent / ".llm_cache"

EXTRACTION_PROMPT_TEMPLATE = """\
Given the following text from a causal estimand field "{field_name}", \
determine whether each of the following semantic components is present in the text.

Text: "{agent_text}"

Components to check (each component lists alternative phrasings separated by |, \
mark true if ANY alternative is present or semantically implied):
{components_list}

Return ONLY a JSON object where keys are the EXACT component strings as listed \
(including the | characters) and values are true or false.

Do not include any other text or explanation.

Example: if the component is "no|absence|without" and the text says "did not receive", \
that matches "no" semantically, so mark it true."""


def _cache_key(agent_text: str, field_name: str, components: list[str]) -> str:
    """Generate a deterministic hash key for caching."""
    payload = json.dumps(
        {"agent_text": agent_text, "field_name": field_name, "components": components},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_from_cache(key: str) -> dict[str, bool] | None:
    """Load cached extraction result if it exists."""
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    return None


def _save_to_cache(key: str, result: dict[str, bool]) -> None:
    """Save extraction result to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{key}.json"
    with open(cache_file, "w") as f:
        json.dump(result, f, indent=2)


def extract_components(
    agent_text: str,
    field_name: str,
    required_components: list[str],
    use_cache: bool = True,
) -> dict[str, bool]:
    """Extract whether each required component is present in the agent's text.

    Uses GPT-4o-mini for semantic extraction. Results are cached by default.

    Args:
        agent_text: The agent's free-text answer for this field.
        field_name: Name of the estimand field (e.g., "population").
        required_components: List of component descriptions to check.
            Each may contain pipe-delimited alternatives (e.g., "IV|intravenous").
        use_cache: Whether to use hash-based file caching.

    Returns:
        Dict mapping each component description to True/False.
    """
    # Use the full pattern string as the component key for transparency
    component_labels = required_components

    cache_key = _cache_key(agent_text, field_name, component_labels)

    if use_cache:
        cached = _load_from_cache(cache_key)
        if cached is not None:
            return cached

    # Build the numbered components list for the prompt
    components_list = "\n".join(
        f"{i + 1}. {comp}" for i, comp in enumerate(component_labels)
    )

    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        field_name=field_name,
        agent_text=agent_text,
        components_list=components_list,
    )

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=500,
    )

    raw_output = response.choices[0].message.content.strip()

    # Parse JSON from the response, handling potential markdown code blocks
    if raw_output.startswith("```"):
        raw_output = raw_output.split("```")[1]
        if raw_output.startswith("json"):
            raw_output = raw_output[4:]
        raw_output = raw_output.strip()

    result = json.loads(raw_output)

    # Normalize: ensure all component labels are present in the result
    normalized_result = {}
    for comp in component_labels:
        # Try exact match first, then case-insensitive search
        if comp in result:
            normalized_result[comp] = bool(result[comp])
        else:
            # Search for a key that matches any alternative in the component
            found = False
            for key, val in result.items():
                if key.lower() in comp.lower() or comp.lower() in key.lower():
                    normalized_result[comp] = bool(val)
                    found = True
                    break
            if not found:
                # Component not found in LLM output — treat as not present
                normalized_result[comp] = False

    if use_cache:
        _save_to_cache(cache_key, normalized_result)

    return normalized_result
