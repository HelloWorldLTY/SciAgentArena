"""Multi-provider LLM clients for generating agent solutions.

Ported from the ``optimization`` branch's ``evaluate_mol_optimizer`` (the
``call_openai`` / ``call_anthropic`` / ``call_google`` helpers), generalized to
take a per-task ``system`` instruction. Provider SDKs are imported lazily, so
this module imports cleanly even if none are installed — you only need the SDK
for the provider you actually use:

    openai     -> ``openai``        (OPENAI_API_KEY, or Azure: AZURE_OPENAI_*)
    anthropic  -> ``anthropic``     (ANTHROPIC_API_KEY)
    google     -> ``google-genai``  (GOOGLE_API_KEY)

Every ``call_*`` returns ``(code, raw_response)`` where ``code`` has any
markdown code fence stripped. The temperature=0 call is retried without the
temperature argument for reasoning models that reject it.
"""

from __future__ import annotations

import importlib.util
import os

DEFAULT_MODELS = {
    "openai": "gpt-5.2",
    "anthropic": "claude-sonnet-4-6",
    "google": "gemini-3-pro-preview",
}

# provider -> importable module used to detect SDK availability
_PROVIDER_SDK = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google.genai",
}


def provider_available(provider: str) -> bool:
    """True if the provider's SDK is importable."""
    mod = _PROVIDER_SDK.get(provider)
    if not mod:
        return False
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def strip_code_fence(text: str) -> str:
    """Return the Python body of a fenced block, or the stripped text."""
    s = (text or "").strip()
    if "```python" in s:
        start = s.find("```python") + len("```python")
        end = s.find("```", start)
        if end != -1:
            return s[start:end].strip()
    if s.startswith("```"):
        first, last = s.find("```"), s.rfind("```")
        if last > first:
            return s[first + 3 : last].strip()
    return s


def _build_openai_client(model: str):
    """Return ``(client, model_to_use)``; Azure when AZURE_OPENAI_ENDPOINT set."""
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if azure_endpoint:
        from openai import AzureOpenAI

        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "AZURE_OPENAI_ENDPOINT is set but AZURE_OPENAI_API_KEY is not"
            )
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT") or model
        client = AzureOpenAI(
            azure_endpoint=azure_endpoint, api_key=api_key, api_version=api_version
        )
        return client, deployment

    from openai import OpenAI

    return OpenAI(), model


def call_openai(system: str, prompt: str, model: str) -> tuple[str, str]:
    client, model_to_use = _build_openai_client(model)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    try:
        response = client.chat.completions.create(
            model=model_to_use, messages=messages, temperature=0
        )
    except Exception as exc:  # reasoning models reject temperature; retry without
        if "temperature" in str(exc).lower():
            response = client.chat.completions.create(
                model=model_to_use, messages=messages
            )
        else:
            raise
    content = response.choices[0].message.content or ""
    return strip_code_fence(content), content


def call_anthropic(system: str, prompt: str, model: str) -> tuple[str, str]:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    kwargs = {
        "model": model,
        "max_tokens": 16000,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    try:
        message = client.messages.create(**kwargs)
    except Exception as exc:
        if "temperature" in str(exc).lower():
            kwargs.pop("temperature", None)
            message = client.messages.create(**kwargs)
        else:
            raise
    parts = [
        block.text
        for block in (getattr(message, "content", []) or [])
        if getattr(block, "type", None) == "text"
    ]
    content = "".join(parts)
    return strip_code_fence(content), content


def call_google(system: str, prompt: str, model: str) -> tuple[str, str]:
    from google import genai

    client = genai.Client()
    contents = f"{system}\n\n{prompt}"
    try:
        response = client.models.generate_content(
            model=model, contents=contents, config={"temperature": 0}
        )
    except Exception as exc:
        if "temperature" in str(exc).lower():
            response = client.models.generate_content(model=model, contents=contents)
        else:
            raise
    content = getattr(response, "text", "") or ""
    if not content:
        for candidate in getattr(response, "candidates", None) or []:
            cont = getattr(candidate, "content", None)
            for part in getattr(cont, "parts", None) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    content += part_text
    return strip_code_fence(content), content


PROVIDERS = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "google": call_google,
}
