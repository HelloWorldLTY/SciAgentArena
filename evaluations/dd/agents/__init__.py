"""LLM agent layer — generate agent solutions for benchmark tasks.

The benchmark's agents all follow one pattern: turn *(task + LLM)* into a
self-contained Python script, which the shared runner then executes and scores.
This package provides the hosted-LLM driver for that pattern.

    from agents import generate_solution
    code, raw, model = generate_solution(task_dict, provider="anthropic")

Supported providers: ``openai`` (incl. Azure), ``anthropic``, ``google``.
See ``.env.example`` for the API keys each one needs.
"""

from __future__ import annotations

from .dotenv import load_dotenv
from .prompts import build_system_instruction, build_user_prompt
from .providers import (
    DEFAULT_MODELS,
    PROVIDERS,
    provider_available,
    strip_code_fence,
)

__all__ = [
    "generate_solution",
    "load_dotenv",
    "build_system_instruction",
    "build_user_prompt",
    "provider_available",
    "PROVIDERS",
    "DEFAULT_MODELS",
    "strip_code_fence",
]


def generate_solution(
    task: dict, provider: str, model: str | None = None
) -> tuple[str, str, str]:
    """Generate an agent solution script for *task* with a hosted LLM.

    Returns ``(code, raw_response, model_used)``. ``code`` is the extracted
    Python body (markdown fences stripped), ready to write to a ``.py`` file
    and run via ``evaluate.py run``.
    """
    if provider not in PROVIDERS:
        raise ValueError(
            f"unknown provider {provider!r}; choose from {sorted(PROVIDERS)}"
        )
    model = model or DEFAULT_MODELS[provider]
    system = build_system_instruction(task)
    user = build_user_prompt(task)
    code, raw = PROVIDERS[provider](system, user, model)
    return code, raw, model
