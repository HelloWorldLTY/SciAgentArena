"""Shared helpers for the ToolUniverse agent scripts.

This module centralises:

- loading environment variables (including an optional ``.env`` file),
- creating the LLM client (Azure OpenAI or OpenAI),
- tool retrieval via ``Tool_Finder_Keyword``,
- the tool-using chat loop.

No secrets are stored here. All credentials must be provided via environment
variables (see ``.env.example``).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple


FORMAT_SUFFIX = (
    "\n\nOutput only executable Python code, no markdown code fence or "
    "explanation. Use import scanpy as sc when Scanpy is relevant."
)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def load_dotenv_if_present(path: str) -> None:
    """Minimal ``.env`` loader (no external dependency).

    Parses ``KEY=VALUE`` lines, skipping comments and blank lines. Values may
    be quoted with single or double quotes. Existing process environment
    variables take precedence so a shell ``export`` always wins.
    """

    if not path or not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if (value.startswith("\"") and value.endswith("\"")) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            os.environ.setdefault(key, value)


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------


@dataclass
class LLMClient:
    """Thin wrapper around OpenAI / AzureOpenAI ``chat.completions.create``."""

    client: object
    model: str
    provider: str  # "azure" or "openai"

    def create(self, **kwargs):  # pragma: no cover - thin delegation
        return self.client.chat.completions.create(**kwargs)


def build_llm_client() -> LLMClient:
    """Build an LLM client from environment variables.

    Priority:
        1. Azure OpenAI when ``AZURE_OPENAI_KEY`` and ``AZURE_OPENAI_ENDPOINT``
           are both present.
        2. OpenAI when ``OPENAI_API_KEY`` is present.

    Raises ``SystemExit`` with a helpful message if neither is configured.
    """

    azure_key = os.environ.get("AZURE_OPENAI_KEY") or os.environ.get(
        "AZURE_OPENAI_API_KEY"
    )
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if azure_key and azure_endpoint:
        from openai import AzureOpenAI  # type: ignore

        api_version = os.environ.get(
            "AZURE_OPENAI_API_VERSION", "2024-12-01-preview"
        )
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            api_key=azure_key,
        )
        return LLMClient(client=client, model=deployment, provider="azure")

    if openai_key:
        from openai import OpenAI  # type: ignore

        model = os.environ.get("OPENAI_MODEL", "gpt-4o")
        client = OpenAI(api_key=openai_key)
        return LLMClient(client=client, model=model, provider="openai")

    sys.stderr.write(
        "ERROR: no LLM credentials found. Set AZURE_OPENAI_ENDPOINT + "
        "AZURE_OPENAI_KEY (Azure) or OPENAI_API_KEY (OpenAI), or create a "
        ".env file next to this script.\n"
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# ToolUniverse helpers
# ---------------------------------------------------------------------------


def load_tool_universe():
    """Import and initialise ToolUniverse, redirecting stdout to stderr.

    ``ToolUniverse.load_tools`` writes loader status to stdout which would
    pollute pipes (e.g. when a caller redirects stdout to capture code). We
    redirect it to stderr here.
    """

    from tooluniverse import ToolUniverse  # type: ignore

    tu = ToolUniverse()
    with contextlib.redirect_stdout(sys.stderr):
        tu.load_tools()
    return tu


def _tools_to_functions(tu, tools) -> list:
    if not tools or not isinstance(tools, list):
        return []
    functions = []
    for t in tools:
        name = t.get("name") if isinstance(t, dict) else getattr(t, "name", None)
        if not name:
            continue
        spec = tu.tool_specification(name)
        if spec:
            functions.append(
                {
                    "name": spec["name"],
                    "description": spec.get("description", ""),
                    "parameters": spec.get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                }
            )
    return functions


DEFAULT_TOOL_HINT = (
    "scanpy anndata h5ad single cell RNA spatial transcriptomics "
    "scrublet harmony scanorama scvi leiden umap pca highly variable genes "
    "differential expression rank_genes_groups squidpy spatial neighbors"
)


def find_relevant_tools(tu, prompt: str, *, limit: int = 12, hint: str = "") -> list:
    """Use ``Tool_Finder_Keyword`` to fetch a shortlist of relevant tools."""

    cleaned = " ".join(prompt.strip().split())
    if len(cleaned) > 480:
        cleaned = cleaned[:480]
    description = f"{cleaned} {DEFAULT_TOOL_HINT}"
    if hint:
        description = f"{description} {hint}"
    with contextlib.redirect_stdout(sys.stderr):
        tools = tu.run(
            {
                "name": "Tool_Finder_Keyword",
                "arguments": {"description": description, "limit": limit},
            }
        )
    return _tools_to_functions(tu, tools)


# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------


MAX_TOOL_RESULT_CHARS = 12000


def _truncate(content: str) -> str:
    if not content or len(content) <= MAX_TOOL_RESULT_CHARS:
        return content
    return content[:MAX_TOOL_RESULT_CHARS] + "\n\n[... truncated ...]"


def _extract_tool_calls(message):
    tool_calls = getattr(message, "tool_calls", None) or None
    function_call = getattr(message, "function_call", None) or None
    calls: List[Tuple[str, str, str]] = []
    if tool_calls:
        for tc in tool_calls:
            calls.append((tc.id, tc.function.name, tc.function.arguments or "{}"))
        return "tool_calls", calls
    if function_call:
        calls.append(
            ("legacy", function_call.name, function_call.arguments or "{}")
        )
        return "function_call", calls
    return None, calls


def chat_with_tools(
    llm: LLMClient,
    tu,
    prompt: str,
    functions: list,
    *,
    system_msg: Optional[str] = None,
    max_rounds: int = 15,
    max_completion_tokens: int = 16384,
    tool_event: Optional[Callable[[str, dict], None]] = None,
    first_round_required: bool = True,
    task_label: str = "",
) -> Tuple[str, List[str]]:
    """Run the tool-calling chat loop. Returns ``(final_text, used_tools)``.

    ``tool_event`` is an optional callback invoked with ``(event_name, info)``
    for external logging.
    """

    messages: list = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": prompt})
    used_tools: List[str] = []
    round_idx = 0
    while round_idx < max_rounds:
        kwargs = {
            "model": llm.model,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
            "temperature": 0.2,
        }
        if functions:
            kwargs["tools"] = [
                {"type": "function", "function": f} for f in functions
            ]
            kwargs["tool_choice"] = (
                "required" if (first_round_required and round_idx == 0) else "auto"
            )

        # Retry loop for parameter / tool compatibility differences
        while True:
            try:
                response = llm.create(**kwargs)
                break
            except Exception as e:  # pragma: no cover - network failure path
                err = str(e).lower()
                if "max_tokens" in err and "max_completion_tokens" not in kwargs:
                    kwargs.setdefault(
                        "max_completion_tokens", max_completion_tokens
                    )
                    kwargs.pop("max_tokens", None)
                    continue
                if functions and kwargs.get("tools") and "tool" in err:
                    kwargs.pop("tools", None)
                    kwargs.pop("tool_choice", None)
                    continue
                raise

        round_idx += 1
        message = response.choices[0].message
        kind, calls = _extract_tool_calls(message)
        if not calls:
            return (message.content or ""), used_tools

        if kind == "tool_calls":
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or None,
                    "tool_calls": [
                        {
                            "id": cid,
                            "type": "function",
                            "function": {"name": cname, "arguments": cargs},
                        }
                        for cid, cname, cargs in calls
                    ],
                }
            )
        else:
            cid, cname, cargs = calls[0]
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or None,
                    "function_call": {"name": cname, "arguments": cargs},
                }
            )

        for cid, cname, cargs in calls:
            used_tools.append(cname)
            prefix = f"[{task_label}] " if task_label else ""
            print(f"{prefix}[Agent called tool] {cname}", file=sys.stderr)
            if tool_event:
                try:
                    tool_event("tool_call", {"name": cname, "arguments": cargs})
                except Exception:
                    pass
            try:
                arguments = json.loads(cargs)
            except json.JSONDecodeError:
                arguments = {}
            try:
                result = tu.run({"name": cname, "arguments": arguments})
            except Exception as e:  # pragma: no cover - tool failure path
                result = {"error": f"tool execution failed: {e!r}"}
            content = (
                json.dumps(result) if not isinstance(result, str) else result
            )
            content = _truncate(content)
            if kind == "tool_calls":
                messages.append(
                    {"role": "tool", "tool_call_id": cid, "content": content}
                )
            else:
                messages.append(
                    {"role": "function", "name": cname, "content": content}
                )
    return "", used_tools


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------


_CODE_FENCE_ONE = re.compile(r"^```(?:python)?\s*\n(.*?)```\s*$", re.DOTALL)
_CODE_FENCE_MANY = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Strip Markdown fences and return only Python code."""
    if not text or not text.strip():
        return ""
    text = text.strip()
    m = _CODE_FENCE_ONE.match(text)
    if m:
        return m.group(1).strip()
    blocks = _CODE_FENCE_MANY.findall(text)
    if blocks:
        return "\n\n".join(b.strip() for b in blocks)
    return text


def append_format_suffix(prompt: str) -> str:
    """Append the standard ``output only code`` hint unless already present."""
    suffix = FORMAT_SUFFIX.strip()
    if suffix in prompt:
        return prompt
    return f"{prompt}\n\n{suffix}"
