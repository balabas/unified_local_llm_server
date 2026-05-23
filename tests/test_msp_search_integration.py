"""Integration test: msp_search tools via all accessible providers.

Tools are called through the live MCP server at MCP_SERVER_URL.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client

sys.path.insert(0, str(Path(__file__).parent.parent))

from unified_local_llm_server import LLMProviderPool

REGISTRY_PATH = Path(__file__).parent.parent / "providers.example.yaml"
MCP_SERVER_URL = "http://127.0.0.1:8325/mcp"

# Models confirmed to support tool calling for each provider
TOOL_MODELS = {
    "ollama":    "gpt-oss:20b",
    "lm_studio": "google/gemma-4-e4b",
    # unsloth: server-side tools only — not included here
}

UNSLOTH_GPT_OSS = (
    "/mnt/M/llms_models/hg_fc/hub/models--unsloth--gpt-oss-20b-GGUF"
    "/snapshots/d449b42d93e1c2c7bda5312f5c25c8fb91dfa9b4"
    "/gpt-oss-20b-UD-Q4_K_XL.gguf"
)

# ---------------------------------------------------------------------------
# MCP bridge: build TOOLS + tool_registry from the live MCP server
# ---------------------------------------------------------------------------

def _mcp_tool_to_openai(tool: Any) -> dict[str, Any]:
    """Convert a FastMCP Tool object to OpenAI function schema."""
    schema = dict(tool.inputSchema or {})
    schema.pop("additionalProperties", None)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": schema,
        },
    }


def _make_mcp_handler(tool_name: str) -> Any:
    """Return an async handler that calls the MCP server for the given tool."""
    async def handler(args: dict[str, Any]) -> str:
        async with Client(MCP_SERVER_URL) as client:
            result = await client.call_tool(tool_name, args)
            if result.is_error:
                return f"[MCP error] {result.content}"
            return result.data if isinstance(result.data, str) else str(result.data)
    return handler


def _load_mcp_tools() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch tool schemas from MCP server; return (openai_tools, tool_registry)."""
    async def _fetch():
        async with Client(MCP_SERVER_URL) as client:
            return await client.list_tools()

    mcp_tools = asyncio.run(_fetch())
    openai_tools = [_mcp_tool_to_openai(t) for t in mcp_tools]
    registry = {t.name: _make_mcp_handler(t.name) for t in mcp_tools}
    return openai_tools, registry


# Load once at collection time
TOOLS, TOOL_REGISTRY = _load_mcp_tools()

# ---------------------------------------------------------------------------
# Provider helpers
# ---------------------------------------------------------------------------

_accessible: list[str] | None = None


def _get_accessible() -> list[str]:
    global _accessible
    if _accessible is not None:
        return _accessible
    server = LLMProviderPool(REGISTRY_PATH)

    async def _check() -> list[str]:
        ok = []
        for name in server.provider_registry.names():
            try:
                if (await server.check_provider(name)).get("ok"):
                    ok.append(name)
            except Exception:
                pass
        return ok

    _accessible = asyncio.run(_check())
    return _accessible


def _tool_pairs() -> list[tuple[str, str]]:
    return [(p, m) for p, m in TOOL_MODELS.items() if p in _get_accessible()]


def _all_inference_pairs() -> list[tuple[str, str]]:
    pairs = list(TOOL_MODELS.items())
    if "unsloth" in _get_accessible():
        pairs.append(("unsloth", UNSLOTH_GPT_OSS))
    return [(p, m) for p, m in pairs if p in _get_accessible()]


def _make_server() -> LLMProviderPool:
    return LLMProviderPool(REGISTRY_PATH)


def _is_live_model_unavailable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        marker in msg
        for marker in (
            "failed to load model",
            "load failed",
            "/load failed",
            "connection reset",
            "http 400",
            "http 500",
            "http 503",
        )
    )


def _run_or_skip(coro: Any, provider: str, model: str) -> str:
    try:
        return asyncio.run(coro)
    except Exception as exc:
        if _is_live_model_unavailable(exc):
            pytest.skip(f"{provider}/{model} unavailable in this live provider state: {exc}")
        raise

# ---------------------------------------------------------------------------
# Tool-call tests (gpt-oss:20b on ollama, gemma-4-e4b on lm_studio)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("provider,model", _tool_pairs())
def test_date_tool(provider: str, model: str) -> None:
    """Model calls get_current_date_time via MCP and returns a date string."""
    server = _make_server()

    async def _run() -> str:
        llm = server.load_model(provider, model)
        return await llm.call(
            messages=[{"role": "user", "content": "What is the current date and time? Use the tool."}],
            tools=TOOLS,
            tool_registry=TOOL_REGISTRY,
            think=True,
            options={"max_tokens": 300},
        )

    result = _run_or_skip(_run(), provider, model)
    print(f"\n[{provider}/{model}] date → {result!r}")
    assert result, f"{provider}: empty response"
    assert any(y in result for y in ("2025", "2026")), f"{provider}: no year in {result!r}"


@pytest.mark.parametrize("provider,model", _tool_pairs())
def test_search_tool(provider: str, model: str) -> None:
    """Model calls search_web via MCP and summarises results."""
    server = _make_server()

    async def _run() -> str:
        llm = server.load_model(provider, model)
        return await llm.call(
            messages=[{"role": "user", "content": "Search for 'Python 3.13 new features' and give a one-sentence summary."}],
            tools=TOOLS,
            tool_registry=TOOL_REGISTRY,
            think=True,
            options={"max_tokens": 500},
        )

    result = _run_or_skip(_run(), provider, model)
    print(f"\n[{provider}/{model}] search → {result!r}")
    assert result and len(result) > 20, f"{provider}: response too short: {result!r}"


# ---------------------------------------------------------------------------
# Basic inference test (all providers including unsloth)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("provider,model", _all_inference_pairs())
def test_basic_inference(provider: str, model: str) -> None:
    """Model responds to a plain question without tools."""
    server = _make_server()

    async def _run() -> str:
        llm = server.load_model(provider, model)
        return await llm.call(
            messages=[{"role": "user", "content": "Reply with exactly one word: hello"}],
            options={"max_tokens": 20},
        )

    result = _run_or_skip(_run(), provider, model)
    label = model.split("/")[-1][:30]
    print(f"\n[{provider}/{label}] inference → {result!r}")
    assert result, f"{provider}: empty response"
