# unified_local_llm_server

Unified async interface for local OpenAI-compatible LLM providers (Ollama, LM Studio, Unsloth, llama.cpp).

All inference goes through a single `LocalLLM` handle obtained from `server.load_model(...)`. The server itself is only used for configuration and model management.

## Installation

```bash
pip install -e .
```

Dependencies: `pydantic>=2.0`, `pyyaml>=6.0`. No OpenAI SDK required.

---

## Quick start

```python
from unified_local_llm_server import LocalLLMServer

server = LocalLLMServer("providers.yaml")

# Discover what's running
for name in server.get_providers():
    status = await server.check_provider_by_name(name)
    print(name, "✓" if status["ok"] else "✗")

# See available models
print(server.list_downloaded_models("ollama"))   # ["llama3.2:3b", "mistral:7b", ...]
print(server.list_loaded_models("ollama"))       # models currently in memory

# Load a model handle and run inference
llm = server.load_model("ollama", "llama3.2:3b")
result = await llm.call(
    messages=[{"role": "user", "content": "Hello!"}]
)
print(result)
```

---

## Provider config

Create `providers.yaml` (or `providers.yml` / `providers.json`) in your working directory, or pass the path explicitly to `LocalLLMServer`. The `LOCAL_LLM_PROVIDERS` environment variable is also supported.

```yaml
providers:
  ollama:
    host: 127.0.0.1
    port: 11434
    health_path: /api/version
    models_path: /v1/models

  lm_studio:
    host: 127.0.0.1
    port: 1234
    models_path: /v1/models

  unsloth:
    host: 127.0.0.1
    port: 8899
    api_key: sk-your-key
    health_path: /api/health
    models_path: /v1/models

  llama_cpp:
    host: 127.0.0.1
    port: 8080
    health_path: /health
    models_path: /v1/models
```

All fields except the provider key are optional — defaults are used when omitted.

### Supported providers

| Key | Default port | Notes |
|-----|-------------|-------|
| `ollama` | 11434 | |
| `lm_studio` | 1234 | |
| `unsloth` | 8895 | Client-side tool calls not supported (server-side only) |
| `llama_cpp` | 8080 | |

---

## API reference

### `LocalLLMServer`

Holds the provider registry and exposes model management. Does not run inference directly.

```python
LocalLLMServer(
    config: str | Path | None = None,   # path to providers.yaml
    *,
    timeout: float = 300.0,
    logger: AsyncLLMLogger | None = None,
)
```

#### `load_model` → `LocalLLM`

The primary method. Returns a `LocalLLM` handle bound to a specific provider and model.

```python
llm = server.load_model(
    provider: str,                       # registry key: "ollama", "lm_studio", …
    model: str,                          # model id: "llama3.2:3b", "mistral:7b", …
    *,
    logger: AsyncLLMLogger | None = None,
    context_length: int | None = None,   # shortcut for options={"num_ctx": N}
    timeout: float | None = None,
    **defaults,                          # default call kwargs applied to every llm.call()
                                         # e.g. temperature=0.2, options={"max_tokens": 512}
)
```

Connection overrides (`host`, `port`, `api_key`, `base_path`, …) are also accepted but rarely needed when using a registry.

#### Model management

```python
server.get_providers()                    # ["llama_cpp", "lm_studio", "ollama", "unsloth"]
await server.check_provider_by_name("ollama")
# {"ok": True, "provider": "ollama", "server_url": "http://127.0.0.1:11434", ...}

server.list_downloaded_models("ollama")   # all models on disk: ["llama3.2:3b", ...]
server.list_loaded_models("ollama")       # models currently in memory
server.list_loaded_instances("lm_studio") # {model_key: instance_id} — LM Studio only
server.unload_all_models("ollama")        # {"unloaded": ["llama3.2:3b"]}

# Omit provider to get a dict across all providers
server.list_downloaded_models()           # {"ollama": [...], "lm_studio": [...], ...}
server.list_loaded_models()
server.unload_all_models()
```

---

### `LocalLLM`

Model handle for all inference. Created by `server.load_model(...)`.

#### `llm.call`

```python
result = await llm.call(
    messages: list[dict],                # required — OpenAI chat message list
    *,
    think: bool = False,                 # enable chain-of-thought / reasoning
    temperature: float | None = None,
    effort: str | None = None,           # reasoning effort: "low" | "medium" | "high"
    options: dict | None = None,         # provider-specific options, e.g. {"num_ctx": 8192}
    schema_dict: dict | None = None,     # JSON schema → structured output (see below)
    tools: list[dict] | None = None,     # OpenAI tool definitions
    tool_registry: dict | None = None,   # {tool_name: callable} handlers
    max_tool_rounds: int = 5,            # max LLM ↔ tool iterations
    max_json_fix_retries: int = 3,       # retries when structured output parsing fails
    max_loop_retries: int = 5,           # retries when output loop is detected
)
```

Returns `str` by default, or a `dict` / Pydantic model when `schema_dict` is given.

Defaults set at `load_model` time are merged with per-call kwargs. Per-call values win; `options` dicts are shallow-merged (per-call keys override, others kept).

```python
# temperature=0.2 applied to every call unless overridden
llm = server.load_model("ollama", "llama3.2:3b", temperature=0.2)

result = await llm.call(
    messages=[{"role": "user", "content": "Hi"}],
    options={"max_tokens": 100},
)
```

#### `llm.batch`

Run multiple calls concurrently with a semaphore.

```python
results = await llm.batch(
    items=[
        [{"role": "user", "content": "Question 1"}],
        [{"role": "user", "content": "Question 2"}],
        {"messages": [...], "temperature": 0.5},  # per-item overrides
    ],
    concurrency=4,              # max parallel calls (default 4)
    return_exceptions=False,    # True → exceptions returned as values instead of raised
    **shared_call_kwargs,       # applied to every item, overridden by per-item kwargs
)
# returns list aligned with items
```

Items can be either a plain message list `[{...}, ...]` or a dict with a `"messages"` key plus any extra call kwargs.

---

### Tool calling

Tools are defined in OpenAI function-call format. Handlers can be sync or async callables; each receives the argument dict and returns a string (or any JSON-serialisable value).

The library runs the full multi-round loop automatically:
model returns `tool_calls` → handlers execute → results sent back → model produces final answer.

```python
import datetime

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Returns the current date and time.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Adds two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        },
    },
]

TOOL_REGISTRY = {
    "get_time": lambda _: datetime.datetime.now().isoformat(),
    "add":      lambda args: str(args["a"] + args["b"]),
}

result = await llm.call(
    messages=[{"role": "user", "content": "What is 123 + 456, and what time is it?"}],
    tools=TOOLS,
    tool_registry=TOOL_REGISTRY,
    think=True,
)
```

> **Unsloth:** Client-side tool calls are not supported. Passing `tools=` raises `NotImplementedError`. Unsloth executes tools server-side via its own `enable_tools` mechanism.

#### MCP bridge

Forward tool calls to a running [FastMCP](https://gofastmcp.com) server by wrapping `Client.call_tool` as the handler:

```python
from fastmcp import Client

MCP_URL = "http://127.0.0.1:8325/mcp"

# Fetch schemas once
async with Client(MCP_URL) as client:
    mcp_tools = await client.list_tools()

def to_openai_tool(t):
    schema = {k: v for k, v in (t.inputSchema or {}).items() if k != "additionalProperties"}
    return {"type": "function", "function": {
        "name": t.name,
        "description": t.description or "",
        "parameters": schema,
    }}

def make_handler(name):
    async def handler(args):
        async with Client(MCP_URL) as c:
            r = await c.call_tool(name, args)
            return r.data if isinstance(r.data, str) else str(r.data)
    return handler

TOOLS         = [to_openai_tool(t) for t in mcp_tools]
TOOL_REGISTRY = {t.name: make_handler(t.name) for t in mcp_tools}

result = await llm.call(
    messages=[{"role": "user", "content": "Search for Python 3.13 new features."}],
    tools=TOOLS,
    tool_registry=TOOL_REGISTRY,
    think=True,
)
```

---

### Structured output

Pass a JSON schema to get a parsed `dict` back. The library prompts the model, validates the response, and retries with corrections on parse failure.

```python
schema = {
    "type": "object",
    "properties": {
        "name":  {"type": "string"},
        "score": {"type": "integer"},
    },
    "required": ["name", "score"],
}

result = await llm.call(
    messages=[{"role": "user", "content": "Extract: Alice scored 42 points."}],
    schema_dict=schema,
)
# {"name": "Alice", "score": 42}
```

---

### Logging

`AsyncLLMLogger` writes a structured stream log of every request/response pair, including thinking blocks, tool execution results, and timing.

```python
from unified_local_llm_server.llm_logger import AsyncLLMLogger

logger = AsyncLLMLogger("logs/session.log")
llm    = server.load_model("ollama", "llama3.2:3b", logger=logger)

await llm.call(...)

await logger.close()
```

Log format markers:

| Marker | Content |
|--------|---------|
| `|REQ-MESSAGES|` | Full message history sent to the model |
| `|REQ-PRM|` | Request parameters (model, temperature, tools, …) |
| `[THINKING]` | Model reasoning / chain-of-thought |
| `[MESSAGE]` | Model output text |
| `|INFO|` | Tool execution results and other events |

Set `LOG_MSG_REFS=1` to deduplicate repeated messages using SHA1 refs (useful in long tool-call sessions).

---

### Unsloth — automatic model loading

Unsloth models are loaded on demand before the first inference call. Pass the `repo_id@quant` shorthand (as returned by `list_downloaded_models`) — it is resolved to the full `.gguf` path via the Unsloth API automatically:

```python
# shorthand — resolved to the actual .gguf path
llm = server.load_model("unsloth", "unsloth/gpt-oss-20b-GGUF@UD-Q4_K_XL")

# or explicit path
llm = server.load_model("unsloth", "/mnt/models/gpt-oss-20b-UD-Q4_K_XL.gguf")
```

If the model is already loaded on the server, the load step is skipped.

---

## Environment variables

| Variable | Purpose |
|----------|---------|
| `LOCAL_LLM_PROVIDERS` | Path to providers config file (fallback when no path given to `LocalLLMServer`) |
| `UNSLOTH_KEY` / `UNSLOTH_API_KEY` | Unsloth API key |
| `LOG_MSG_REFS` | Set to `1` to deduplicate repeated messages in logs by SHA1 reference |
