# unified_local_llm_server

Unified async interface for local OpenAI-compatible LLM providers (Ollama, LM Studio, Unsloth, llama.cpp).

All inference goes through a single `LocalLLM` handle obtained from `pool.load_model(...)`. The server itself is only used for configuration and model management.

## Requirements

| | |
|---|---|
| **OS** | Linux with systemd (Ubuntu 20.04+, Fedora 36+, or equivalent) |
| **Python** | 3.12+ |
| **GPU** | Optional — CUDA-capable GPU for hardware acceleration |
| **Providers** | At least one of: Ollama, LM Studio, Unsloth Studio, llama.cpp |

> **Root access required once:** `scripts/setup_service_restart.sh` must be run with `sudo` to write to `/etc/polkit-1/rules.d/` (polkit rules) and `/etc/systemd/system/` (service files) — both are root-owned system directories. The polkit rule grants the current user passwordless `systemctl restart` for the listed services only. After that, providers can be temporary reconfigured and restarted without a password from any context (terminal, Jupyter, scripts).

## Installation

```bash
pip install -e .
```

Dependencies: `pydantic>=2.0`, `pyyaml>=6.0`. No OpenAI SDK required.

To enable provider restart support (optional), run the setup script once:

```bash
sudo bash scripts/setup_service_restart.sh
```

This is separate from `pip install` — it writes polkit rules and systemd service files to system directories and cannot be done by pip.

---

## Quick start

```python
from unified_local_llm_server import LLMProviderPool

pool = LLMProviderPool("providers.yaml")

# Discover what's running
for name in pool.get_providers():
    status = await pool.check_provider(name)
    print(name, "✓" if status["ok"] else "✗")

# Per-provider handle: browse models, set defaults, run inference
ollama = pool.get_provider("ollama")
print(ollama.list_downloaded_models())   # [{"name": "llama3.2:3b"}, ...]

ollama.configure(num_parallel=2, kv_cache_type="q8_0")  # set once, inherited by all load_model calls

model = ollama.load_model("llama3.2:3b")
result = await model.call(messages=[{"role": "user", "content": "Hello!"}])
print(result)
```

---

## Provider config

Create `providers.yaml` (or `providers.yml` / `providers.json`) in your working directory, or pass the path explicitly to `LLMProviderPool`. The `LOCAL_LLM_PROVIDERS` environment variable is also supported.

```yaml
providers:
  ollama:
    systemd_service: ollama.service        # optional — enables restart()
    host: 127.0.0.1
    port: 11434
    health_path: /api/version
    models_path: /v1/models

  lm_studio:
    systemd_service: lm_studio_serv.service
    host: 127.0.0.1
    port: 1234
    models_path: /v1/models

  unsloth:
    systemd_service: unsloth_studio.service
    host: 127.0.0.1
    port: 8899
    api_key: sk-your-key
    health_path: /health
    models_path: /v1/models

  llama_cpp:
    systemd_service: llama_cpp.service
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

### `LLMProviderPool`

Holds the provider registry and exposes model management. Does not run inference directly.

```python
LLMProviderPool(
    config: str | Path | None = None,   # path to providers.yaml
    *,
    timeout: float = 300.0,
    load_timeout: float = 60.0,         # max seconds to wait for model loading (applies to all providers)
    logger: AsyncLLMLogger | None = None,
)
```

#### `get_provider` → `LLMProviderPool`

Returns an `LLMProviderPool` handle for a single provider — host, port, and api key resolved from the registry.

```python
ollama = pool.get_provider("ollama")
```

Use it to configure provider defaults, browse models, or call provider-specific methods directly:

```python
await ollama.check_provider()
ollama.list_downloaded_models()
ollama.configure(num_parallel=2)
```

#### `configure` — provider-level parameter defaults

Sets server-level³ parameter defaults on a provider handle. Inherited by every subsequent `load_model()` call unless overridden per call. Returns `self` for fluent chaining.

```python
ollama = pool.get_provider("ollama")
ollama.configure(num_parallel=2, kv_cache_type="q8_0")

model1 = ollama.load_model("llama3.2:3b")     # inherits num_parallel + kv_cache_type
model2 = ollama.load_model("mistral:7b")      # same — no restart if config unchanged
model3 = ollama.load_model("phi4", num_parallel=4)  # override for this model only
```

Fluent chaining:

```python
model = pool.get_provider("ollama").configure(num_parallel=2).load_model("llama3.2:3b")
```

#### `load_model` → `LocalLLM`

Returns a `LocalLLM` handle bound to a specific provider and model.

```python
model = pool.load_model(
    provider: str,                       # registry key: "ollama", "lm_studio", …
    model: str,                          # model id: "llama3.2:3b", "mistral:7b", …
    *,
    logger: AsyncLLMLogger | None = None,
    context_length: int | None = None,   # shortcut for options={"num_ctx": N}
    timeout: float | None = None,
    load_timeout: float | None = None,   # override server-level load_timeout for this handle
    # --- load-time parameters (see table below) ---
    gpu_layers: int | None = None,
    ngl: int | None = None,              # alias for gpu_layers
    cmoe: bool | None = None,
    cpu_moe: int | None = None,
    flash_attn: bool | None = None,
    batch_size: int | None = None,
    kv_offload: bool | None = None,
    num_experts: int | None = None,
    threads: int | None = None,
    main_gpu: int | None = None,
    gpu_offload: float | str | None = None,  # LM Studio: lms load --gpu <value>
    load_params: dict | None = None,     # arbitrary provider-specific params (escape hatch)
    # --- server-restart level params³ (auto-restarts if value changes; set via configure() to apply once) ---
    num_parallel: int | None = None,
    kv_cache_type: str | None = None,
    gpu_overhead: int | None = None,
    max_loaded_models: int | None = None,
    **defaults,                          # default call kwargs applied to every model.call()
                                         # e.g. temperature=0.2, options={"max_tokens": 512}
)
```

Connection overrides (`host`, `port`, `api_key`, `base_path`, …) are also accepted but rarely needed when using a registry.

#### Load parameters

Named params use **llama.cpp names** as the canonical form. Each provider receives the equivalent field name automatically.

`load_params` is an escape hatch that accepts any provider-specific key/value pairs merged into the load request body. Named params take precedence over `load_params` values on conflict.

| Parameter | Type | llama_cpp | Ollama | LM Studio | Unsloth |
|-----------|------|-----------|--------|-----------|---------|
| `context_length` | `int` | `ctx_size` | `num_ctx` | `context_length` | `max_seq_length` |
| `gpu_layers` / `ngl` | `int` | `gpu_layers` | `num_gpu`¹ | — | — |
| `cmoe` | `bool` | `cmoe` | — | — | — |
| `cpu_moe` | `int` | `cpu_moe` | — | — | — |
| `flash_attn` | `bool` | `flash_attn` | — | `flash_attention` | — |
| `batch_size` | `int` | `batch_size` | `num_batch` | `eval_batch_size` | — |
| `kv_offload` | `bool` | `kv_offload` | — | `offload_kv_cache_to_gpu` | — |
| `num_experts` | `int` | — | — | `num_experts` | — |
| `threads` | `int` | `threads` | `num_thread` | — | — |
| `main_gpu` | `int` | `main_gpu` | `main_gpu` | — | — |
| `gpu_offload` | `float\|str` | — | — | `lms load --gpu`² | — |
| `load_params` | `dict` | merged as-is | merged into options | merged as-is | merged as-is |

¹ Ollama `num_gpu` can also be set via `load_params={"num_gpu": N}` or `options={"num_gpu": N}`.
² `gpu_offload` for LM Studio uses the `lms` CLI (`lms load <model> --gpu <value>`). Accepts `0.0`–`1.0` fraction, `"max"`, or `"off"`. Requires `lms` on `PATH`.

Any other llama-server flag (see `llama_server_manager.available_params()`) can be passed via `load_params`.

#### Server-restart parameters³

These take effect by restarting the provider service. The pool restarts automatically when a value changes; no restart if the config is already current. Set once with `configure()` to avoid repeating per call.

| Parameter | Type | Ollama env var | llama_cpp env var | Notes |
|-----------|------|----------------|-------------------|-------|
| `num_parallel` | `int` | `OLLAMA_NUM_PARALLEL` | `LLAMA_ARG_N_PARALLEL` | concurrent inference streams |
| `flash_attn` | `bool` | `OLLAMA_FLASH_ATTENTION` | — | Ollama: restart; llama_cpp / LM Studio: load-time |
| `kv_cache_type` | `str` | `OLLAMA_KV_CACHE_TYPE` | — | e.g. `"q8_0"`, `"q4_0"` |
| `gpu_overhead` | `int` | `OLLAMA_GPU_OVERHEAD` | — | bytes reserved outside models |
| `max_loaded_models` | `int` | `OLLAMA_MAX_LOADED_MODELS` | — | models kept resident in VRAM |

³ Requires `systemd_service` set in `providers.yaml`.

#### Model management

```python
pool.get_providers()               # ["llama_cpp", "lm_studio", "ollama", "unsloth"]
await pool.check_provider("ollama")
# {"ok": True, "provider": "ollama", "server_url": "http://127.0.0.1:11434", ...}

provider = pool.get_provider("ollama")
provider.list_downloaded_models()  # all models on disk: [{"name": "llama3.2:3b"}, ...]
provider.list_loaded_models()      # models currently in memory
provider.unload_all_models()       # {"unloaded": ["llama3.2:3b"]}

provider = pool.get_provider("lm_studio")
provider.list_loaded_instances()   # {model_key: instance_id}

# Pool-level: pass provider name, or None for all providers at once
pool.list_downloaded_models("ollama")
pool.list_downloaded_models(None)  # {"ollama": [...], "lm_studio": [...], ...}
pool.unload_all_models(None)       # unreachable providers return {"unloaded": [], "error": "..."}
```

---

### `LocalLLM`

Model handle for all inference. Created by `pool.load_model(...)`.

#### `model.call`

```python
result = await model.call(
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
    load_timeout: float | None = None,   # override load_timeout for this call only
)
```

Returns `str` by default, or a `dict` / Pydantic model when `schema_dict` is given.

Defaults set at `load_model` time are merged with per-call kwargs. Per-call values win; `options` dicts are shallow-merged (per-call keys override, others kept).

```python
# temperature=0.2 applied to every call unless overridden
model = pool.load_model("ollama", "llama3.2:3b", temperature=0.2)

result = await model.call(
    messages=[{"role": "user", "content": "Hi"}],
    options={"max_tokens": 100},
)
```

#### `model.batch`

Run multiple calls concurrently with a semaphore.

```python
results = await model.batch(
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

result = await model.call(
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

result = await model.call(
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

result = await model.call(
    messages=[{"role": "user", "content": "Extract: Alice scored 42 points."}],
    schema_dict=schema,
)
# {"name": "Alice", "score": 42}
```

---

### Safety pipelines

Every `model.call` runs two automatic guard mechanisms after the model produces a response, before it is returned to the caller.

#### Loop guard

Detects when the model has entered a repetition loop (copy-paste repetition, stuttering, n-gram cycling, ping-pong exchange, or a monotone run of identical tokens). On detection the guard appends a corrective prompt and retries the generation, up to `max_loop_retries` times.

Triggered automatically — no configuration needed. Tune sensitivity with environment variables:

| Variable | Default | Effect |
|---|---|---|
| `LOOP_GUARD_DISABLED` | `0` | Set `1` to turn off the guard entirely |
| `LOOP_GUARD_MIN_LEN` | `60` | Minimum response length before checks run |
| `LOOP_GUARD_REPEAT_RATIO` | `0.35` | Threshold for copy-paste ratio detector |
| `LOOP_GUARD_NGRAM_SIZE` | `6` | N-gram size for cycling detector |
| `LOOP_GUARD_NGRAM_RATIO` | `0.4` | Threshold for n-gram cycling ratio |
| `LOOP_GUARD_MONO_RUN` | `40` | Minimum identical-token run to flag |

#### JSON fix

When `schema_dict` is given, validates the model's output against the schema. On failure it:
1. Attempts lenient repairs (strips markdown fences, extracts embedded JSON, fixes truncation).
2. If repair fails, appends the validation error to the messages and asks the model to correct its output.
3. Retries up to `max_json_fix_retries` times before raising `ValueError`.

Both guards share the same retry loop, so a response that triggers both will exhaust retries independently.

---

### Logging

`AsyncLLMLogger` writes a structured stream log of every request/response pair, including thinking blocks, tool execution results, and timing.

```python
from unified_local_llm_server.llm_logger import AsyncLLMLogger

logger = AsyncLLMLogger("logs/session.log")
model    = pool.load_model("ollama", "llama3.2:3b", logger=logger)

await model.call(...)

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

#### `restart`

Recovery action — restarts a frozen or crashed provider service. Requires `systemd_service` in `providers.yaml`.

```python
provider = pool.get_provider("ollama")
provider.restart()
# {"service": "ollama.service", "method": "user", "returncode": 0, "waited_s": 3.1}

pool.restart("llama_cpp", wait_timeout=60.0)
```

Tries `systemctl --user restart` → `systemctl restart` (polkit) → `sudo -n systemctl restart` in order. Polls the health endpoint until reachable before returning.

Server-level parameters³ (`num_parallel`, `kv_cache_type`, …) are set via `configure()` or `load_model()` — the pool restarts automatically when a value changes. Pass `env=` to inject raw env vars on a manual restart.

---

### Unsloth — automatic model loading

Unsloth models are loaded on demand before the first inference call. Pass the `repo_id@quant` shorthand (as returned by `list_downloaded_models`) — it is resolved to the full `.gguf` path via the Unsloth API automatically:

```python
# shorthand — resolved to the actual .gguf path
model = pool.load_model("unsloth", "unsloth/gpt-oss-20b-GGUF@UD-Q4_K_XL")

# or explicit path
model = pool.load_model("unsloth", "/mnt/models/gpt-oss-20b-UD-Q4_K_XL.gguf")
```

If the model is already loaded on the server, the load step is skipped.

---

## Environment variables

| Variable | Purpose |
|----------|---------|
| `LOCAL_LLM_PROVIDERS` | Path to providers config file (fallback when no path given to `LLMProviderPool`) |
| `UNSLOTH_KEY` / `UNSLOTH_API_KEY` | Unsloth API key |
| `LOG_MSG_REFS` | Set to `1` to deduplicate repeated messages in logs by SHA1 reference |
