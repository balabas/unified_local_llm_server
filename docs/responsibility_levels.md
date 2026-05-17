# Responsibility Levels

The package keeps provider endpoints, model selection, inference transport, and
repair pipelines separate.

## Provider Config

Files:

- `unified_local_llm_server/providers.py`
- `unified_local_llm_server/provider_registry.py`
- `providers.example.json`

Owns only real provider endpoint details: provider kind, host, port, API token,
OpenAI-compatible base path, health path, and model-list path.

Does not own model names.

## Model Selection

File:

- `unified_local_llm_server/server.py`

`LocalLLMServer.model` is runtime call state. A caller can set it directly, or
the server can resolve the first id from the provider's configured models path.

For providers with multiple loaded models, pass `model=` to `LocalLLMServer.call()`.
That selects a model for one call without changing provider config or requiring
a separate server instance.

Preferred experiment API:

```python
server = LocalLLMServer()
llm1 = server.load_model("ollama", "llama3.2:3b", temperature=0.2, context_length=8192)
llm2 = server.load_model("lm_studio", "qwen3-4b", temperature=0.1)

resp1 = await llm1.call(messages=[{"role": "user", "content": "..."}])
resp2 = await llm2.call(messages=[{"role": "user", "content": "..."}])
```

`LocalLLM` handles carry model-specific inference defaults. Provider config
still only describes the endpoint.

Batch is also a `LocalLLM` responsibility because batch experiments normally
target one loaded model handle with fixed defaults:

```python
results = await llm1.batch(
    [
        [{"role": "user", "content": "Prompt one"}],
        {"messages": [{"role": "user", "content": "Prompt two"}], "temperature": 0.1},
    ],
    concurrency=4,
    return_exceptions=False,
)
```

`LocalLLMServer.batch()` is intentionally not the primary API. Mixed-provider
batches belong in a separate experiment-runner layer.

## OpenAI-Compatible Client

Files:

- `unified_local_llm_server/client.py`
- `unified_local_llm_server/transport.py`

Owns HTTP request/response adaptation for `/chat/completions` and `/models`.
It does not know about retry policy, loop handling, schema repair, or provider
lifecycle.

## Task Pipeline

File:

- `unified_local_llm_server/pipelines/task.py`

Owns general call preparation: message copying, `think`, `effort`,
temperature, and token option mapping.

Does not own structured-output schema conversion or JSON repair.

## JSON Fix Pipeline

File:

- `unified_local_llm_server/pipelines/json_fix.py`

Owns structured-output preparation and repair: schema conversion through
`dict_to_pydantic_schema`, schema prompt insertion, JSON extraction, best-effort
repair, Pydantic validation, and retry messages after invalid JSON.

`max_json_fix_retries` applies to this JSON-fix retry path only when `schema_dict` is
provided.

## Loop Guard Pipeline

File:

- `unified_local_llm_server/pipelines/loop_guard.py`

Owns repetition detection and retry messages after loop detection.

`max_loop_retries` applies to this layer and defaults to `5`. Plain unstructured
calls do not use `max_json_fix_retries`.

## Tool/MCP Pipeline

File:

- `unified_local_llm_server/pipelines/tool.py`

Owns tool-call normalization, tool execution, and construction of assistant/tool
messages for the next LLM round. It accepts normal OpenAI-style tool schemas
and a `tool_registry` mapping tool names to sync or async Python handlers.

MCP is not implemented here yet. The intended boundary is: an MCP adapter should
populate or proxy `tool_registry` handlers. MCP connection/session details
should not enter provider config, the OpenAI-compatible client, task
preparation, JSON repair, or loop guard code.

## Orchestrator

File:

- `unified_local_llm_server/server.py`

`LocalLLMServer.call()` composes the layers in order. It should delegate work to
the appropriate layer instead of implementing provider config, schema repair,
or loop detection inline.

## Logging

File:

- `unified_local_llm_server/llm_logger.py`

Owns the readable stream-style log format. It does not make retry decisions or
parse model output.
