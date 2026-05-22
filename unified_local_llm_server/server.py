from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote
from typing import Any

from .client import LocalAsyncOpenAI
from .llm_logger import AsyncLLMLogger
from .pipelines.json_fix import JsonFixPipeline
from .pipelines.loop_guard import LoopGuardPipeline
from .pipelines.tool import ToolHandler, ToolPipeline
from .provider_registry import ProviderRegistry, provider_entry_to_server_kwargs
from .providers import ProviderConfig, ProviderKind, known_provider_kinds, resolve_provider_config
from .transport import OpenAICompatibleTransport, TransportError



@dataclass(slots=True)
class LocalLLM:
    server: "LLMProviderPool"
    provider: str
    model: str
    context_length: int | None = None
    gpu_layers: int | None = None
    cmoe: bool | None = None
    cpu_moe: int | None = None
    flash_attn: bool | None = None
    batch_size: int | None = None
    kv_offload: bool | None = None
    num_experts: int | None = None
    threads: int | None = None
    main_gpu: int | None = None
    gpu_offload: float | str | None = None
    load_params: dict[str, Any] | None = None
    load_timeout: float | None = None
    # server-restart level params (applied before model load)
    num_parallel: int | None = None
    kv_cache_type: str | None = None
    gpu_overhead: int | None = None
    max_loaded_models: int | None = None
    defaults: dict[str, Any] = field(default_factory=dict)

    @property
    def server_url(self) -> str:
        return self.server.server_url

    @property
    def base_url(self) -> str:
        return self.server.base_url

    async def call(self, *, return_usage: bool = False, **kwargs: Any) -> Any:
        """Run a single inference call.

        Parameters
        ----------
        messages : list[dict]
            Required. Conversation history as ``[{"role": ..., "content": ...}, ...]``.
        return_usage : bool
            If True, return ``(result, {"prompt_tokens": int, "completion_tokens": int})``
            instead of just the result.  Default ``False``.
        model : str, optional
            Override the model name bound to this handle for this call only.
        temperature : float, optional
            Sampling temperature.  ``None`` uses the provider default.
        think : bool
            Allow the model to emit hidden chain-of-thought / reasoning tokens.
            Default ``False`` (injects a "final answer only" system prompt).
        effort : str, optional
            Reasoning-effort hint for reasoning models, e.g. ``"high"`` or ``"low"``.
            Injected as a leading system message.
        schema_dict : dict, optional
            JSON Schema describing the required response structure.  When set,
            the response text is parsed and returned as a Python object.
            Retries up to ``max_json_fix_retries`` times on parse failure.
        options : dict, optional
            Provider-specific generation options.

            *Ollama*: ``num_ctx``, ``num_predict``, ``num_thread``, ``mirostat``,
            ``mirostat_eta``, ``mirostat_tau``, ``repeat_last_n``,
            ``repeat_penalty``, ``tfs_z``, ``typical_p``.

            *OpenAI-compatible*: ``top_p``, ``seed``, ``stop``,
            ``presence_penalty``, ``frequency_penalty``, ``reasoning_effort``.

            The keys ``max_tokens``, ``max_completion_tokens``, and ``num_predict``
            are all accepted as aliases for the generation-token limit.
        context_length : int, optional
            Context-window size in tokens.  Triggers model (re-)loading at the
            requested size for llama_cpp, LM Studio, Unsloth, and Ollama.
            Propagated as ``options["num_ctx"]`` for Ollama.
        gpu_layers : int, optional
            Number of transformer layers to offload to GPU (llama_cpp only).
        tools : list[dict], optional
            OpenAI-format tool definitions::

                [{"type": "function",
                  "function": {"name": "...", "parameters": {...}}}]

            Requires ``tool_registry`` to execute calls returned by the model.
        tool_registry : dict[str, callable], optional
            Mapping of tool name → async callable (or ``ToolHandler``).
            Invoked automatically for each tool call the model emits.
        max_tool_rounds : int
            Maximum tool-call / tool-result round trips before raising
            ``RuntimeError``.  Default ``5``.
        max_json_fix_retries : int
            Maximum retries on structured-output parse failure (``schema_dict``
            mode).  Default ``3``.
        max_loop_retries : int
            Maximum retries when a repetition loop is detected in the output.
            Default ``5``.

        Returns
        -------
        str | object | tuple
            Plain-text response string, or the parsed object when ``schema_dict``
            is set.  With ``return_usage=True``, returns a two-element tuple
            ``(result, usage_dict)``.

        Examples
        --------
        Basic call::

            result = await llm.call(messages=[{"role": "user", "content": "Hello"}])

        With usage stats::

            text, usage = await llm.call(
                messages=[{"role": "user", "content": "Hi"}],
                return_usage=True,
            )
            print(usage["completion_tokens"])

        Structured output::

            schema = {"type": "object", "properties": {"name": {"type": "string"}},
                      "required": ["name"]}
            obj = await llm.call(messages=[...], schema_dict=schema)

        Ollama with explicit context window::

            result = await llm.call(
                messages=[...],
                options={"num_ctx": 32768, "num_predict": 512},
            )
        """
        merged = self._merge_defaults(kwargs)
        merged["model"] = self.model
        if self.context_length is not None:
            merged.setdefault("context_length", self.context_length)
        if self.gpu_layers is not None:
            merged.setdefault("gpu_layers", self.gpu_layers)
        if self.cmoe is not None:
            merged.setdefault("cmoe", self.cmoe)
        if self.cpu_moe is not None:
            merged.setdefault("cpu_moe", self.cpu_moe)
        if self.flash_attn is not None:
            merged.setdefault("flash_attn", self.flash_attn)
        if self.batch_size is not None:
            merged.setdefault("batch_size", self.batch_size)
        if self.kv_offload is not None:
            merged.setdefault("kv_offload", self.kv_offload)
        if self.num_experts is not None:
            merged.setdefault("num_experts", self.num_experts)
        if self.threads is not None:
            merged.setdefault("threads", self.threads)
        if self.main_gpu is not None:
            merged.setdefault("main_gpu", self.main_gpu)
        if self.gpu_offload is not None:
            merged.setdefault("gpu_offload", self.gpu_offload)
        if self.load_params is not None:
            merged.setdefault("load_params", self.load_params)
        if self.load_timeout is not None:
            merged.setdefault("load_timeout", self.load_timeout)
        if self.num_parallel is not None:
            merged.setdefault("num_parallel", self.num_parallel)
        if self.kv_cache_type is not None:
            merged.setdefault("kv_cache_type", self.kv_cache_type)
        if self.gpu_overhead is not None:
            merged.setdefault("gpu_overhead", self.gpu_overhead)
        if self.max_loaded_models is not None:
            merged.setdefault("max_loaded_models", self.max_loaded_models)
        return await self.server._call(return_usage=return_usage, **merged)

    async def batch(
        self,
        items: list[list[dict[str, Any]] | dict[str, Any]],
        *,
        concurrency: int = 4,
        return_exceptions: bool = False,
        **shared_call_kwargs: Any,
    ) -> list[Any]:
        """Run multiple inference calls concurrently.

        Each item in ``items`` is either a bare message list (shorthand) or a
        full call-kwargs dict that may override any shared kwarg for that item.
        Options dicts are deep-merged: item-level keys win over shared keys.

        Parameters
        ----------
        items : list
            Each element is one of:

            * ``list[dict]`` — a message list, equivalent to
              ``{"messages": [...]}``
            * ``dict`` — a full kwargs dict; **must** contain a ``"messages"``
              key.  Any key overrides the corresponding ``shared_call_kwargs``
              value for that item only.
        concurrency : int
            Maximum number of calls executed in parallel.  Default ``4``.
        return_exceptions : bool
            When ``True``, exceptions are captured and returned in-place
            (same semantics as ``asyncio.gather(return_exceptions=True)``).
            When ``False`` (default), the first exception propagates and
            cancels remaining tasks.
        **shared_call_kwargs :
            Any keyword argument accepted by :meth:`call` (e.g. ``temperature``,
            ``schema_dict``, ``options``, ``return_usage``).  Applied to every
            item unless the item dict overrides it.

        Returns
        -------
        list
            Results in the same order as ``items``.  Each element matches what
            :meth:`call` would return for that item (plain text, parsed object,
            or ``(result, usage)`` tuple when ``return_usage=True``).

        Examples
        --------
        Batch of bare message lists::

            results = await llm.batch(
                [
                    [{"role": "user", "content": "What is 2+2?"}],
                    [{"role": "user", "content": "Capital of France?"}],
                ],
                temperature=0.0,
            )

        Per-item overrides::

            results = await llm.batch(
                [
                    {"messages": [...], "temperature": 0.2},
                    {"messages": [...], "temperature": 0.8},
                ],
                concurrency=2,
            )

        Collect errors instead of raising::

            results = await llm.batch(items, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    print("error:", r)
        """
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")

        semaphore = asyncio.Semaphore(concurrency)
        call_kwargs = [
            self._build_batch_call_kwargs(item, shared_call_kwargs)
            for item in items
        ]

        async def run_one(kwargs: dict[str, Any]) -> Any:
            async with semaphore:
                return await self.call(**kwargs)

        return await asyncio.gather(
            *(run_one(kwargs) for kwargs in call_kwargs),
            return_exceptions=return_exceptions,
        )

    def _merge_defaults(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        merged = dict(self.defaults)
        default_options = merged.get("options")
        call_options = kwargs.get("options")
        merged.update(kwargs)
        if isinstance(default_options, dict) or isinstance(call_options, dict):
            options: dict[str, Any] = {}
            if isinstance(default_options, dict):
                options.update(default_options)
            if isinstance(call_options, dict):
                options.update(call_options)
            merged["options"] = options
        return merged

    def _build_batch_call_kwargs(
        self,
        item: list[dict[str, Any]] | dict[str, Any],
        shared_call_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        if isinstance(item, list):
            return self._merge_call_kwargs(shared_call_kwargs, {"messages": item})
        if isinstance(item, dict):
            if "messages" not in item:
                raise ValueError("batch item dict must include 'messages'")
            return self._merge_call_kwargs(shared_call_kwargs, item)
        raise TypeError("batch items must be message lists or call kwargs dicts")

    def _merge_call_kwargs(
        self,
        base: dict[str, Any],
        override: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(base)
        base_options = merged.get("options")
        override_options = override.get("options")
        merged.update(override)
        if isinstance(base_options, dict) or isinstance(override_options, dict):
            options: dict[str, Any] = {}
            if isinstance(base_options, dict):
                options.update(base_options)
            if isinstance(override_options, dict):
                options.update(override_options)
            merged["options"] = options
        return merged


_SELF_PROVIDER = object()  # sentinel: "use self.provider"


class LLMProviderPool:
    def __init__(
        self,
        config: str | Path | None = None,
        *,
        provider: str = ProviderKind.OLLAMA,
        model: str | None = None,
        host: str | None = None,
        port: int | None = None,
        api_key: str | None = None,
        base_path: str | None = None,
        health_path: str | None = None,
        models_path: str | None = None,
        timeout: float = 300.0,
        load_timeout: float = 60.0,
        logger: AsyncLLMLogger | None = None,
        transport: OpenAICompatibleTransport | None = None,
        provider_transport: OpenAICompatibleTransport | None = None,
        provider_registry: ProviderRegistry | None = None,
    ):
        self.provider_registry = provider_registry or ProviderRegistry.load(config)
        self._transport_override = transport
        self._provider_transport_override = provider_transport
        self.config: ProviderConfig = resolve_provider_config(
            provider,
            host=host,
            port=port,
            api_key=api_key,
            base_path=base_path,
            health_path=health_path,
            models_path=models_path,
        )
        self._model = model
        self.timeout = timeout
        self.load_timeout = load_timeout
        self.logger = logger
        self._transport = transport or OpenAICompatibleTransport(
            self.config.base_url,
            timeout=timeout,
            api_key=self.config.api_key,
        )
        self._provider_transport = provider_transport or transport or OpenAICompatibleTransport(
            self.config.server_url,
            timeout=timeout,
            api_key=self.config.api_key,
        )
        self._client = LocalAsyncOpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=timeout,
            transport=self._transport,
        )

        self._loop_guard = LoopGuardPipeline()
        self._json_fix = JsonFixPipeline()
        self._tool_pipeline = ToolPipeline()
        self._active_server_config: dict[str, Any] = {}
        self._server_defaults: dict[str, Any] = {}

    def load_model(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        host: str | None = None,
        port: int | None = None,
        api_key: str | None = None,
        base_path: str | None = None,
        health_path: str | None = None,
        models_path: str | None = None,
        timeout: float | None = None,
        load_timeout: float | None = None,
        logger: AsyncLLMLogger | None = None,
        transport: OpenAICompatibleTransport | None = None,
        provider_transport: OpenAICompatibleTransport | None = None,
        context_length: int | None = None,
        gpu_layers: int | None = None,
        ngl: int | None = None,
        cmoe: bool | None = None,
        cpu_moe: int | None = None,
        flash_attn: bool | None = None,
        batch_size: int | None = None,
        kv_offload: bool | None = None,
        num_experts: int | None = None,
        threads: int | None = None,
        main_gpu: int | None = None,
        gpu_offload: float | str | None = None,
        load_params: dict[str, Any] | None = None,
        num_parallel: int | None = None,
        kv_cache_type: str | None = None,
        gpu_overhead: int | None = None,
        max_loaded_models: int | None = None,
        **defaults: Any,
    ) -> LocalLLM:
        resolved_provider = provider or self.provider
        resolved_model = model or (self.model if resolved_provider == self.provider else None)
        if not resolved_model:
            raise ValueError("model is required when loading a model handle")

        # Inherit server-level defaults set via configure() unless overridden per call
        d = self._server_defaults
        if num_parallel is None:      num_parallel      = d.get("num_parallel")
        if kv_cache_type is None:     kv_cache_type     = d.get("kv_cache_type")
        if gpu_overhead is None:      gpu_overhead      = d.get("gpu_overhead")
        if max_loaded_models is None: max_loaded_models = d.get("max_loaded_models")
        if flash_attn is None:        flash_attn        = d.get("flash_attn")

        if context_length is not None:
            options = dict(defaults.get("options") or {})
            options.setdefault("num_ctx", context_length)
            defaults["options"] = options


        has_connection_overrides = any(v is not None for v in (
            host, port, api_key, base_path, health_path, models_path, timeout, load_timeout, transport, provider_transport
        ))
        use_self = resolved_provider == self.provider and not has_connection_overrides
        if use_self:
            target_server = self
        elif has_connection_overrides:
            target_server = LLMProviderPool(
                provider=resolved_provider,
                model=resolved_model,
                host=host,
                port=port,
                api_key=api_key,
                base_path=base_path,
                health_path=health_path,
                models_path=models_path,
                timeout=timeout if timeout is not None else self.timeout,
                load_timeout=load_timeout if load_timeout is not None else self.load_timeout,
                transport=transport,
                provider_transport=provider_transport,
                provider_registry=self.provider_registry,
            )
        else:
            target_server = self.get_provider(resolved_provider)
        if logger is not None:
            target_server.logger = logger
        resolved_gpu_layers = gpu_layers if gpu_layers is not None else ngl
        return LocalLLM(
            server=target_server,
            provider=target_server.provider,
            model=resolved_model,
            context_length=context_length,
            gpu_layers=resolved_gpu_layers,
            cmoe=cmoe,
            cpu_moe=cpu_moe,
            flash_attn=flash_attn,
            batch_size=batch_size,
            kv_offload=kv_offload,
            num_experts=num_experts,
            threads=threads,
            main_gpu=main_gpu,
            gpu_offload=gpu_offload,
            load_params=load_params,
            load_timeout=load_timeout,
            num_parallel=num_parallel,
            kv_cache_type=kv_cache_type,
            gpu_overhead=gpu_overhead,
            max_loaded_models=max_loaded_models,
            defaults=defaults,
        )

    def configure(self, **server_params: Any) -> "LLMProviderPool":
        """Set server-level parameter defaults for this provider handle.

        Stored defaults are inherited by every subsequent ``load_model()`` call
        unless explicitly overridden.  Returns ``self`` for fluent chaining::

            ollama = pool.get_provider("ollama")
            ollama.configure(num_parallel=2, kv_cache_type="q8_0")
            llm = ollama.load_model("llama3.2:3b")
        """
        self._server_defaults.update(server_params)
        return self

    @property
    def provider(self) -> str:
        return self.config.name

    @property
    def model(self) -> str | None:
        return self._model

    @model.setter
    def model(self, value: str | None) -> None:
        self._model = value

    @property
    def base_url(self) -> str:
        return self.config.base_url

    @property
    def server_url(self) -> str:
        return self.config.server_url

    @property
    def async_client(self) -> LocalAsyncOpenAI:
        return self._client

    def openai_async_client(self) -> LocalAsyncOpenAI:
        try:
            from openai import AsyncOpenAI  # type: ignore
        except Exception:
            return self._client
        return AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.config.api_key or "local",
        )

    def _resolve_unsloth_model_path(self, model_name: str) -> str:
        """Resolve 'repo_id@quant' shorthand to the full local .gguf file path."""
        if "@" not in model_name or model_name.startswith("/"):
            return model_name
        repo_id, quant = model_name.rsplit("@", 1)
        cached = self._provider_transport.get_json_sync("/api/models/cached-gguf")
        cache_path = next(
            (m["cache_path"] for m in (cached.get("cached") or []) if m.get("repo_id") == repo_id),
            None,
        )
        if not cache_path:
            raise RuntimeError(f"Unsloth: repo '{repo_id}' not found in cached-gguf")
        variants = self._provider_transport.get_json_sync(
            f"/api/models/gguf-variants?repo_id={quote(repo_id, safe='')}"
        )
        filename = next(
            (v["filename"] for v in (variants.get("variants") or []) if v.get("quant") == quant),
            None,
        )
        if not filename:
            raise RuntimeError(f"Unsloth: quant '{quant}' not found for repo '{repo_id}'")
        matches = list(Path(cache_path).glob(f"snapshots/*/{filename}"))
        if not matches:
            raise RuntimeError(f"Unsloth: file '{filename}' not found under '{cache_path}/snapshots'")
        return str(matches[0])

    def _stream_chat_sync(
        self,
        path: str,
        body: dict[str, Any],
        *,
        ollama_native: bool = False,
        _logger: "AsyncLLMLogger | None" = None,
    ) -> Any:
        """Stream a chat request and return an assembled response object.

        Works for both OpenAI-SSE format (all /v1/chat/completions providers) and
        Ollama-native format (/api/chat, ollama_native=True).
        """
        import json as _json
        import types

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls_acc: dict[int, dict[str, Any]] = {}
        prompt_tok = 0
        completion_tok = 0

        def _fire_log(text: str, is_thinking: bool) -> None:
            if _logger is not None and text:
                with _logger._lock:
                    _logger._write_output_chunk(text, is_thinking=is_thinking)
                    _logger._f.flush()

        for line in self._provider_transport.stream_lines_sync(path, body):
            if not line:
                continue

            if ollama_native:
                # Ollama /api/chat streams newline-delimited JSON objects (not SSE)
                try:
                    chunk = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                msg = chunk.get("message") or {}
                delta_thinking = msg.get("thinking") or ""
                delta_content = msg.get("content") or ""
                if delta_thinking:
                    reasoning_parts.append(delta_thinking)
                    _fire_log(delta_thinking, True)
                if delta_content:
                    content_parts.append(delta_content)
                    _fire_log(delta_content, False)
                if chunk.get("done"):
                    prompt_tok = chunk.get("prompt_eval_count", 0)
                    completion_tok = chunk.get("eval_count", 0)
            else:
                # OpenAI SSE: "data: {...}" lines
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if raw == "[DONE]":
                    break
                try:
                    chunk = _json.loads(raw)
                except _json.JSONDecodeError:
                    continue
                u = chunk.get("usage") or {}
                if u.get("prompt_tokens"):
                    prompt_tok = u["prompt_tokens"]
                if u.get("completion_tokens"):
                    completion_tok = u["completion_tokens"]
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        content_parts.append(delta["content"])
                        _fire_log(delta["content"], False)
                    for key in ("reasoning_content", "reasoning"):
                        if delta.get(key):
                            reasoning_parts.append(delta[key])
                            _fire_log(delta[key], True)
                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                        if tc.get("id"):
                            tool_calls_acc[idx]["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            tool_calls_acc[idx]["name"] = fn["name"]
                        if fn.get("arguments"):
                            tool_calls_acc[idx]["arguments"] += fn["arguments"]

        content = "".join(content_parts)
        reasoning = "".join(reasoning_parts) or None

        # Normalize: some models/providers embed thinking as <think>...</think> at the
        # start of content instead of a dedicated field. Extract it so callers always
        # get a clean (reasoning, content) split regardless of which method was used.
        if reasoning is None and "<think>" in content:
            import re as _re
            m = _re.match(r"<think>(.*?)</think>\s*", content, _re.DOTALL)
            if m:
                reasoning = m.group(1)
                content = content[m.end():]

        tool_calls = None
        if tool_calls_acc:
            tool_calls = [
                types.SimpleNamespace(
                    id=tc["id"],
                    type="function",
                    function=types.SimpleNamespace(name=tc["name"], arguments=tc["arguments"]),
                )
                for tc in tool_calls_acc.values()
            ]

        usage_ns = types.SimpleNamespace(prompt_tokens=prompt_tok, completion_tokens=completion_tok)
        msg = types.SimpleNamespace(tool_calls=tool_calls)
        choice_ns = types.SimpleNamespace(message=msg)
        return types.SimpleNamespace(content=content, reasoning=reasoning, choices=[choice_ns], usage=usage_ns)

    def _ollama_preload_sync(self, model: str, native_opts: dict[str, Any], *, load_timeout: float | None = None) -> None:
        """Warm the model at the requested context size before inference."""
        preload: dict[str, Any] = {"model": model, "prompt": "", "stream": False, "keep_alive": -1}
        if native_opts:
            preload["options"] = dict(native_opts)
        self._provider_transport.post_json_sync("/api/generate", preload, timeout=load_timeout)

    def _lms_load_via_cli(
        self,
        model_name: str,
        *,
        gpu_offload: float | str,
        context_length: int | None = None,
        load_timeout: float | None = None,
    ) -> None:
        import shutil
        import subprocess
        lms = shutil.which("lms")
        if not lms:
            raise RuntimeError("lms CLI not found on PATH — install LM Studio CLI")
        cmd = [lms, "load", model_name, "--gpu", str(gpu_offload)]
        if context_length is not None:
            cmd += ["--context-length", str(context_length)]
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=load_timeout or 300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"lms load failed (exit {result.returncode}): {result.stderr.strip()}")

    def _ensure_lm_studio_loaded(
        self,
        model_name: str,
        context_length: int | None = None,
        *,
        gpu_offload: float | str | None = None,
        flash_attn: bool | None = None,
        batch_size: int | None = None,
        kv_offload: bool | None = None,
        num_experts: int | None = None,
        load_params: dict[str, Any] | None = None,
        load_timeout: float | None = None,
    ) -> None:
        # gpu_offload requires the CLI path (REST API has no gpu field)
        if gpu_offload is not None:
            self._lms_load_via_cli(
                model_name,
                gpu_offload=gpu_offload,
                context_length=context_length,
                load_timeout=load_timeout,
            )
            return

        has_load_params = any(v is not None for v in (
            context_length, flash_attn, batch_size, kv_offload, num_experts,
        )) or bool(load_params)
        if not has_load_params:
            return
        # Skip if already loaded (LM Studio returns 500 if you reload an active instance)
        info = self._provider_transport.get_json_sync("/api/v1/models")
        for m in (info or {}).get("models") or []:
            if m.get("key") == model_name and m.get("loaded_instances"):
                return
        body: dict[str, Any] = {"model": model_name}
        if load_params:
            body.update(load_params)
        # named params use LM Studio field names; override anything in load_params
        if context_length is not None:
            body["context_length"] = context_length
        if flash_attn is not None:
            body["flash_attention"] = flash_attn
        if batch_size is not None:
            body["eval_batch_size"] = batch_size
        if kv_offload is not None:
            body["offload_kv_cache_to_gpu"] = kv_offload
        if num_experts is not None:
            body["num_experts"] = num_experts
        r = self._provider_transport.post_json_sync("/api/v1/models/load", body, timeout=load_timeout)
        if r.status >= 400:
            raise RuntimeError(f"LM Studio /api/v1/models/load failed (HTTP {r.status}): {r.text!r}")

    def _ensure_llama_cpp_loaded(
        self,
        model_name: str,
        context_length: int | None = None,
        gpu_layers: int | None = None,
        *,
        cmoe: bool | None = None,
        cpu_moe: int | None = None,
        flash_attn: bool | None = None,
        batch_size: int | None = None,
        kv_offload: bool | None = None,
        threads: int | None = None,
        main_gpu: int | None = None,
        load_params: dict[str, Any] | None = None,
        load_timeout: float | None = None,
    ) -> None:
        r = self._provider_transport._do_request("GET", "/models/loaded", None)
        if r.status >= 400:
            raise RuntimeError(f"llama_cpp /models/loaded failed (HTTP {r.status}): {r.text!r}")
        loaded = r.json() or []
        needs_load = model_name not in loaded or bool(load_params) or any(
            v is not None for v in (
                context_length, gpu_layers, cmoe, cpu_moe,
                flash_attn, batch_size, kv_offload, threads, main_gpu,
            )
        )
        if not needs_load:
            return
        body: dict[str, Any] = {"model": model_name}
        if load_params:
            body.update(load_params)
        # named params use llama.cpp field names; override anything in load_params
        if context_length is not None:
            body["ctx_size"] = context_length
        if gpu_layers is not None:
            body["gpu_layers"] = gpu_layers
        if cmoe is not None:
            body["cmoe"] = cmoe
        if cpu_moe is not None:
            body["cpu_moe"] = cpu_moe
        if flash_attn is not None:
            body["flash_attn"] = flash_attn
        if batch_size is not None:
            body["batch_size"] = batch_size
        if kv_offload is not None:
            body["kv_offload"] = kv_offload
        if threads is not None:
            body["threads"] = threads
        if main_gpu is not None:
            body["main_gpu"] = main_gpu
        if model_name in loaded:
            body["force_restart"] = True
        r2 = self._provider_transport.post_json_sync("/models/load", body, timeout=load_timeout)
        if r2.status >= 400:
            raise RuntimeError(f"llama_cpp /models/load failed (HTTP {r2.status}): {r2.text!r}")

    def _ensure_unsloth_loaded(self, model_path: str, context_length: int | None = None, *, load_params: dict[str, Any] | None = None, load_timeout: float | None = None) -> None:
        resolved = self._resolve_unsloth_model_path(model_path)
        r = self._provider_transport._do_request("GET", "/api/inference/status", None)
        if r.status >= 400:
            raise RuntimeError(f"Unsloth /api/inference/status failed (HTTP {r.status}): {r.text!r}")
        status = r.json()
        already_loaded = status and status.get("active_model") == resolved
        ctx_matches = (
            context_length is None
            or (status and status.get("context_length") == context_length)
        )
        if already_loaded and ctx_matches and not load_params:
            return
        body: dict[str, Any] = {"model_path": resolved}
        if load_params:
            body.update(load_params)
        if context_length is not None:
            body["max_seq_length"] = context_length
        r2 = self._provider_transport.post_json_sync("/api/inference/load", body, timeout=load_timeout)
        if r2.status >= 400:
            raise RuntimeError(f"Unsloth /api/inference/load failed (HTTP {r2.status}): {r2.text!r}")

    def list_loaded_models(self, provider=_SELF_PROVIDER) -> list[str] | dict[str, list[str]]:
        if provider is _SELF_PROVIDER:
            provider = self.provider
        if provider is None:
            return {p: self.list_loaded_models(p) for p in self.get_providers()}
        srv = self.get_provider(provider)
        if srv.config.name == ProviderKind.OLLAMA:
            data = srv._provider_transport.get_json_sync("/api/ps")
            return [
                m.get("name") or m.get("model")
                for m in (data.get("models") or [])
                if m.get("name") or m.get("model")
            ]
        if srv.config.name == ProviderKind.UNSLOTH:
            data = srv._provider_transport.get_json_sync("/api/inference/status")
            return list(data.get("loaded") or [])
        if srv.config.name == ProviderKind.LM_STUDIO:
            return list(self.list_loaded_instances(provider))
        data = srv._provider_transport.get_json_sync(srv.config.models_path)
        if isinstance(data, dict):
            return [m["id"] for m in (data.get("data") or []) if m.get("id")]
        return []

    def restart(
        self,
        provider=_SELF_PROVIDER,
        *,
        wait: bool = True,
        wait_timeout: float = 30.0,
        poll_interval: float = 1.0,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Restart the systemd service backing a provider.

        Can be called on a per-provider handle without arguments::

            pool.get_provider("ollama").restart()

        Or on the pool with an explicit name::

            pool.restart("ollama")

        Tries ``systemctl --user restart``, then ``systemctl restart`` (polkit),
        then ``sudo -n systemctl restart`` in order.  Requires ``systemd_service``
        to be set in ``providers.yaml``.

        Parameters
        ----------
        provider : str, optional
            Registry name.  Defaults to this instance's own provider.
        wait : bool
            Poll the health endpoint until reachable (default True).
        wait_timeout : float
            Seconds before giving up (default 30).
        poll_interval : float
            Seconds between health checks (default 1).
        env : dict[str, str], optional
            Env vars injected via a temporary systemd drop-in, removed once
            the provider is reachable so the next restart uses defaults.
            Normally set automatically by ``load_model`` when server-level
            params (``num_parallel``, ``kv_cache_type``, …) change.
        """
        import subprocess
        import time

        if provider is _SELF_PROVIDER:
            provider = self.provider

        effective_env = env or None

        try:
            entry = self.provider_registry.get(provider)
            service = entry.systemd_service
        except KeyError:
            service = None
        if not service:
            raise ValueError(
                f"No systemd_service configured for provider {provider!r}. "
                "Add 'systemd_service: <name>.service' to providers.yaml."
            )

        dropin_user: bool | None = None
        if effective_env:
            path_user = Path.home() / ".config" / "systemd" / "user" / f"{service}.d" / "llm_overrides.conf"
            path_sys  = Path("/etc/systemd/system") / f"{service}.d" / "llm_overrides.conf"
            for is_user, dropin_path in ((True, path_user), (False, path_sys)):
                try:
                    dropin_path.parent.mkdir(parents=True, exist_ok=True)
                    dropin_path.write_text(
                        "[Service]\n" + "".join(f"Environment={k}={v}\n" for k, v in effective_env.items())
                    )
                    reload = ["systemctl", "--user", "daemon-reload"] if is_user else ["systemctl", "daemon-reload"]
                    subprocess.run(reload, capture_output=True)
                    dropin_user = is_user
                    break
                except PermissionError:
                    continue

        def _remove_dropin():
            if dropin_user is None:
                return
            p = (Path.home() / ".config" / "systemd" / "user" / f"{service}.d" / "llm_overrides.conf"
                 if dropin_user else Path("/etc/systemd/system") / f"{service}.d" / "llm_overrides.conf")
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
            reload = ["systemctl", "--user", "daemon-reload"] if dropin_user else ["systemctl", "daemon-reload"]
            subprocess.run(reload, capture_output=True)

        result = None
        chosen_method = "system"
        for method, cmd in [
            ("user",   ["systemctl", "--user", "restart", service]),
            ("polkit", ["systemctl", "restart", service]),
            ("sudo",   ["sudo", "-n", "systemctl", "restart", service]),
        ]:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                chosen_method = method
                break
        else:
            _remove_dropin()
            return {"service": service, "method": chosen_method,
                    "returncode": result.returncode, "stderr": result.stderr.strip()}

        if not wait:
            return {"service": service, "method": chosen_method, "returncode": 0, "waited_s": 0.0}

        srv = self.get_provider(provider)
        deadline = time.monotonic() + wait_timeout
        while time.monotonic() < deadline:
            try:
                srv._provider_transport.get_json_sync(srv.config.health_path or srv.config.models_path)
                waited = round(wait_timeout - (deadline - time.monotonic()), 1)
                _remove_dropin()
                return {"service": service, "method": chosen_method, "returncode": 0, "waited_s": waited}
            except TransportError:
                time.sleep(poll_interval)

        _remove_dropin()
        return {"service": service, "method": chosen_method, "returncode": 0,
                "waited_s": wait_timeout,
                "warning": f"service restarted but did not respond within {wait_timeout}s"}

    def get_providers(self) -> list[str]:
        names = set(known_provider_kinds())
        names.update(self.provider_registry.canonical_names())
        names.add(self.provider)
        return sorted(names)

    def get_provider(self, provider: str) -> "LLMProviderPool":
        try:
            entry = self.provider_registry.get(provider)
        except KeyError:
            return LLMProviderPool(
                provider=provider,
                timeout=self.timeout,
                load_timeout=self.load_timeout,
                logger=self.logger,
                transport=self._transport_override,
                provider_transport=self._provider_transport_override,
                provider_registry=self.provider_registry,
            )
        return LLMProviderPool(
            **provider_entry_to_server_kwargs(entry),
            timeout=self.timeout,
            load_timeout=self.load_timeout,
            logger=self.logger,
            transport=self._transport_override,
            provider_transport=self._provider_transport_override,
            provider_registry=self.provider_registry,
        )


    provider_server = get_provider   # backward-compat alias
    restart_provider = restart       # backward-compat alias

    def list_loaded_instances(self, provider=_SELF_PROVIDER) -> dict[str, str]:
        """Return {model_key: instance_id} for each model currently loaded in LM Studio."""
        if provider is _SELF_PROVIDER:
            provider = self.provider
        srv = self.get_provider(provider)
        if srv.config.name != ProviderKind.LM_STUDIO:
            return {}
        data = srv._provider_transport.get_json_sync("/api/v1/models")
        instances: dict[str, str] = {}
        for m in data.get("models") or []:
            key = m.get("key")
            loaded = m.get("loaded_instances") or []
            if key and loaded:
                instances[key] = loaded[0]["id"]
        return instances

    def list_downloaded_models(self, provider=_SELF_PROVIDER) -> list[dict[str, Any]] | dict[str, list[dict[str, Any]]]:
        """Return downloaded models as a normalized list of dicts.

        Every entry has at minimum ``{"name": str}``.
        llama_cpp entries also carry ``size_gb``, ``path``, and ``shards``.
        lm_studio entries carry ``size_gb`` when the API exposes ``size_bytes``.
        """
        if provider is _SELF_PROVIDER:
            provider = self.provider
        if provider is None:
            return {p: self.list_downloaded_models(p) for p in self.get_providers()}
        srv = self.get_provider(provider)
        if srv.config.name == ProviderKind.LLAMA_CPP:
            data = srv._provider_transport.get_json_sync("/models/downloaded")
            if isinstance(data, list) and data:
                return [
                    {
                        "name":    m.get("name") or m.get("path", ""),
                        "path":    m.get("path", ""),
                        "size_gb": m.get("total_size_gb"),
                        "shards":  m.get("shards"),
                    }
                    for m in data if isinstance(m, dict) and m.get("path")
                ]
        if srv.config.name == ProviderKind.UNSLOTH:
            data = srv._provider_transport.get_json_sync("/api/models/cached-gguf") or {}
            entries = []
            for m in data.get("cached") or []:
                repo_id = m.get("repo_id")
                if not repo_id:
                    continue
                try:
                    variants_data = srv._provider_transport.get_json_sync(
                        f"/api/models/gguf-variants?repo_id={quote(repo_id, safe='')}"
                    )
                    for v in (variants_data or {}).get("variants") or []:
                        if v.get("downloaded") and v.get("quant"):
                            entries.append({"name": f"{repo_id}@{v['quant']}"})
                except Exception:
                    pass
            if entries:
                return entries
            # cached-gguf endpoint unavailable — fall back to /v1/models
        if srv.config.name == ProviderKind.LM_STUDIO:
            data = srv._provider_transport.get_json_sync("/api/v1/models")
            return [
                {
                    "name":    m["key"],
                    "size_gb": round(m["size_bytes"] / 1024**3, 2) if m.get("size_bytes") else None,
                }
                for m in (data.get("models") or [])
                if m.get("key") and m.get("size_bytes", 0) > 0
            ]
        data = srv._provider_transport.get_json_sync(srv.config.models_path)
        if isinstance(data, dict):
            return [{"name": m["id"]} for m in (data.get("data") or []) if m.get("id")]
        return []

    def unload_all_models(self, provider=_SELF_PROVIDER) -> Any:
        if provider is _SELF_PROVIDER:
            provider = self.provider
        if provider is None:
            results = {}
            for p in self.get_providers():
                try:
                    results[p] = self.unload_all_models(p)
                except TransportError as exc:
                    results[p] = {"unloaded": [], "error": f"unreachable: {exc}"}
            return results
        srv = self.get_provider(provider)
        if srv.config.name == ProviderKind.LM_STUDIO:
            instances = self.list_loaded_instances(provider)
            for key, instance_id in instances.items():
                srv._provider_transport.post_json_sync(
                    "/api/v1/models/unload", {"instance_id": instance_id}
                )
            return {"unloaded": list(instances.keys())}
        if srv.config.name == ProviderKind.UNSLOTH:
            status = srv._provider_transport.get_json_sync("/api/inference/status")
            active = status.get("active_model") if status else None
            if active:
                srv._provider_transport.post_json_sync("/api/inference/unload", {"model_path": active})
                return {"unloaded": [active]}
            return {"unloaded": []}
        if srv.config.name == ProviderKind.LLAMA_CPP:
            r = srv._provider_transport.post_json_sync("/models/unload", {})
            return r.json() or {"unloaded": []}
        if srv.config.name != ProviderKind.OLLAMA:
            return {"unloaded": [], "reason": f"unload not supported for provider: {srv.config.name}"}
        ps = srv._provider_transport.get_json_sync("/api/ps")
        running = ps.get("models") or []
        unloaded = []
        for entry in running:
            name = entry.get("name") or entry.get("model")
            if name:
                srv._provider_transport.post_json_sync("/api/generate", {"model": name, "keep_alive": 0})
                unloaded.append(name)
        return {"unloaded": unloaded}

    async def check_provider(self, provider: str | None = None) -> dict[str, Any]:
        if provider is not None:
            return await self.get_provider(provider).check_provider()

        if self.config.health_path is not None:
            try:
                data = await self._provider_transport.get_json(self.config.health_path)
                return {
                    "provider": self.provider,
                    "server_url": self.server_url,
                    "ok": True,
                    "kind": "health",
                    "data": data,
                }
            except json.JSONDecodeError:
                return {
                    "provider": self.provider,
                    "server_url": self.server_url,
                    "ok": True,
                    "kind": "health",
                    "data": None,
                }
            except Exception as exc:
                return {
                    "provider": self.provider,
                    "server_url": self.server_url,
                    "ok": False,
                    "kind": "health",
                    "error": str(exc),
                }

        try:
            data = await self._provider_transport.get_json(self.config.models_path)
            return {
                "provider": self.provider,
                "server_url": self.server_url,
                "ok": True,
                "kind": "models",
                "data": data,
            }
        except Exception as exc:
            return {
                "provider": self.provider,
                "server_url": self.server_url,
                "ok": False,
                "kind": "models",
                "error": str(exc),
            }

    check_provider_by_name = check_provider  # backward-compat alias

    async def resolve_model(self) -> str:
        if self.model:
            return self.model

        models = await self._provider_transport.get_json(self.config.models_path)
        model_ids: list[str] = []
        if isinstance(models, dict):
            data = models.get("data")
            if isinstance(data, list):
                model_ids = [m.get("id") for m in data if isinstance(m, dict) and m.get("id")]
        if not model_ids:
            raise ValueError("model is required and /models returned no model ids")

        self.model = model_ids[0]
        return self.model

    async def resolve_call_model(self, model: str | None = None) -> str:
        if model:
            return model
        return await self.resolve_model()

    def _build_payload(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
        options: dict[str, Any] | None,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = tools

        opts = dict(options or {})

        if self.config.name == ProviderKind.OLLAMA:
            # num_ctx  — context window (Ollama-native, keep in options)
            # num_predict — generation limit (Ollama-native alias for max_tokens)
            if max_tokens is not None:
                opts.setdefault("num_predict", max_tokens)
            # remove generic aliases that don't belong in Ollama options
            opts.pop("max_tokens", None)
            opts.pop("max_completion_tokens", None)
            if opts:
                payload["options"] = opts
        else:
            # All other providers: use standard OpenAI fields, drop Ollama-specific keys
            ollama_only = {"num_ctx", "num_predict", "num_keep", "num_thread",
                           "mirostat", "mirostat_eta", "mirostat_tau",
                           "repeat_last_n", "repeat_penalty", "tfs_z", "typical_p"}
            for key in list(opts.keys()):
                if key in ollama_only:
                    opts.pop(key)
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            # promote remaining standard keys
            for key in ("top_p", "seed", "stop", "presence_penalty",
                        "frequency_penalty", "reasoning_effort"):
                if key in opts:
                    payload[key] = opts.pop(key)
            # any remaining provider-specific keys pass through
            if opts:
                payload.update(opts)

        return payload

    async def _call(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str | None = None,
        schema_dict: dict[str, Any] | None = None,
        think: bool = False,
        temperature: float | None = None,
        effort: str | None = None,
        options: dict[str, Any] | None = None,
        context_length: int | None = None,
        gpu_layers: int | None = None,
        cmoe: bool | None = None,
        cpu_moe: int | None = None,
        flash_attn: bool | None = None,
        batch_size: int | None = None,
        kv_offload: bool | None = None,
        num_experts: int | None = None,
        threads: int | None = None,
        main_gpu: int | None = None,
        gpu_offload: float | str | None = None,
        load_params: dict[str, Any] | None = None,
        num_parallel: int | None = None,
        kv_cache_type: str | None = None,
        gpu_overhead: int | None = None,
        max_loaded_models: int | None = None,
        max_json_fix_retries: int = 3,
        max_loop_retries: int = 5,
        tools: list[dict[str, Any]] | None = None,
        tool_registry: dict[str, ToolHandler] | None = None,
        max_tool_rounds: int = 5,
        return_usage: bool = False,
        load_timeout: float | None = None,
    ) -> Any:
        # Prepare messages
        prepared = [dict(m) for m in messages]
        if effort:
            prepared.insert(0, {"role": "system", "content": f"Reasoning effort: {effort}."})
        if not think:
            prepared.insert(0, {"role": "system", "content": "Respond with the final answer only. Do not emit hidden reasoning."})

        # Extract scalar params from options
        opts = dict(options or {})
        if context_length is not None and self.config.name == ProviderKind.OLLAMA:
            opts.setdefault("num_ctx", context_length)
        if temperature is None:
            temperature = float(opts["temperature"]) if "temperature" in opts else None
        max_tokens: int | None = None
        for key in ("max_tokens", "max_completion_tokens", "num_predict"):
            if key in opts:
                max_tokens = int(opts[key])
                break

        model_name = model or self.model or await self.resolve_call_model(model)
        effective_load_timeout = load_timeout if load_timeout is not None else self.load_timeout

        # Auto-restart if server-level params changed since last call
        _server_params: dict[str, Any] = {}
        _server_env: dict[str, str] = {}
        if self.config.name == ProviderKind.OLLAMA:
            if flash_attn is not None:
                _server_params["flash_attn"] = flash_attn
                _server_env["OLLAMA_FLASH_ATTENTION"] = "1" if flash_attn else "0"
            if kv_cache_type is not None:
                _server_params["kv_cache_type"] = kv_cache_type
                _server_env["OLLAMA_KV_CACHE_TYPE"] = kv_cache_type
            if num_parallel is not None:
                _server_params["num_parallel"] = num_parallel
                _server_env["OLLAMA_NUM_PARALLEL"] = str(num_parallel)
            if gpu_overhead is not None:
                _server_params["gpu_overhead"] = gpu_overhead
                _server_env["OLLAMA_GPU_OVERHEAD"] = str(gpu_overhead)
            if max_loaded_models is not None:
                _server_params["max_loaded_models"] = max_loaded_models
                _server_env["OLLAMA_MAX_LOADED_MODELS"] = str(max_loaded_models)
        elif self.config.name == ProviderKind.LLAMA_CPP:
            if num_parallel is not None:
                _server_params["num_parallel"] = num_parallel
                _server_env["LLAMA_ARG_N_PARALLEL"] = str(num_parallel)
        if _server_params:
            changed = {k: v for k, v in _server_params.items() if self._active_server_config.get(k) != v}
            if changed:
                await asyncio.to_thread(self.restart, env=_server_env)
                self._active_server_config.update(_server_params)

        if self.config.name == ProviderKind.LM_STUDIO and model_name:
            await asyncio.to_thread(
                self._ensure_lm_studio_loaded, model_name, context_length,
                gpu_offload=gpu_offload, flash_attn=flash_attn, batch_size=batch_size,
                kv_offload=kv_offload, num_experts=num_experts, load_params=load_params,
                load_timeout=effective_load_timeout,
            )

        if self.config.name == ProviderKind.LLAMA_CPP and model_name:
            await asyncio.to_thread(
                self._ensure_llama_cpp_loaded, model_name, context_length, gpu_layers,
                cmoe=cmoe, cpu_moe=cpu_moe, flash_attn=flash_attn, batch_size=batch_size,
                kv_offload=kv_offload, threads=threads, main_gpu=main_gpu,
                load_params=load_params, load_timeout=effective_load_timeout,
            )

        if self.config.name == ProviderKind.UNSLOTH and model_name:
            if tools:
                raise NotImplementedError(
                    "Unsloth does not support client-side tool calls. "
                    "Tools are executed server-side only via enable_tools=true."
                )
            await asyncio.to_thread(self._ensure_unsloth_loaded, model_name, context_length, load_params=load_params, load_timeout=effective_load_timeout)

        structured_request = None
        if schema_dict is not None:
            structured_request = self._json_fix.build_request(
                messages=prepared,
                schema_dict=schema_dict,
            )
        attempt_messages = list(structured_request.messages if structured_request else prepared)
        request_tools = self._tool_pipeline.openai_tools(tools)
        tool_registry = tool_registry or {}
        json_retries_used = 0
        loop_retries_used = 0
        usage: dict[str, int] = {}

        while True:
            text = ""
            for tool_round in range(max_tool_rounds + 1):
                payload = self._build_payload(
                    model=model_name,
                    messages=attempt_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    options=opts,
                    tools=request_tools,
                )


                if self.logger is not None:
                    await self.logger.log_request(
                        provider=self.provider,
                        model=model_name,
                        messages=attempt_messages,
                        payload=payload,
                    )

                if self.config.name == ProviderKind.OLLAMA:
                    # Ollama native /api/chat: only path that correctly applies num_ctx.
                    # Pre-load to warm the model at the right context size / load params first.
                    native_opts: dict[str, Any] = {}
                    # load_params passthrough (user uses Ollama option names directly)
                    if load_params:
                        native_opts.update(load_params)
                    # unified load params → Ollama option names
                    if context_length is not None:
                        native_opts["num_ctx"] = context_length
                    if threads is not None:
                        native_opts.setdefault("num_thread", threads)
                    if main_gpu is not None:
                        native_opts.setdefault("main_gpu", main_gpu)
                    if batch_size is not None:
                        native_opts.setdefault("num_batch", batch_size)
                    if max_tokens is not None:
                        native_opts.setdefault("num_predict", max_tokens)
                    for key in ("num_predict", "num_keep", "num_thread", "mirostat",
                                "mirostat_eta", "mirostat_tau", "repeat_last_n",
                                "repeat_penalty", "tfs_z", "typical_p", "temperature"):
                        if key in opts:
                            native_opts.setdefault(key, opts[key])
                    if native_opts:
                        await asyncio.to_thread(self._ollama_preload_sync, model_name, native_opts, load_timeout=effective_load_timeout)
                    ollama_body: dict[str, Any] = {
                        "model": model_name,
                        "messages": attempt_messages,
                        "stream": True,
                    }
                    if native_opts:
                        ollama_body["options"] = native_opts
                    response = await asyncio.to_thread(
                        self._stream_chat_sync, "/api/chat", ollama_body,
                        ollama_native=True,
                        _logger=self.logger,
                    )
                else:
                    stream_body = dict(payload)
                    stream_body["stream"] = True
                    stream_body["stream_options"] = {"include_usage": True}
                    response = await asyncio.to_thread(
                        self._stream_chat_sync, "/v1/chat/completions", stream_body,
                        _logger=self.logger,
                    )
                if hasattr(response, "usage") and response.usage is not None:
                    u = response.usage
                    usage = {
                        "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                        "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
                    }
                text = response.content
                tool_calls = response.choices[0].message.tool_calls if response.choices else []
                if not tool_calls:
                    if think and response.reasoning:
                        text = f"<think>{response.reasoning}</think>\n\n{response.content or ''}"
                    else:
                        text = response.content or response.reasoning or ""
                    break

                if tool_round >= max_tool_rounds:
                    raise RuntimeError(f"max tool rounds exceeded: {max_tool_rounds}")

                normalized_tool_calls = self._tool_pipeline.normalize_tool_calls(tool_calls)
                executions = await self._tool_pipeline.execute_tool_calls(
                    normalized_tool_calls,
                    tool_registry,
                )
                if self.logger is not None:
                    for execution in executions:
                        await self.logger.log_info(
                            "\n[TOOL_RESULT: "
                            f"{execution.name}]\n"
                            + json.dumps(
                                {
                                    "name": execution.name,
                                    "arguments": execution.arguments,
                                    "result": execution.result,
                                },
                                ensure_ascii=False,
                                indent=2,
                            )
                        )
                attempt_messages.append(
                    self._tool_pipeline.assistant_message(text, normalized_tool_calls)
                )
                attempt_messages.extend(self._tool_pipeline.tool_messages(executions))

            loop_reason = self._loop_guard.check(text)
            if loop_reason is not None:
                if self.logger is not None:
                    await self.logger.log_response(
                        provider=self.provider,
                        model=model_name,
                        text=text,
                        error=f"loop_guard:{loop_reason}",
                    )
                if loop_retries_used >= max_loop_retries:
                    raise RuntimeError(f"loop detected: {loop_reason}")
                loop_retries_used += 1
                attempt_messages = list(attempt_messages) + self._loop_guard.build_retry_messages(text, loop_reason)
                continue

            if structured_request is not None:
                try:
                    parsed = self._json_fix.parse(text, structured_request.model_cls)
                except Exception as exc:
                    if self.logger is not None:
                        await self.logger.log_response(
                            provider=self.provider,
                            model=model_name,
                            text=text,
                            error=str(exc),
                        )
                    if json_retries_used >= max_json_fix_retries:
                        attempts = json_retries_used + 1
                        raise ValueError(
                            f"JSON fix failed after {attempts} attempt(s): {exc}"
                        ) from exc
                    json_retries_used += 1
                    attempt_messages = list(attempt_messages) + self._json_fix.build_retry_messages(
                        text,
                        exc,
                        structured_request.schema_json,
                    )
                    continue

                if self.logger is not None:
                    await self.logger.log_end(
                        completion_tokens=usage.get("completion_tokens"),
                        ollama_prompt_tokens=usage.get("prompt_tokens"),
                    )
                    await self.logger.log_info(f"parsed={json.dumps(parsed, ensure_ascii=False)}")
                return (parsed, usage) if return_usage else parsed

            if self.logger is not None:
                await self.logger.log_end(
                    completion_tokens=usage.get("completion_tokens"),
                    ollama_prompt_tokens=usage.get("prompt_tokens"),
                )
            return (text, usage) if return_usage else text

    async def close(self) -> None:
        if self.logger is not None:
            await self.logger.close()
