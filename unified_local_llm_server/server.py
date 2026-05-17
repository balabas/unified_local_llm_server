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
from .transport import OpenAICompatibleTransport



@dataclass(slots=True)
class LocalLLM:
    server: "LocalLLMServer"
    provider: str
    model: str
    context_length: int | None = None
    defaults: dict[str, Any] = field(default_factory=dict)

    @property
    def server_url(self) -> str:
        return self.server.server_url

    @property
    def base_url(self) -> str:
        return self.server.base_url

    async def call(self, *, return_usage: bool = False, **kwargs: Any) -> Any:
        merged = self._merge_defaults(kwargs)
        merged["model"] = self.model
        if self.context_length is not None:
            merged.setdefault("context_length", self.context_length)
        return await self.server._call(return_usage=return_usage, **merged)

    async def batch(
        self,
        items: list[list[dict[str, Any]] | dict[str, Any]],
        *,
        concurrency: int = 4,
        return_exceptions: bool = False,
        **shared_call_kwargs: Any,
    ) -> list[Any]:
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


class LocalLLMServer:
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
        logger: AsyncLLMLogger | None = None,
        transport: OpenAICompatibleTransport | None = None,
        provider_transport: OpenAICompatibleTransport | None = None,
        context_length: int | None = None,
        **defaults: Any,
    ) -> LocalLLM:
        resolved_provider = provider or self.provider
        resolved_model = model or (self.model if resolved_provider == self.provider else None)
        if not resolved_model:
            raise ValueError("model is required when loading a model handle")

        if context_length is not None:
            options = dict(defaults.get("options") or {})
            options.setdefault("num_ctx", context_length)
            defaults["options"] = options


        has_connection_overrides = any(v is not None for v in (
            host, port, api_key, base_path, health_path, models_path, timeout, transport, provider_transport
        ))
        use_self = resolved_provider == self.provider and not has_connection_overrides
        if use_self:
            target_server = self
        elif has_connection_overrides:
            target_server = LocalLLMServer(
                provider=resolved_provider,
                model=resolved_model,
                host=host,
                port=port,
                api_key=api_key,
                base_path=base_path,
                health_path=health_path,
                models_path=models_path,
                timeout=timeout if timeout is not None else self.timeout,
                transport=transport,
                provider_transport=provider_transport,
                provider_registry=self.provider_registry,
            )
        else:
            target_server = self.provider_server(resolved_provider)
        if logger is not None:
            target_server.logger = logger
        return LocalLLM(
            server=target_server,
            provider=target_server.provider,
            model=resolved_model,
            context_length=context_length,
            defaults=defaults,
        )

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

    def _ollama_native_chat_sync(
        self,
        model: str,
        messages: list[dict[str, Any]],
        context_length: int | None,
        max_tokens: int | None,
        opts: dict[str, Any],
    ) -> Any:
        """Call Ollama's native /api/chat, which correctly applies num_ctx."""
        import types

        native_opts: dict[str, Any] = {}
        if context_length is not None:
            native_opts["num_ctx"] = context_length
        if max_tokens is not None:
            native_opts["num_predict"] = max_tokens
        for key in ("num_predict", "num_keep", "num_thread", "mirostat",
                    "mirostat_eta", "mirostat_tau", "repeat_last_n",
                    "repeat_penalty", "tfs_z", "typical_p", "temperature"):
            if key in opts:
                native_opts.setdefault(key, opts[key])
        # Pre-load the model with the correct context before the inference call.
        # This ensures the model is warm and avoids an empty response on the first call.
        preload: dict[str, Any] = {"model": model, "prompt": "", "stream": False, "keep_alive": -1}
        if native_opts:
            preload["options"] = dict(native_opts)
        self._provider_transport.post_json_sync("/api/generate", preload)

        body: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if native_opts:
            body["options"] = native_opts
        for _attempt in range(2):
            r = self._provider_transport.post_json_sync("/api/chat", body)
            data = r.json()
            if data is None or r.status >= 400:
                raise RuntimeError(f"Ollama /api/chat failed (HTTP {r.status}): {r.text!r}")
            content = (data.get("message") or {}).get("content") or ""
            if content:
                break
        usage_ns = types.SimpleNamespace(
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
        )
        msg = types.SimpleNamespace(tool_calls=None)
        choice = types.SimpleNamespace(message=msg)
        return types.SimpleNamespace(content=content, reasoning=None, choices=[choice], usage=usage_ns)

    def _unsloth_chat_sync(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None,
        opts: dict[str, Any],
    ) -> Any:
        """Call unsloth via streaming so the final chunk includes real token counts."""
        import json as _json
        import types

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        for key in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty"):
            if key in opts:
                body[key] = opts[key]

        content_parts: list[str] = []
        prompt_tok = 0
        completion_tok = 0

        for line in self._provider_transport.stream_lines_sync("/v1/chat/completions", body):
            if not line or not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if raw == "[DONE]":
                break
            try:
                chunk = _json.loads(raw)
            except _json.JSONDecodeError:
                continue
            for ch in chunk.get("choices") or []:
                delta = ch.get("delta") or {}
                if delta.get("content"):
                    content_parts.append(delta["content"])
            u = chunk.get("usage") or {}
            if u.get("prompt_tokens"):
                prompt_tok = u["prompt_tokens"]
            if u.get("completion_tokens"):
                completion_tok = u["completion_tokens"]

        content = "".join(content_parts)
        usage_ns = types.SimpleNamespace(prompt_tokens=prompt_tok, completion_tokens=completion_tok)
        msg = types.SimpleNamespace(tool_calls=None)
        choice_ns = types.SimpleNamespace(message=msg)
        return types.SimpleNamespace(content=content, reasoning=None, choices=[choice_ns], usage=usage_ns)

    def _ensure_lm_studio_loaded(self, model_name: str, context_length: int | None = None) -> None:
        if context_length is None:
            return
        # Skip if already loaded (LM Studio returns 500 if you reload an active instance)
        info = self._provider_transport.get_json_sync("/api/v1/models")
        for m in (info or {}).get("models") or []:
            if m.get("key") == model_name and m.get("loaded_instances"):
                return
        body: dict[str, Any] = {
            "model": model_name,
            "context_length": context_length,
        }
        r = self._provider_transport.post_json_sync("/api/v1/models/load", body)
        if r.status >= 400:
            raise RuntimeError(f"LM Studio /api/v1/models/load failed (HTTP {r.status}): {r.text!r}")

    def _ensure_llama_cpp_loaded(self, model_name: str, context_length: int | None = None) -> None:
        r = self._provider_transport._do_request("GET", "/models/loaded", None)
        if r.status >= 400:
            raise RuntimeError(f"llama_cpp /models/loaded failed (HTTP {r.status}): {r.text!r}")
        loaded = r.json() or []
        if model_name in loaded and context_length is None:
            return
        body: dict[str, Any] = {"model": model_name}
        if context_length is not None:
            body["ctx_size"] = context_length
            if model_name in loaded:
                body["force_restart"] = True
        r2 = self._provider_transport.post_json_sync("/models/load", body)
        if r2.status >= 400:
            raise RuntimeError(f"llama_cpp /models/load failed (HTTP {r2.status}): {r2.text!r}")

    def _ensure_unsloth_loaded(self, model_path: str, context_length: int | None = None) -> None:
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
        if already_loaded and ctx_matches:
            return
        body: dict[str, Any] = {"model_path": resolved}
        if context_length is not None:
            body["max_seq_length"] = context_length
        r2 = self._provider_transport.post_json_sync("/api/inference/load", body)
        if r2.status >= 400:
            raise RuntimeError(f"Unsloth /api/inference/load failed (HTTP {r2.status}): {r2.text!r}")

    def list_loaded_models(self, provider: str | None = None) -> list[str] | dict[str, list[str]]:
        if provider is None:
            return {p: self.list_loaded_models(p) for p in self.get_providers()}
        srv = self.provider_server(provider)
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

    def get_providers(self) -> list[str]:
        names = set(known_provider_kinds())
        names.update(self.provider_registry.canonical_names())
        names.add(self.provider)
        return sorted(names)

    def provider_server(self, provider: str) -> "LocalLLMServer":
        try:
            entry = self.provider_registry.get(provider)
        except KeyError:
            return LocalLLMServer(
                provider=provider,
                timeout=self.timeout,
                logger=self.logger,
                transport=self._transport_override,
                provider_transport=self._provider_transport_override,
                provider_registry=self.provider_registry,
            )
        return LocalLLMServer(
            **provider_entry_to_server_kwargs(entry),
            timeout=self.timeout,
            logger=self.logger,
            transport=self._transport_override,
            provider_transport=self._provider_transport_override,
            provider_registry=self.provider_registry,
        )


    def list_provider_models(self, provider: str) -> list[str]:
        return self.list_loaded_models(provider)

    def list_loaded_instances(self, provider: str) -> dict[str, str]:
        """Return {model_key: instance_id} for each model currently loaded in LM Studio."""
        srv = self.provider_server(provider)
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

    def list_downloaded_models(self, provider: str | None = None) -> Any:
        if provider is None:
            return {p: self.list_downloaded_models(p) for p in self.get_providers()}
        srv = self.provider_server(provider)
        if srv.config.name == ProviderKind.LLAMA_CPP:
            data = srv._provider_transport.get_json_sync("/models/downloaded")
            if isinstance(data, list) and data:
                return [m.get("name") or m["path"] for m in data if isinstance(m, dict) and m.get("path")]
        if srv.config.name == ProviderKind.UNSLOTH:
            data = srv._provider_transport.get_json_sync("/api/models/cached-gguf") or {}
            names = []
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
                            names.append(f"{repo_id}@{v['quant']}")
                except Exception:
                    pass
            if names:
                return names
            # cached-gguf endpoint unavailable — fall back to /v1/models
        if srv.config.name == ProviderKind.LM_STUDIO:
            data = srv._provider_transport.get_json_sync("/api/v1/models")
            return [
                m["key"] for m in (data.get("models") or [])
                if m.get("key") and m.get("size_bytes", 0) > 0
            ]
        data = srv._provider_transport.get_json_sync(srv.config.models_path)
        if isinstance(data, dict):
            return [m["id"] for m in (data.get("data") or []) if m.get("id")]
        return []

    def unload_all_models(self, provider: str | None = None) -> Any:
        if provider is None:
            return {p: self.unload_all_models(p) for p in self.get_providers()}
        srv = self.provider_server(provider)
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

    async def check_provider_by_name(self, provider: str) -> dict[str, Any]:
        return await self.provider_server(provider).check_provider()

    async def check_provider(self) -> dict[str, Any]:
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
        max_json_fix_retries: int = 3,
        max_loop_retries: int = 5,
        tools: list[dict[str, Any]] | None = None,
        tool_registry: dict[str, ToolHandler] | None = None,
        max_tool_rounds: int = 5,
        return_usage: bool = False,
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

        if self.config.name == ProviderKind.LM_STUDIO and model_name:
            await asyncio.to_thread(self._ensure_lm_studio_loaded, model_name, context_length)

        if self.config.name == ProviderKind.LLAMA_CPP and model_name:
            await asyncio.to_thread(self._ensure_llama_cpp_loaded, model_name, context_length)

        if self.config.name == ProviderKind.UNSLOTH and model_name:
            if tools:
                raise NotImplementedError(
                    "Unsloth does not support client-side tool calls. "
                    "Tools are executed server-side only via enable_tools=true."
                )
            await asyncio.to_thread(self._ensure_unsloth_loaded, model_name, context_length)

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

                if self.config.name == ProviderKind.OLLAMA and context_length is not None:
                    response = await asyncio.to_thread(
                        self._ollama_native_chat_sync,
                        model_name, attempt_messages, context_length, max_tokens, opts,
                    )
                elif self.config.name == ProviderKind.UNSLOTH:
                    response = await asyncio.to_thread(
                        self._unsloth_chat_sync,
                        model_name, attempt_messages, max_tokens, opts,
                    )
                else:
                    response = await self._client.chat.completions.create(**payload)
                if hasattr(response, "usage") and response.usage is not None:
                    u = response.usage
                    usage = {
                        "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                        "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
                    }
                text = response.content
                tool_calls = response.choices[0].message.tool_calls if response.choices else []
                if not tool_calls:
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
                    await self.logger.log_response(
                        provider=self.provider,
                        model=model_name,
                        text=text,
                        parsed=parsed,
                    )
                return (parsed, usage) if return_usage else parsed

            if self.logger is not None:
                await self.logger.log_response(
                    provider=self.provider,
                    model=model_name,
                    text=text,
                )
            return (text, usage) if return_usage else text

    async def close(self) -> None:
        if self.logger is not None:
            await self.logger.close()
