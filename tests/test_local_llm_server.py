from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from unified_local_llm_server.llm_logger import AsyncLLMLogger
from unified_local_llm_server.pipelines.json_fix import JsonFixPipeline
from unified_local_llm_server.pipelines.loop_guard import LoopGuardPipeline
from unified_local_llm_server.pipelines.tool import ToolPipeline
from unified_local_llm_server.provider_registry import ProviderRegistry
from unified_local_llm_server.providers import ProviderKind, resolve_provider_config
from unified_local_llm_server.pool import LLMProviderPool
from unified_local_llm_server.transport import HttpResult


class FakeTransport:
    def __init__(self, *, status: int = 200, text: str = '{"data": [{"id": "fake-model"}]}'):
        self.calls: list[str] = []
        self.status = status
        self.text = text

    def get_json_sync(self, path: str):
        self.calls.append(path)
        return {"data": [{"id": "fake-model"}]}

    async def get_json(self, path: str):
        self.calls.append(path)
        return {"data": [{"id": "fake-model"}]}

    async def _request(self, method: str, path: str, payload: dict[str, Any] | None):
        self.calls.append(path)
        return HttpResult(status=self.status, headers={}, text=self.text)


class FakeProviderTransport:
    def __init__(self, models: dict[str, list[str]]):
        self.models = models
        self.calls: list[str] = []

    def get_json_sync(self, path: str):
        self.calls.append(path)
        ids = self.models.get(path, [])
        if path == "/api/ps":
            return {"models": [{"name": item} for item in ids]}
        if path == "/api/v1/models":
            return {"models": [{"key": item, "size_bytes": 1, "loaded_instances": [{"id": f"inst-{item}"}]} for item in ids]}
        return {"data": [{"id": item} for item in ids]}

    async def get_json(self, path: str):
        self.calls.append(path)
        ids = self.models.get(path, [])
        return {"data": [{"id": item} for item in ids]}


class FakeStreamTransport:
    def __init__(self, lines: list[str]):
        self.lines = lines

    def stream_lines_sync(self, path: str, payload: dict[str, Any]):
        yield from self.lines


class LLMProviderPoolTests(unittest.IsolatedAsyncioTestCase):
    def test_provider_defaults(self):
        lm = resolve_provider_config(ProviderKind.LM_STUDIO)
        ollama = resolve_provider_config(ProviderKind.OLLAMA)
        llama = resolve_provider_config(ProviderKind.LLAMA_CPP)
        unsloth = resolve_provider_config(ProviderKind.UNSLOTH)
        self.assertEqual(lm.port, 1234)
        self.assertEqual(ollama.port, 11434)
        self.assertEqual(llama.port, 8080)
        self.assertEqual(unsloth.port, 8895)
        self.assertEqual(ollama.health_path, "/api/version")
        self.assertEqual(llama.health_path, "/health")
        self.assertTrue(lm.base_url.endswith("/v1"))
        self.assertEqual(ollama.server_url, "http://127.0.0.1:11434")

    def test_loop_guard_detects_repetition(self):
        guard = LoopGuardPipeline()
        text = ("alpha beta gamma " * 500).strip()
        self.assertIsNotNone(guard.check(text))

    def test_json_fix_recovers_simple_invalid_json(self):
        pipeline = JsonFixPipeline()
        schema = {"name": str, "age": int}
        from unified_local_llm_server.helpers.pydantic_helper import dict_to_pydantic_schema

        model = dict_to_pydantic_schema(schema)
        text = """Here is the result:
        ```json
        {'name': 'Ana', 'age': 31,}
        ```
        """
        parsed = pipeline.parse(text, model)
        self.assertEqual(parsed["name"], "Ana")
        self.assertEqual(parsed["age"], 31)

    def test_json_fix_reports_empty_output(self):
        pipeline = JsonFixPipeline()
        from unified_local_llm_server.helpers.pydantic_helper import dict_to_pydantic_schema

        model = dict_to_pydantic_schema({"name": str, "age": int})
        with self.assertRaisesRegex(ValueError, "response was empty"):
            pipeline.parse("", model)

    def test_json_fix_reports_no_json_object(self):
        pipeline = JsonFixPipeline()
        from unified_local_llm_server.helpers.pydantic_helper import dict_to_pydantic_schema

        model = dict_to_pydantic_schema({"name": str, "age": int})
        with self.assertRaisesRegex(ValueError, "No JSON object or array found"):
            pipeline.parse("I cannot provide that.", model)

    def test_json_fix_pipeline_owns_schema_prompt_preparation(self):
        pipeline = JsonFixPipeline()
        request = pipeline.build_request(
            messages=[
                {"role": "system", "content": "Use schema:\n<SCHEMA_DICT>"},
                {"role": "user", "content": "Return a person"},
            ],
            schema_dict={"name": str, "age": int},
        )
        self.assertIn('"name"', request.schema_json)
        self.assertIn('"age"', request.schema_json)
        self.assertIn('"name"', request.messages[0]["content"])
        self.assertNotIn("<SCHEMA_DICT>", request.messages[0]["content"])

    async def test_tool_pipeline_executes_registered_tool(self):
        pipeline = ToolPipeline()

        async def add(arguments: dict[str, Any]) -> dict[str, Any]:
            return {"sum": arguments["a"] + arguments["b"]}

        executions = await pipeline.execute_tool_calls(
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "add", "arguments": "{\"a\": 2, \"b\": 3}"},
                }
            ],
            {"add": add},
        )
        self.assertEqual(executions[0].name, "add")
        self.assertEqual(executions[0].arguments, {"a": 2, "b": 3})
        self.assertEqual(json.loads(executions[0].result), {"sum": 5})
        self.assertEqual(pipeline.tool_messages(executions)[0]["role"], "tool")

    async def test_tool_pipeline_accepts_streamed_namespace_tool_calls(self):
        pipeline = ToolPipeline()
        tool_calls = [
            SimpleNamespace(
                id="call_1",
                type="function",
                function=SimpleNamespace(
                    name="add",
                    arguments='{"a": 2, "b": 3}',
                ),
            )
        ]

        def add(arguments: dict[str, Any]) -> dict[str, Any]:
            return {"sum": arguments["a"] + arguments["b"]}

        normalized = pipeline.normalize_tool_calls(tool_calls)
        self.assertEqual(normalized[0]["id"], "call_1")
        self.assertEqual(normalized[0]["function"]["name"], "add")
        self.assertEqual(normalized[0]["function"]["arguments"], {"a": 2, "b": 3})

        executions = await pipeline.execute_tool_calls(tool_calls, {"add": add})
        self.assertEqual(json.loads(executions[0].result), {"sum": 5})

        assistant = pipeline.assistant_message("", tool_calls)
        self.assertEqual(assistant["tool_calls"][0]["function"]["arguments"], '{"a": 2, "b": 3}')

    def test_ollama_native_stream_extracts_tool_calls(self):
        transport = FakeStreamTransport([
            json.dumps({
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "add",
                                "arguments": {"a": 2, "b": 3},
                            }
                        }
                    ]
                },
                "done": True,
            })
        ])
        server = LLMProviderPool(
            provider=ProviderKind.OLLAMA,
            provider_transport=transport,
        )

        response = server._stream_chat_sync("/api/chat", {}, ollama_native=True)

        tool_calls = response.choices[0].message.tool_calls
        self.assertEqual(tool_calls[0].function.name, "add")
        self.assertEqual(tool_calls[0].function.arguments, '{"a": 2, "b": 3}')

    def test_provider_registry_loads_named_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "providers.json"
            path.write_text(
                json.dumps(
                    {
                        "providers": {
                            "ollama": {
                                "port": 11434,
                            },
                            "lmstudio": {
                                "provider": "lm_studio",
                                "port": 1234,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            registry = ProviderRegistry.load(path)
            ollama = registry.get("ollama")
            lmstudio = registry.get("lmstudio")
            self.assertEqual(ollama.provider, "ollama")
            self.assertEqual(ollama.port, 11434)
            self.assertEqual(lmstudio.provider, "lm_studio")
            self.assertEqual(registry.canonical_names(), ["lm_studio", "ollama"])

    async def test_get_providers_includes_registry_and_default(self):
        registry = ProviderRegistry({})
        server = LLMProviderPool(provider=ProviderKind.OLLAMA, provider_registry=registry)
        self.assertIn(ProviderKind.OLLAMA, server.get_providers())
        self.assertIn(ProviderKind.LM_STUDIO, server.get_providers())
        self.assertIn(ProviderKind.UNSLOTH, server.get_providers())
        self.assertIn(ProviderKind.LLAMA_CPP, server.get_providers())
        self.assertNotIn("lmstudio", server.get_providers())
        self.assertNotIn("llama-cpp", server.get_providers())

    async def test_list_loaded_models_routes_by_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "providers.json"
            path.write_text(
                json.dumps(
                    {
                        "providers": {
                            "ollama": {
                                "port": 11434,
                                "models_path": "/ollama-models",
                            },
                            "lmstudio": {
                                "provider": "lm_studio",
                                "port": 1234,
                                "models_path": "/lmstudio-models",
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            registry = ProviderRegistry.load(path)
            provider_transport = FakeProviderTransport({
                "/api/ps": ["ollama-a"],
                "/api/v1/models": ["lmstudio-b"],
            })
            server = LLMProviderPool(
                provider=ProviderKind.OLLAMA,
                provider_registry=registry,
                provider_transport=provider_transport,
                transport=FakeTransport(),
            )
            ollama_models = server.list_loaded_models("ollama")
            lmstudio_models = server.list_loaded_models("lmstudio")
            self.assertEqual(ollama_models[0], "ollama-a")
            self.assertEqual(lmstudio_models[0], "lmstudio-b")

    async def test_provider_check_uses_endpoint_config(self):
        transport = FakeTransport()
        server = LLMProviderPool(
            provider=ProviderKind.LM_STUDIO,
            transport=transport,
        )
        status = await server.check_provider()
        self.assertTrue(status["ok"])
        self.assertEqual(status["server_url"], "http://127.0.0.1:1234")
        self.assertEqual(status["status"], 200)

    async def test_provider_check_reports_http_error_status(self):
        transport = FakeTransport(status=503, text='{"error": "starting"}')
        server = LLMProviderPool(
            provider=ProviderKind.LM_STUDIO,
            transport=transport,
        )

        status = await server.check_provider()

        self.assertFalse(status["ok"])
        self.assertEqual(status["kind"], "models")
        self.assertEqual(status["status"], 503)
        self.assertIn("starting", status["error"])

    async def test_provider_check_accepts_empty_successful_health_response(self):
        transport = FakeTransport(status=204, text="")
        server = LLMProviderPool(
            provider=ProviderKind.OLLAMA,
            provider_transport=transport,
        )

        status = await server.check_provider()

        self.assertTrue(status["ok"])
        self.assertEqual(status["kind"], "health")
        self.assertEqual(status["status"], 204)
        self.assertIsNone(status["data"])

    async def test_async_logger_writes_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "llm.log"
            logger = AsyncLLMLogger(path)
            await logger.log_request(
                provider="ollama",
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                payload={"model": "m"},
            )
            await logger.log_response(provider="ollama", model="m", text="ok")
            await logger.close()
            text = path.read_text(encoding="utf-8")
            self.assertIn("llm_request #1", text)
            self.assertIn("|REQ-MESSAGES|", text)
            self.assertIn("|REQ-PRM|", text)
            self.assertIn("[MESSAGE]", text)
            self.assertIn("ok", text)
            self.assertIn("llm_response_end #1", text)


if __name__ == "__main__":
    unittest.main()
