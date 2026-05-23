from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from unified_local_llm_server.llm_logger import AsyncLLMLogger
from unified_local_llm_server.pipelines.json_fix import JsonFixPipeline
from unified_local_llm_server.pipelines.loop_guard import LoopGuardPipeline
from unified_local_llm_server.pipelines.tool import ToolPipeline
from unified_local_llm_server.provider_registry import ProviderRegistry
from unified_local_llm_server.providers import ProviderKind, resolve_provider_config
from unified_local_llm_server.pool import LocalLLM, LLMProviderPool


class FakeTransport:
    def __init__(self, responses: list[str | dict[str, Any]]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def post_json(self, path: str, payload: dict[str, Any]):
        self.calls.append({"path": path, "payload": payload})
        if not self.responses:
            raise AssertionError("No more fake responses")
        response = self.responses.pop(0)
        if isinstance(response, dict):
            message = {
                "role": "assistant",
                "content": response.get("content", ""),
            }
            if response.get("tool_calls"):
                message["tool_calls"] = response["tool_calls"]
        else:
            message = {"role": "assistant", "content": response}
        data = {
            "id": "fake-id",
            "model": payload.get("model"),
            "choices": [{"index": 0, "message": message}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

        class _FakeResult:
            status = 200
            headers: dict[str, str] = {}

            def json(self_nonlocal):
                return data

        return _FakeResult()

    def get_json_sync(self, path: str):
        return {"data": [{"id": "fake-model"}]}

    async def get_json(self, path: str):
        return {"data": [{"id": "fake-model"}]}


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


class SlowCountingTransport:
    def __init__(self, delay: float = 0.01):
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self.calls: list[dict[str, Any]] = []

    async def post_json(self, path: str, payload: dict[str, Any]):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.calls.append({"path": path, "payload": payload})
        try:
            await asyncio.sleep(self.delay)
            content = payload["messages"][-1]["content"]
        finally:
            self.active -= 1

        data = {
            "id": "fake-id",
            "model": payload.get("model"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

        class _FakeResult:
            status = 200
            headers: dict[str, str] = {}

            def json(self_nonlocal):
                return data

        return _FakeResult()

    async def get_json(self, path: str):
        return {"data": [{"id": "fake-model"}]}


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

    async def test_call_repairs_json_then_returns_dict(self):
        transport = FakeTransport([
            "```json\n{'name': 'Ana', 'age': 31,}\n```",
        ])
        server = LLMProviderPool(provider=ProviderKind.OLLAMA, transport=transport)
        llm = server.load_model(model="small-test-model")
        result = await llm.call(
            messages=[{"role": "user", "content": "Return a person"}],
            schema_dict={"name": str, "age": int},
            max_json_fix_retries=0,
        )
        self.assertEqual(result, {"name": "Ana", "age": 31})

    async def test_call_retries_after_loop_then_succeeds(self):
        transport = FakeTransport([
            ("repeat " * 600).strip(),
            json.dumps({"name": "Ana", "age": 31}),
        ])
        server = LLMProviderPool(provider=ProviderKind.LM_STUDIO, transport=transport)
        llm = server.load_model(model="small-test-model")
        result = await llm.call(
            messages=[{"role": "user", "content": "Return a person"}],
            schema_dict={"name": str, "age": int},
            max_json_fix_retries=1,
            max_loop_retries=1,
        )
        self.assertEqual(result, {"name": "Ana", "age": 31})
        self.assertGreaterEqual(len(transport.calls), 2)

    async def test_unstructured_call_uses_default_loop_retry_budget(self):
        transport = FakeTransport([
            ("repeat " * 600).strip(),
            "ok",
        ])
        server = LLMProviderPool(provider=ProviderKind.LM_STUDIO, transport=transport)
        llm = server.load_model(model="small-test-model")
        result = await llm.call(
            messages=[{"role": "user", "content": "Return plain text"}],
            max_json_fix_retries=0,
        )
        self.assertEqual(result, "ok")
        self.assertEqual(len(transport.calls), 2)

    async def test_unstructured_call_does_not_use_json_retry_budget_for_loop_limit(self):
        transport = FakeTransport([
            ("repeat " * 600).strip(),
            "ok",
        ])
        server = LLMProviderPool(provider=ProviderKind.LM_STUDIO, transport=transport)
        llm = server.load_model(model="small-test-model")
        with self.assertRaisesRegex(RuntimeError, "loop detected"):
            await llm.call(
                messages=[{"role": "user", "content": "Return plain text"}],
                max_json_fix_retries=5,
                max_loop_retries=0,
            )
        self.assertEqual(len(transport.calls), 1)

    async def test_loop_guard_uses_own_retry_budget(self):
        transport = FakeTransport([
            ("repeat " * 600).strip(),
            "ok",
        ])
        server = LLMProviderPool(provider=ProviderKind.LM_STUDIO, transport=transport)
        llm = server.load_model(model="small-test-model")
        result = await llm.call(
            messages=[{"role": "user", "content": "Return plain text"}],
            max_json_fix_retries=0,
            max_loop_retries=1,
        )
        self.assertEqual(result, "ok")
        self.assertEqual(len(transport.calls), 2)

    async def test_call_retries_after_empty_json_then_succeeds(self):
        transport = FakeTransport([
            "",
            json.dumps({"name": "Ana", "age": 31}),
        ])
        server = LLMProviderPool(provider=ProviderKind.LM_STUDIO, transport=transport)
        llm = server.load_model(model="small-test-model")
        result = await llm.call(
            messages=[{"role": "user", "content": "Return a person"}],
            schema_dict={"name": str, "age": int},
            max_json_fix_retries=1,
        )
        self.assertEqual(result, {"name": "Ana", "age": 31})
        retry_messages = transport.calls[1]["payload"]["messages"]
        self.assertIn("response was empty", retry_messages[-1]["content"])

    async def test_structured_empty_output_exhausts_json_fix_retries(self):
        transport = FakeTransport(["", "", "", ""])
        server = LLMProviderPool(provider=ProviderKind.LM_STUDIO, transport=transport)
        llm = server.load_model(model="small-test-model")
        with self.assertRaisesRegex(ValueError, "JSON fix failed after 4 attempt"):
            await llm.call(
                messages=[{"role": "user", "content": "Return a person"}],
                schema_dict={"name": str, "age": int},
                max_json_fix_retries=3,
            )
        self.assertEqual(len(transport.calls), 4)

    async def test_call_auto_selects_first_model(self):
        transport = FakeTransport(["hello"])
        server = LLMProviderPool(provider=ProviderKind.LM_STUDIO, transport=transport)
        # no model set — server resolves from /models endpoint
        model = await server.resolve_model()
        llm = server.load_model(model=model)
        result = await llm.call(
            messages=[{"role": "user", "content": "Say hello"}],
        )
        self.assertEqual(result, "hello")
        self.assertEqual(transport.calls[-1]["payload"]["model"], "fake-model")

    async def test_call_model_override_supports_multiple_loaded_models(self):
        transport = FakeTransport(["from model b"])
        server = LLMProviderPool(provider=ProviderKind.LM_STUDIO, transport=transport)
        llm_a = server.load_model(model="model-a")
        llm_b = server.load_model(model="model-b")
        result = await llm_b.call(
            messages=[{"role": "user", "content": "Say which model"}],
        )
        self.assertEqual(result, "from model b")
        self.assertEqual(llm_a.model, "model-a")
        self.assertEqual(transport.calls[0]["payload"]["model"], "model-b")

    async def test_load_model_returns_handle_with_defaults(self):
        transport = FakeTransport([
            "from handle",
        ])
        server = LLMProviderPool(
            provider=ProviderKind.OLLAMA,
            transport=transport,
        )
        llm = server.load_model(
            model="model-a",
            temperature=0.4,
            options={"num_predict": 100},
            context_length=4096,
        )
        self.assertIsInstance(llm, LocalLLM)
        result = await llm.call(
            messages=[{"role": "user", "content": "hello"}],
            options={"num_predict": 25},
        )
        self.assertEqual(result, "from handle")
        payload = transport.calls[0]["payload"]
        self.assertEqual(payload["model"], "model-a")
        self.assertEqual(payload["temperature"], 0.4)
        self.assertEqual(payload["options"]["num_predict"], 25)
        self.assertEqual(payload["options"]["num_ctx"], 4096)
        self.assertNotIn("max_tokens", payload)

    async def test_load_model_can_target_different_provider(self):
        transport = FakeTransport([
            "from lm studio",
        ])
        manager = LLMProviderPool(provider=ProviderKind.OLLAMA)
        llm = manager.load_model(
            ProviderKind.LM_STUDIO,
            "model-b",
            transport=transport,
            port=1234,
            temperature=0.1,
        )
        result = await llm.call(messages=[{"role": "user", "content": "hello"}])
        self.assertEqual(result, "from lm studio")
        self.assertEqual(llm.provider, ProviderKind.LM_STUDIO)
        self.assertEqual(llm.server.server_url, "http://127.0.0.1:1234")
        self.assertEqual(transport.calls[0]["payload"]["model"], "model-b")

    async def test_llm_batch_accepts_message_lists(self):
        transport = FakeTransport([
            "one",
            "two",
        ])
        server = LLMProviderPool(provider=ProviderKind.OLLAMA, transport=transport)
        llm = server.load_model("ollama", "model-a", temperature=0.3)

        results = await llm.batch(
            [
                [{"role": "user", "content": "first"}],
                [{"role": "user", "content": "second"}],
            ],
            concurrency=2,
        )

        self.assertEqual(results, ["one", "two"])
        self.assertEqual(transport.calls[0]["payload"]["model"], "model-a")
        self.assertEqual(transport.calls[0]["payload"]["temperature"], 0.3)
        self.assertEqual(transport.calls[1]["payload"]["messages"][-1]["content"], "second")

    async def test_llm_batch_accepts_per_item_call_kwargs(self):
        transport = FakeTransport([
            "first",
            "second",
        ])
        server = LLMProviderPool(provider=ProviderKind.OLLAMA, transport=transport)
        llm = server.load_model(
            "ollama",
            "model-a",
            options={"num_ctx": 4096, "num_predict": 100},
        )

        results = await llm.batch(
            [
                {
                    "messages": [{"role": "user", "content": "first"}],
                    "temperature": 0.1,
                },
                {
                    "messages": [{"role": "user", "content": "second"}],
                    "options": {"num_predict": 25},
                },
            ],
            temperature=0.7,
            options={"top_p": 0.9},
        )

        self.assertEqual(results, ["first", "second"])
        first_payload = transport.calls[0]["payload"]
        second_payload = transport.calls[1]["payload"]
        self.assertEqual(first_payload["temperature"], 0.1)
        self.assertEqual(first_payload["options"]["top_p"], 0.9)
        self.assertEqual(first_payload["options"]["num_ctx"], 4096)
        self.assertEqual(second_payload["temperature"], 0.7)
        self.assertEqual(second_payload["options"]["num_predict"], 25)
        self.assertEqual(second_payload["options"]["num_ctx"], 4096)
        self.assertNotIn("max_tokens", second_payload)

    async def test_llm_batch_limits_concurrency(self):
        transport = SlowCountingTransport(delay=0.01)
        server = LLMProviderPool(provider=ProviderKind.OLLAMA, transport=transport)
        llm = server.load_model("ollama", "model-a")

        results = await llm.batch(
            [
                [{"role": "user", "content": "one"}],
                [{"role": "user", "content": "two"}],
                [{"role": "user", "content": "three"}],
                [{"role": "user", "content": "four"}],
            ],
            concurrency=2,
        )

        self.assertEqual(results, ["one", "two", "three", "four"])
        self.assertLessEqual(transport.max_active, 2)

    async def test_llm_batch_can_return_exceptions(self):
        transport = FakeTransport([
            "not json",
            json.dumps({"name": "Ana", "age": 31}),
        ])
        server = LLMProviderPool(provider=ProviderKind.OLLAMA, transport=transport)
        llm = server.load_model("ollama", "model-a")

        results = await llm.batch(
            [
                [{"role": "user", "content": "bad"}],
                [{"role": "user", "content": "good"}],
            ],
            concurrency=1,
            return_exceptions=True,
            schema_dict={"name": str, "age": int},
            max_json_fix_retries=0,
        )

        self.assertIsInstance(results[0], Exception)
        self.assertEqual(results[1], {"name": "Ana", "age": 31})

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
                transport=FakeTransport([]),
            )
            ollama_models = server.list_loaded_models("ollama")
            lmstudio_models = server.list_loaded_models("lmstudio")
            self.assertEqual(ollama_models[0], "ollama-a")
            self.assertEqual(lmstudio_models[0], "lmstudio-b")

    async def test_provider_check_uses_endpoint_config(self):
        transport = FakeTransport([])
        server = LLMProviderPool(
            provider=ProviderKind.LM_STUDIO,
            transport=transport,
        )
        status = await server.check_provider()
        self.assertTrue(status["ok"])
        self.assertEqual(status["server_url"], "http://127.0.0.1:1234")

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

    async def test_mock_llm_call_writes_persistent_response_log(self):
        path = Path("test_logs") / "llm_response_mock.log"
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            path.unlink()

        logger = AsyncLLMLogger(path)
        transport = FakeTransport([
            json.dumps({"name": "Ana", "age": 31}),
        ])
        server = LLMProviderPool(
            provider=ProviderKind.OLLAMA,
            transport=transport,
        )
        llm = server.load_model(model="small-test-model", logger=logger)
        result = await llm.call(
            messages=[{"role": "user", "content": "Return a person"}],
            schema_dict={"name": str, "age": int},
            max_json_fix_retries=0,
        )
        await server.close()

        text = path.read_text(encoding="utf-8")
        self.assertEqual(result, {"name": "Ana", "age": 31})
        self.assertIn("llm_request #1", text)
        self.assertIn("|REQ-MESSAGES|", text)
        self.assertIn("|REQ-PRM|", text)
        self.assertIn("[MESSAGE]", text)
        self.assertIn('{"name": "Ana", "age": 31}', text)
        self.assertIn('|INFO| parsed={"name": "Ana", "age": 31}', text)
        self.assertIn("llm_response_end #1", text)

    async def test_call_runs_tool_loop_before_final_answer(self):
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "add", "arguments": "{\"a\": 2, \"b\": 3}"},
            }
        ]
        transport = FakeTransport([
            {"content": "", "tool_calls": tool_calls},
            "The sum is 5.",
        ])

        def add(arguments: dict[str, Any]) -> dict[str, int]:
            return {"sum": arguments["a"] + arguments["b"]}

        server = LLMProviderPool(provider=ProviderKind.OLLAMA, transport=transport)
        llm = server.load_model(model="small-test-model")
        result = await llm.call(
            messages=[{"role": "user", "content": "Add 2 and 3"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "add",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "integer"},
                                "b": {"type": "integer"},
                            },
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
            tool_registry={"add": add},
            max_tool_rounds=2,
        )
        self.assertEqual(result, "The sum is 5.")
        self.assertEqual(len(transport.calls), 2)
        second_messages = transport.calls[1]["payload"]["messages"]
        self.assertEqual(second_messages[-1]["role"], "tool")
        self.assertEqual(second_messages[-1]["name"], "add")
        self.assertEqual(json.loads(second_messages[-1]["content"]), {"sum": 5})


if __name__ == "__main__":
    unittest.main()
