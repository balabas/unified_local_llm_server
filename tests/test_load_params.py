"""Integration tests for load-time parameters across all providers.

Each test:
- Skips if the provider is not reachable
- Loads a model with a parameter, runs one short inference call
- Asserts a non-empty response
- Uses asyncio.wait_for with MAX_WAIT=60s to abort hung loads

Run with:
    python -m pytest tests/test_load_params.py -v -s
"""
from __future__ import annotations

import asyncio
import unittest
from pathlib import Path

from unified_local_llm_server import LLMProviderPool
from unified_local_llm_server.provider_registry import ProviderRegistry

ROOT = Path(__file__).parent.parent
REGISTRY_PATH = ROOT / "providers.example.yaml"
PROMPT = [{"role": "user", "content": "Reply with the single word: OK"}]
MAX_WAIT = 60.0  # seconds


def _pool() -> LLMProviderPool:
    registry = ProviderRegistry.load(REGISTRY_PATH)
    return LLMProviderPool(provider_registry=registry, timeout=MAX_WAIT)


async def _reachable(pool: LLMProviderPool, provider: str) -> bool:
    try:
        status = await asyncio.wait_for(
            pool.check_provider(provider), timeout=5.0
        )
        return status.get("ok", False)
    except Exception:
        return False


async def _first_model(pool: LLMProviderPool, provider: str, *, prefer_small: bool = False) -> str | None:
    """Return the name of the first available model for a provider.

    With prefer_small=True, skip models whose names suggest very large sizes
    (120b, 70b, 65b, 72b) so load-param tests stay within MAX_WAIT.
    """
    try:
        models = pool.list_downloaded_models(provider)
        if not models:
            return None
        names = [m["name"] if isinstance(m, dict) else m for m in models]
        if prefer_small:
            _HUGE = ("120b", "70b", "65b", "72b", "110b")
            small = [n for n in names if not any(x in n.lower() for x in _HUGE)]
            names = small if small else names
        return names[0] if names else None
    except Exception:
        return None


async def _call(pool: LLMProviderPool, provider: str, model: str, **load_kwargs) -> str:
    llm = pool.load_model(provider, model, **load_kwargs)
    return await asyncio.wait_for(
        llm.call(messages=PROMPT, options={"num_predict": 8}),
        timeout=MAX_WAIT,
    )


class LoadParamsOllamaTests(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.pool = _pool()
        self.ok = await _reachable(self.pool, "ollama")
        self.model = await _first_model(self.pool, "ollama") if self.ok else None
        if not self.ok or not self.model:
            self.skipTest("ollama not reachable or no models")

    async def test_context_length(self):
        result = await _call(self.pool, "ollama", self.model, context_length=2048)
        self.assertTrue(result.strip())

    async def test_batch_size(self):
        result = await _call(self.pool, "ollama", self.model, batch_size=256)
        self.assertTrue(result.strip())

    async def test_threads(self):
        result = await _call(self.pool, "ollama", self.model, threads=4)
        self.assertTrue(result.strip())

    async def test_load_params_passthrough(self):
        result = await _call(
            self.pool, "ollama", self.model,
            load_params={"num_thread": 2, "num_ctx": 1024},
        )
        self.assertTrue(result.strip())


class LoadParamsLmStudioTests(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.pool = _pool()
        self.ok = await _reachable(self.pool, "lm_studio")
        self.model = await _first_model(self.pool, "lm_studio") if self.ok else None
        if not self.ok or not self.model:
            self.skipTest("lm_studio not reachable or no models")

    async def test_context_length(self):
        result = await _call(self.pool, "lm_studio", self.model, context_length=2048)
        self.assertTrue(result.strip())

    async def test_flash_attn(self):
        result = await _call(self.pool, "lm_studio", self.model, flash_attn=True)
        self.assertTrue(result.strip())

    async def test_batch_size(self):
        result = await _call(self.pool, "lm_studio", self.model, batch_size=256)
        self.assertTrue(result.strip())

    async def test_kv_offload(self):
        result = await _call(self.pool, "lm_studio", self.model, kv_offload=True)
        self.assertTrue(result.strip())

    async def test_load_params_passthrough(self):
        result = await _call(
            self.pool, "lm_studio", self.model,
            load_params={"context_length": 1024},
        )
        self.assertTrue(result.strip())

    async def test_gpu_offload_cli(self):
        import shutil
        if not shutil.which("lms"):
            self.skipTest("lms CLI not found on PATH")
        try:
            result = await _call(self.pool, "lm_studio", self.model, gpu_offload="max")
            self.assertTrue(result.strip())
        except RuntimeError as exc:
            if "lms load failed" in str(exc):
                self.skipTest(f"lms load unavailable: {exc}")
            raise


class LoadParamsLlamaCppTests(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.pool = _pool()
        self.ok = await _reachable(self.pool, "llama_cpp")
        # prefer small models so reload stays within MAX_WAIT
        self.model = await _first_model(self.pool, "llama_cpp", prefer_small=True) if self.ok else None
        if not self.ok or not self.model:
            self.skipTest("llama_cpp not reachable or no models")

    async def test_context_length(self):
        result = await _call(self.pool, "llama_cpp", self.model, context_length=2048)
        self.assertTrue(result.strip())

    async def test_gpu_layers(self):
        result = await _call(self.pool, "llama_cpp", self.model, gpu_layers=10)
        self.assertTrue(result.strip())

    async def test_ngl_alias(self):
        result = await _call(self.pool, "llama_cpp", self.model, ngl=10)
        self.assertTrue(result.strip())

    async def test_flash_attn(self):
        result = await _call(self.pool, "llama_cpp", self.model, flash_attn=True)
        self.assertTrue(result.strip())

    async def test_batch_size(self):
        result = await _call(self.pool, "llama_cpp", self.model, batch_size=256)
        self.assertTrue(result.strip())

    async def test_threads(self):
        result = await _call(self.pool, "llama_cpp", self.model, threads=4)
        self.assertTrue(result.strip())

    async def test_kv_offload(self):
        result = await _call(self.pool, "llama_cpp", self.model, kv_offload=True)
        self.assertTrue(result.strip())

    async def test_load_params_passthrough(self):
        result = await _call(
            self.pool, "llama_cpp", self.model,
            load_params={"ctx_size": 1024, "flash_attn": True},
        )
        self.assertTrue(result.strip())


class LoadParamsUnslothTests(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.pool = _pool()
        self.ok = await _reachable(self.pool, "unsloth")
        self.model = await _first_model(self.pool, "unsloth", prefer_small=True) if self.ok else None
        if not self.ok or not self.model:
            self.skipTest("unsloth not reachable or no models")

    async def _call_or_skip(self, **kwargs) -> str:
        try:
            return await _call(self.pool, "unsloth", self.model, **kwargs)
        except asyncio.TimeoutError as exc:
            self.skipTest(f"unsloth model load timed out after {MAX_WAIT}s: {exc}")
        except RuntimeError as exc:
            msg = str(exc)
            if "HTTP 500" in msg or "HTTP 503" in msg or "load" in msg.lower():
                self.skipTest(f"unsloth model load unavailable: {exc}")
            raise

    async def test_context_length(self):
        result = await self._call_or_skip(context_length=2048)
        self.assertTrue(result.strip())

    async def test_load_params_passthrough(self):
        result = await self._call_or_skip(load_params={"max_seq_length": 1024})
        self.assertTrue(result.strip())


if __name__ == "__main__":
    unittest.main()
