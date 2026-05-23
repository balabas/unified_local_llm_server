"""
Inference interface test across all providers.
Runs a standard prompt on each available provider, unloading before switching.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from unified_local_llm_server.provider_registry import ProviderRegistry
from unified_local_llm_server.pool import LLMProviderPool

ROOT = Path(__file__).parent.parent
REGISTRY_PATH = ROOT / "providers.example.yaml"

PREFERRED_MODELS = {
    "ollama": "gpt-oss:20b",
    "lm_studio": "google/gemma-4-e4b",
    "unsloth": "unsloth/gpt-oss-20b-GGUF@UD-Q4_K_XL",
    "llama_cpp": "gpt-oss-20b-MXFP4",
}

PROMPT = "Return one short sentence about why provider abstraction is useful for local LLMs."


def sep(provider: str, model: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {provider} / {model}")
    print(f"{'='*60}")


async def run() -> None:
    registry = ProviderRegistry.load(REGISTRY_PATH)
    server = LLMProviderPool(provider_registry=registry)

    providers = server.get_providers()
    print(f"Providers: {providers}")

    # Check which providers are up
    statuses: dict[str, dict] = {}
    for name in providers:
        statuses[name] = await server.check_provider(name)
        ok = statuses[name]["ok"]
        url = statuses[name]["server_url"]
        print(f"  {name}: {'OK' if ok else 'DOWN'} ({url})")

    previous_provider: str | None = None

    for provider in providers:
        status = statuses[provider]
        if not status["ok"]:
            print(f"\n[{provider}] SKIP — not reachable")
            continue

        provider_server = server.get_provider(provider)

        # Unload previous provider's model before switching
        if previous_provider and previous_provider != provider:
            print(f"\n[{previous_provider}] unloading...")
            try:
                result = server.unload_all_models(previous_provider)
                print(f"[{previous_provider}] unloaded: {result}")
            except Exception as exc:
                print(f"[{previous_provider}] unload error (continuing): {exc}")

        # Resolve model
        preferred = PREFERRED_MODELS.get(provider)
        try:
            model = preferred or await provider_server.resolve_call_model(None)
        except Exception as exc:
            print(f"\n[{provider}] SKIP — could not resolve model: {exc}")
            continue

        sep(provider, model)

        llm = server.load_model(provider, model)

        try:
            result = await llm.call(
                messages=[{"role": "user", "content": PROMPT}],
                options={"num_predict": 128},
            )
            print(f"Response: {result}")
            previous_provider = provider
        except Exception as exc:
            print(f"ERROR: {exc}")

    # Unload last provider
    if previous_provider:
        print(f"\n[{previous_provider}] unloading (cleanup)...")
        try:
            result = server.unload_all_models(previous_provider)
            print(f"[{previous_provider}] unloaded: {result}")
        except Exception as exc:
            print(f"[{previous_provider}] unload error: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(run())
