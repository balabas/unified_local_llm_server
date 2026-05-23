from __future__ import annotations

import asyncio
import argparse
from dataclasses import dataclass

from unified_local_llm_server.provider_registry import ProviderRegistry, provider_entry_to_server_kwargs
from unified_local_llm_server.providers import ProviderKind
from unified_local_llm_server.pool import LLMProviderPool


@dataclass(slots=True)
class SmokeCase:
    provider: str
    model: str
    prompt: str


CASES = [
    SmokeCase(ProviderKind.OLLAMA, "", "Write one short sentence about the sea."),
    SmokeCase(ProviderKind.LM_STUDIO, "", "Return JSON with keys name and age."),
    SmokeCase(ProviderKind.LLAMA_CPP, "", "Explain in one line why loops are bad."),
    SmokeCase(ProviderKind.UNSLOTH, "", "Summarize in one sentence."),
]


async def run_default_cases() -> None:
    for case in CASES:
        server = LLMProviderPool(provider=case.provider, model=case.model or None)
        try:
            status = await server.check_provider()
            if not status["ok"]:
                print(f"[{case.provider}] skipped: {status['server_url']} not ready ({status.get('error')})")
                continue
            model = await server.resolve_call_model(None)
            llm = server.load_model(model=model)
            result = await llm.call(
                messages=[{"role": "user", "content": case.prompt}],
            )
            print(f"[{case.provider}] {model}: {result}")
        except Exception as exc:
            print(f"[{case.provider}] {case.model}: skipped ({exc})")


async def run_provider_case(
    registry_path: str | None,
    provider_name: str,
    model: str | None,
    prompt: str,
    status_only: bool,
) -> None:
    registry = ProviderRegistry.load(registry_path)
    entry = registry.get(provider_name)
    server = LLMProviderPool(**provider_entry_to_server_kwargs(entry, model=model))
    status = await server.check_provider()
    print(f"[{provider_name}] {status['server_url']} ready={status['ok']} check={status['kind']}")
    if not status["ok"]:
        raise RuntimeError(f"Provider {provider_name!r} is not ready at {status['server_url']}: {status.get('error')}")
    if status_only:
        return
    resolved_model = model or await server.resolve_call_model(None)
    llm = server.load_model(model=resolved_model)
    result = await llm.call(
        messages=[{"role": "user", "content": prompt}],
    )
    print(result)


async def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", help="Path to providers.json")
    parser.add_argument("--provider-name", help="Named provider config from registry")
    parser.add_argument("--model", help="Model id to use. If omitted, first /v1/models id is used.")
    parser.add_argument("--prompt", default="Return one short sentence.")
    parser.add_argument("--status-only", action="store_true", help="Check provider port readiness without generation.")
    args = parser.parse_args()

    if args.provider_name:
        await run_provider_case(args.registry, args.provider_name, args.model, args.prompt, args.status_only)
    else:
        await run_default_cases()


if __name__ == "__main__":
    asyncio.run(run())
