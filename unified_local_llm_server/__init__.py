"""unified_local_llm_server — public package surface.

Responsibility
--------------
Re-exports the symbols that external code should import.  Nothing is
implemented here; this file is purely a curated import list.

Layer position
--------------
Top of the dependency graph.  All other modules are internal; callers
should only need to import from this package root.

Typical usage::

    from unified_local_llm_server import LLMProviderPool
    pool = LLMProviderPool("providers.yaml")
"""
from .client import LocalAsyncOpenAI, LocalChatCompletion, LocalChatCompletions
from .provider_registry import ProviderRegistry, ProviderRegistryEntry
from .providers import ProviderConfig, ProviderKind, known_provider_kinds, resolve_provider_config
from .pool import LocalLLM, LLMProviderPool
from .transport import TransportError

LocalLLMServer = LLMProviderPool  # backward-compat alias

__all__ = [
    "LocalAsyncOpenAI",
    "LocalChatCompletion",
    "LocalChatCompletions",
    "LocalLLM",
    "LLMProviderPool",
    "LocalLLMServer",
    "ProviderConfig",
    "ProviderKind",
    "ProviderRegistry",
    "ProviderRegistryEntry",
    "TransportError",
    "known_provider_kinds",
    "resolve_provider_config",
]
