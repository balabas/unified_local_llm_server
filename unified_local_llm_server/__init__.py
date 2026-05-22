from .client import LocalAsyncOpenAI, LocalChatCompletion, LocalChatCompletions
from .provider_registry import ProviderRegistry, ProviderRegistryEntry
from .providers import ProviderConfig, ProviderKind, known_provider_kinds, resolve_provider_config
from .server import LocalLLM, LLMProviderPool
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
