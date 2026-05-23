"""Provider kind constants, connection config, and per-provider defaults.

Responsibility
--------------
Knows what providers exist and what their default connection parameters are.
Nothing here performs I/O — this module only describes topology.

Layer position
--------------
Core domain layer.  Imported by ``provider_registry.py`` and ``pool.py``;
has no internal imports.
"""
from __future__ import annotations

from dataclasses import dataclass
import os


class ProviderKind:
    """String constants for the four supported provider kinds.

    Used throughout the codebase as ``ProviderKind.OLLAMA`` etc. rather than
    raw strings so typos are caught at import time.
    """

    LM_STUDIO = "lm_studio"
    OLLAMA    = "ollama"
    UNSLOTH   = "unsloth"
    LLAMA_CPP = "llama_cpp"


@dataclass(slots=True)
class ProviderConfig:
    """Fully resolved connection config for one provider instance.

    Produced by ``resolve_provider_config``; consumed by ``LLMProviderPool``
    and ``OpenAICompatibleTransport``.

    Attributes
    ----------
    name:
        Canonical provider kind string (e.g. ``"ollama"``).
    host:
        Hostname or IP the provider listens on.
    port:
        TCP port.
    api_key:
        Bearer token for the ``Authorization`` header; ``None`` when the
        provider does not require authentication.
    base_path:
        Path prefix for inference endpoints (``/v1`` for all OpenAI-compat
        providers).
    health_path:
        Optional path polled to determine liveness; ``None`` means the
        provider offers no dedicated health endpoint.
    models_path:
        Path used to enumerate available models.
    """

    name: str
    host: str = "127.0.0.1"
    port: int = 0
    api_key: str | None = None
    base_path: str = "/v1"
    health_path: str | None = None
    models_path: str = "/v1/models"

    @property
    def server_url(self) -> str:
        """Root URL without path, e.g. ``http://127.0.0.1:11434``."""
        return f"http://{self.host}:{self.port}"

    @property
    def base_url(self) -> str:
        """Inference base URL including ``base_path``, e.g. ``http://127.0.0.1:11434/v1``."""
        return f"{self.server_url}{self.base_path}"


# Per-provider defaults applied when the registry entry omits a field.
# Unsloth's API key is read from the environment at import time so it is
# picked up from the shell without requiring providers.yaml to contain it.
_DEFAULTS: dict[str, dict[str, object]] = {
    ProviderKind.LM_STUDIO: {
        "port": 1234,
        "api_key": "lm-studio",
        "health_path": None,
        "models_path": "/v1/models",
    },
    ProviderKind.OLLAMA: {
        "port": 11434,
        "api_key": "ollama",
        "health_path": "/api/version",
        "models_path": "/v1/models",
    },
    ProviderKind.UNSLOTH: {
        "port": 8895,
        "api_key": os.environ.get("UNSLOTH_KEY") or os.environ.get("UNSLOTH_API_KEY") or "unsloth",
        "health_path": "/health",
        "models_path": "/v1/models",
    },
    ProviderKind.LLAMA_CPP: {
        "port": 8080,
        "api_key": "no-key",
        "health_path": "/health",
        "models_path": "/v1/models",
    },
}


def known_provider_kinds() -> list[str]:
    """Return the sorted list of built-in provider kind strings."""
    return sorted(_DEFAULTS)


def resolve_provider_config(
    provider: str,
    *,
    host: str | None = None,
    port: int | None = None,
    api_key: str | None = None,
    base_path: str | None = None,
    health_path: str | None = None,
    models_path: str | None = None,
) -> ProviderConfig:
    """Build a :class:`ProviderConfig` by merging caller overrides with built-in defaults.

    Unknown provider kinds are accepted — they get generic defaults so custom
    or future providers work without changes to this module.

    Parameters
    ----------
    provider:
        Provider kind string; normalised to lowercase before lookup.
    host, port, api_key, base_path, health_path, models_path:
        Explicit values; ``None`` means "use the built-in default for this
        provider kind".
    """
    key = provider.strip().lower()
    defaults = _DEFAULTS.get(key, {})
    resolved_host         = host or "127.0.0.1"
    resolved_port         = int(port if port is not None else defaults.get("port", 8080))
    resolved_key          = api_key if api_key is not None else defaults.get("api_key")
    resolved_base_path    = base_path or "/v1"
    resolved_health_path  = health_path if health_path is not None else defaults.get("health_path")
    resolved_models_path  = models_path or str(defaults.get("models_path", "/v1/models"))
    return ProviderConfig(
        name=key,
        host=resolved_host,
        port=resolved_port,
        api_key=str(resolved_key) if resolved_key is not None else None,
        base_path=resolved_base_path,
        health_path=str(resolved_health_path) if resolved_health_path is not None else None,
        models_path=resolved_models_path,
    )
