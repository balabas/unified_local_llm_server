from __future__ import annotations

from dataclasses import dataclass
import os


class ProviderKind:
    LM_STUDIO = "lm_studio"
    OLLAMA = "ollama"
    UNSLOTH = "unsloth"
    LLAMA_CPP = "llama_cpp"


@dataclass(slots=True)
class ProviderConfig:
    name: str
    host: str = "127.0.0.1"
    port: int = 0
    api_key: str | None = None
    base_path: str = "/v1"
    health_path: str | None = None
    models_path: str = "/v1/models"

    @property
    def server_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def base_url(self) -> str:
        return f"{self.server_url}{self.base_path}"


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
    key = provider.strip().lower()
    defaults = _DEFAULTS.get(key, {})
    resolved_host = host or "127.0.0.1"
    resolved_port = int(port if port is not None else defaults.get("port", 8080))
    resolved_key = api_key if api_key is not None else defaults.get("api_key")
    resolved_base_path = base_path or "/v1"
    resolved_health_path = health_path if health_path is not None else defaults.get("health_path")
    resolved_models_path = models_path or str(defaults.get("models_path", "/v1/models"))
    return ProviderConfig(
        name=key,
        host=resolved_host,
        port=resolved_port,
        api_key=str(resolved_key) if resolved_key is not None else None,
        base_path=resolved_base_path,
        health_path=str(resolved_health_path) if resolved_health_path is not None else None,
        models_path=resolved_models_path,
    )
