from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .providers import ProviderConfig, resolve_provider_config


def _load_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml  # type: ignore
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _find_config(cwd: Path) -> Path | None:
    for name in ("providers.yaml", "providers.yml", "providers.json"):
        p = cwd / name
        if p.exists():
            return p
    return None


@dataclass(slots=True)
class ProviderRegistryEntry:
    provider: str
    host: str | None = None
    port: int | None = None
    api_key: str | None = None
    base_path: str | None = None
    health_path: str | None = None
    models_path: str | None = None
    systemd_service: str | None = None
    def to_provider_config(self) -> ProviderConfig:
        return resolve_provider_config(
            self.provider,
            host=self.host,
            port=self.port,
            api_key=self.api_key,
            base_path=self.base_path,
            health_path=self.health_path,
            models_path=self.models_path,
        )


class ProviderRegistry:
    def __init__(self, entries: dict[str, ProviderRegistryEntry]):
        self.entries = entries

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ProviderRegistry":
        raw_path = path or os.environ.get("LOCAL_LLM_PROVIDERS")
        if raw_path is not None:
            path_obj = Path(raw_path)
        else:
            path_obj = _find_config(Path.cwd())
        if path_obj is None or not path_obj.exists():
            return cls({})

        data = _load_file(path_obj)
        entries: dict[str, ProviderRegistryEntry] = {}
        for name, item in data.get("providers", {}).items():
            if not isinstance(item, dict):
                continue
            # The object key is the provider kind by default. Use "provider"
            # only when a friendly alias maps to a different provider kind.
            provider = item.get("provider", name)
            entries[name] = ProviderRegistryEntry(
                provider=str(provider),
                host=item.get("host"),
                port=int(item["port"]) if item.get("port") is not None else None,
                api_key=item.get("api_key"),
                base_path=item.get("base_path"),
                health_path=item.get("health_path"),
                models_path=item.get("models_path"),
                systemd_service=item.get("systemd_service"),
            )
        return cls(entries)

    def get(self, name: str) -> ProviderRegistryEntry:
        try:
            return self.entries[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.entries)) or "(none)"
            raise KeyError(f"Unknown provider config {name!r}. Available: {available}") from exc

    def names(self) -> list[str]:
        return sorted(self.entries)

    def canonical_names(self) -> list[str]:
        names = set(self.entries)
        for alias, entry in self.entries.items():
            if alias != entry.provider:
                names.discard(alias)
                names.add(entry.provider)
        return sorted(names)


def provider_entry_to_server_kwargs(
    entry: ProviderRegistryEntry,
    *,
    model: str | None = None,
) -> dict[str, Any]:
    return {
        "provider": entry.provider,
        "model": model,
        "host": entry.host,
        "port": entry.port,
        "api_key": entry.api_key,
        "base_path": entry.base_path,
        "health_path": entry.health_path,
        "models_path": entry.models_path,
    }
