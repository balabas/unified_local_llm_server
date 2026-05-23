"""Provider registry — config file loading and provider lookup.

Responsibility
--------------
Parses ``providers.yaml`` (or ``.yml`` / ``.json``) and exposes a dict-like
registry of :class:`ProviderRegistryEntry` objects keyed by provider name.
Also supports friendly aliases: a registry entry whose key differs from its
``provider`` field is treated as an alias for the canonical provider kind.

The registry is read-only after construction.  ``LLMProviderPool`` holds one
instance and calls ``get()`` / ``canonical_names()`` during pool operations.

Layer position
--------------
Configuration layer, one step above ``providers.py``.
Imports: ``providers`` (for ``resolve_provider_config``).
Used by: ``pool.py``.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .providers import ProviderConfig, resolve_provider_config


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_file(path: Path) -> dict[str, Any]:
    """Read and parse a YAML or JSON config file."""
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml  # type: ignore
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _find_config(cwd: Path) -> Path | None:
    """Search ``cwd`` for a providers config file in priority order."""
    for name in ("providers.yaml", "providers.yml", "providers.json"):
        p = cwd / name
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ProviderRegistryEntry:
    """Raw config entry for one provider as read from the config file.

    All connection fields are optional; ``None`` means "use the built-in
    default for this provider kind" (resolved by ``resolve_provider_config``).

    Attributes
    ----------
    provider:
        Canonical provider kind (e.g. ``"ollama"``).  Equals the registry key
        for normal entries; differs for aliases (e.g. key ``"my_ollama"``
        with ``provider="ollama"``).
    systemd_service:
        Service unit name used by ``LLMProviderPool.restart()``.  ``None``
        when restart support is not needed.
    """

    provider: str
    host: str | None = None
    port: int | None = None
    api_key: str | None = None
    base_path: str | None = None
    health_path: str | None = None
    models_path: str | None = None
    systemd_service: str | None = None

    def to_provider_config(self) -> ProviderConfig:
        """Resolve this entry into a fully-populated :class:`ProviderConfig`."""
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
    """Immutable registry of provider entries loaded from a config file.

    Construction
    ------------
    Use :meth:`load` — do not instantiate directly.  Returns an empty registry
    (not an error) when no config file is found, so code that works with a
    single provider without a config file still functions.

    Alias support
    -------------
    A registry entry whose key differs from its ``provider`` field is an alias.
    :meth:`canonical_names` returns only the resolved provider kinds, stripping
    alias keys so the pool doesn't surface duplicate entries.
    """

    def __init__(self, entries: dict[str, ProviderRegistryEntry]):
        self.entries = entries

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ProviderRegistry":
        """Load the registry from a config file.

        Resolution order:
        1. ``path`` argument
        2. ``LOCAL_LLM_PROVIDERS`` environment variable
        3. Auto-discovery in the current working directory
           (``providers.yaml`` → ``providers.yml`` → ``providers.json``)

        Returns an empty registry when no file is found.
        """
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
            # The object key is the provider kind by default.  A "provider"
            # sub-key allows a friendly alias to map to a different kind.
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
        """Return the entry for ``name``, or raise ``KeyError`` with helpful context."""
        try:
            return self.entries[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.entries)) or "(none)"
            raise KeyError(f"Unknown provider config {name!r}. Available: {available}") from exc

    def names(self) -> list[str]:
        """All registered names including aliases, sorted."""
        return sorted(self.entries)

    def canonical_names(self) -> list[str]:
        """Provider kind strings with alias keys removed, sorted.

        Ensures the pool's provider list doesn't contain both ``"my_ollama"``
        and ``"ollama"`` when one is an alias for the other.
        """
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
    """Convert a registry entry to ``**kwargs`` for ``LLMProviderPool.__init__``.

    Used by ``LLMProviderPool.get_provider`` to construct a child pool from a
    registry entry without duplicating the field mapping in two places.
    """
    return {
        "provider":     entry.provider,
        "model":        model,
        "host":         entry.host,
        "port":         entry.port,
        "api_key":      entry.api_key,
        "base_path":    entry.base_path,
        "health_path":  entry.health_path,
        "models_path":  entry.models_path,
    }
