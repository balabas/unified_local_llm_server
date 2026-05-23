"""HTTP transport layer for OpenAI-compatible provider endpoints.

Responsibility
--------------
Owns all network I/O for the library.  Every outbound HTTP call — whether a
JSON request/response or a streaming SSE/NDJSON response — goes through this
module.  Higher-level code (pool.py) never touches ``urllib`` or sockets
directly; it only calls methods on :class:`OpenAICompatibleTransport`.

The implementation is intentionally stdlib-only (``urllib``) so the library
has no mandatory network dependency beyond Python itself.  Async methods
offload blocking I/O to a thread pool via ``asyncio.to_thread`` so the event
loop is never blocked.

Public surface
--------------
- :class:`HttpResult`          — plain response container (status, headers, body)
- :class:`TransportError`      — raised when a request cannot be completed
- :class:`OpenAICompatibleTransport` — the main transport class
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest


@dataclass(slots=True)
class HttpResult:
    """Parsed HTTP response returned by every non-streaming transport call.

    Attributes
    ----------
    status:
        HTTP status code (e.g. 200, 404, 500).
    headers:
        Response headers as a plain ``{name: value}`` dict.
    text:
        Full response body decoded as UTF-8.
    """

    status: int
    headers: dict[str, str]
    text: str

    def json(self) -> Any:
        """Deserialize the body as JSON.

        Returns ``None`` when the status is 4xx/5xx, the body is empty, or
        parsing fails — callers that need to distinguish these cases should
        check ``status`` directly before calling this method.
        """
        if self.status >= 400 or not self.text.strip():
            return None
        try:
            return json.loads(self.text)
        except json.JSONDecodeError:
            return None


class TransportError(RuntimeError):
    """Raised when a request fails at the network or OS level.

    HTTP error responses (4xx / 5xx) are *not* raised as ``TransportError``;
    they are returned as :class:`HttpResult` with the appropriate status code
    so callers can inspect the body.  ``TransportError`` is reserved for
    connection failures, timeouts, and other I/O exceptions.
    """


class OpenAICompatibleTransport:
    """Thin HTTP client for OpenAI-compatible REST endpoints.

    Wraps Python's stdlib ``urllib`` so the library needs no third-party HTTP
    dependency.  All blocking calls are synchronous internally; async wrappers
    (``get_json``, ``post_json``) delegate to ``asyncio.to_thread``.

    Parameters
    ----------
    base_url:
        Root URL of the provider, e.g. ``"http://127.0.0.1:11434"``.
        A trailing slash is stripped automatically.
    timeout:
        Default socket timeout in seconds applied to every request.
        Individual calls can override it via the ``timeout`` kwarg.
    api_key:
        Optional Bearer token sent in the ``Authorization`` header.
    """

    def __init__(self, base_url: str, *, timeout: float = 300.0, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key

    def _headers(self) -> dict[str, str]:
        """Build request headers, injecting Bearer auth when an API key is set."""
        headers = {"Content-Type": "application/json"}
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # ------------------------------------------------------------------
    # Synchronous helpers — called directly by model-management code and
    # by the async wrappers below via asyncio.to_thread.
    # ------------------------------------------------------------------

    def get_json_sync(self, path: str, *, timeout: float | None = None) -> Any:
        """GET ``path`` and return the parsed JSON body, or ``None`` on error."""
        return self._do_request("GET", path, None, timeout=timeout).json()

    def post_json_sync(self, path: str, payload: dict[str, Any], *, timeout: float | None = None) -> HttpResult:
        """POST ``payload`` to ``path`` and return the raw :class:`HttpResult`.

        The caller is responsible for checking ``result.status`` and
        ``result.json()`` — this method never raises on HTTP errors.
        """
        return self._do_request("POST", path, payload, timeout=timeout)

    # ------------------------------------------------------------------
    # Async wrappers — used by pool._call() and check_provider()
    # ------------------------------------------------------------------

    async def get_json(self, path: str) -> Any:
        """Async GET — offloads the blocking call to a thread pool."""
        result = await asyncio.to_thread(self._do_request, "GET", path, None)
        return result.json()

    async def post_json(self, path: str, payload: dict[str, Any]) -> HttpResult:
        """Async POST — offloads the blocking call to a thread pool."""
        return await asyncio.to_thread(self._do_request, "POST", path, payload)

    # ------------------------------------------------------------------
    # Core request machinery
    # ------------------------------------------------------------------

    def _do_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        *,
        timeout: float | None = None,
    ) -> HttpResult:
        """Execute a single HTTP request and return an :class:`HttpResult`.

        HTTP error responses (4xx / 5xx) are captured and returned as an
        ``HttpResult`` with the error body preserved so callers can log or
        surface the detail.  Only network-level failures raise
        :class:`TransportError`.
        """
        url = f"{self.base_url}/{path.lstrip('/')}"
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urlrequest.Request(url, data=data, headers=self._headers(), method=method)
        try:
            with urlrequest.urlopen(req, timeout=timeout or self.timeout) as resp:
                raw = resp.read().decode("utf-8")
                headers = {k: v for k, v in resp.headers.items()}
                return HttpResult(status=getattr(resp, "status", 200), headers=headers, text=raw)
        except urlerror.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            headers = {k: v for k, v in exc.headers.items()} if exc.headers else {}
            return HttpResult(status=exc.code, headers=headers, text=text)
        except Exception as exc:
            raise TransportError(str(exc)) from exc

    def stream_lines_sync(self, path: str, payload: dict[str, Any]):
        """POST ``payload`` and yield response lines one at a time.

        Used for both OpenAI SSE streams (``/v1/chat/completions``) and Ollama
        NDJSON streams (``/api/chat``).  The caller decides how to parse each
        line — this method only handles the wire-level framing.

        Yields
        ------
        str
            Each line with trailing ``\\r\\n`` stripped.  Empty lines are
            yielded as empty strings so the caller can detect SSE delimiters.

        Raises
        ------
        TransportError
            On HTTP errors or network failures.
        """
        url = f"{self.base_url}/{path.lstrip('/')}"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urlrequest.Request(url, data=data, headers=self._headers(), method="POST")
        try:
            with urlrequest.urlopen(req, timeout=self.timeout) as resp:
                for raw_line in resp:
                    yield raw_line.decode("utf-8").rstrip("\r\n")
        except urlerror.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            raise TransportError(f"HTTP {exc.code}: {text}") from exc
        except Exception as exc:
            raise TransportError(str(exc)) from exc

    async def _request(self, method: str, path: str, payload: dict[str, Any] | None) -> HttpResult:
        """Generic async request — thin wrapper around ``_do_request``."""
        return await asyncio.to_thread(self._do_request, method, path, payload)
