from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest


@dataclass(slots=True)
class HttpResult:
    status: int
    headers: dict[str, str]
    text: str

    def json(self) -> Any:
        if self.status >= 400 or not self.text.strip():
            return None
        try:
            return json.loads(self.text)
        except json.JSONDecodeError:
            return None


class TransportError(RuntimeError):
    pass


class OpenAICompatibleTransport:
    def __init__(self, base_url: str, *, timeout: float = 300.0, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def get_json_sync(self, path: str, *, timeout: float | None = None) -> Any:
        return self._do_request("GET", path, None, timeout=timeout).json()

    def post_json_sync(self, path: str, payload: dict[str, Any], *, timeout: float | None = None) -> HttpResult:
        return self._do_request("POST", path, payload, timeout=timeout)

    async def get_json(self, path: str) -> Any:
        result = await asyncio.to_thread(self._do_request, "GET", path, None)
        return result.json()

    async def post_json(self, path: str, payload: dict[str, Any]) -> HttpResult:
        return await asyncio.to_thread(self._do_request, "POST", path, payload)

    def _do_request(self, method: str, path: str, payload: dict[str, Any] | None, *, timeout: float | None = None) -> HttpResult:
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
        """POST and yield SSE lines one at a time without buffering the full body."""
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
        return await asyncio.to_thread(self._do_request, method, path, payload)

