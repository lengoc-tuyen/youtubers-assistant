"""Bounded-retry, OpenAI-compatible JSON client for literal and Groq providers."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_USER_AGENT = "youtubers-assistant/1.0"


@dataclass(frozen=True)
class RetryPolicy:
    """Network retry limits. Retries are intentionally finite."""

    attempts: int = 3
    base_delay_seconds: float = 1.0


class ProviderRequestError(RuntimeError):
    """A sanitized provider failure that never includes request credentials."""


class OpenAICompatibleJsonClient:
    """POST chat-completions messages and extract strict JSON from assistant content."""

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 30.0,
        retry_policy: RetryPolicy = RetryPolicy(),
        user_agent: str = DEFAULT_USER_AGENT,
        opener: Callable[..., Any] = urlopen,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not endpoint.startswith("https://"):
            raise ValueError("Provider endpoint must use HTTPS.")
        if not api_key.strip() or not model.strip():
            raise ValueError("Provider API key and model are required.")
        if timeout_seconds <= 0 or retry_policy.attempts < 1:
            raise ValueError("Timeout and retry attempts must be positive.")
        if not user_agent.strip():
            raise ValueError("Provider user agent is required.")
        self._endpoint = endpoint
        self._api_key = api_key
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._retry_policy = retry_policy
        self._user_agent = user_agent
        self._opener = opener
        self._sleep = sleep

    def request_json(self, system_prompt: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Call an OpenAI-compatible endpoint and parse one JSON assistant message."""
        request_body = json.dumps({
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ],
            "response_format": {"type": "json_object"},
        }).encode("utf-8")

        last_error: Optional[Exception] = None
        for attempt in range(self._retry_policy.attempts):
            request = Request(
                self._endpoint,
                data=request_body,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": self._user_agent,
                },
                method="POST",
            )
            try:
                with self._opener(request, timeout=self._timeout_seconds) as response:
                    response_body = response.read().decode("utf-8")
                return self._extract_json_response(response_body)
            except HTTPError as error:
                last_error = error
                if error.code not in {429, 500, 502, 503, 504}:
                    break
            except (URLError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError) as error:
                last_error = error

            if attempt + 1 < self._retry_policy.attempts:
                self._sleep(self._retry_policy.base_delay_seconds * (2 ** attempt))

        if isinstance(last_error, HTTPError):
            raise ProviderRequestError(f"Provider request failed with HTTP {last_error.code}.") from last_error
        raise ProviderRequestError("Provider request failed after bounded retries.") from last_error

    @staticmethod
    def _extract_json_response(response_body: str) -> Mapping[str, Any]:
        try:
            outer = json.loads(response_body)
            content = outer["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("Assistant content is not text.")
            parsed = json.loads(content)
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ProviderRequestError("Provider returned an invalid JSON completion response.") from error
        if not isinstance(parsed, Mapping):
            raise ProviderRequestError("Provider JSON completion must be an object.")
        return parsed
