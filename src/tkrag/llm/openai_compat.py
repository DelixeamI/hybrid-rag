"""Real LLM over an OpenAI-compatible /chat/completions endpoint (Ollama, Gemini, OpenAI...)."""

import time

import httpx

_RETRY_STATUS = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


def _daily_quota_exhausted(response: httpx.Response) -> bool:
    """Waiting a minute does not help when the per-day quota is gone: fail at once."""
    return response.status_code == 429 and "PerDay" in response.text


class OpenAICompatibleLLM:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        *,
        timeout: float = 120.0,
        max_retries: int = 6,
        json_mode: bool = True,
        min_interval: float = 0.0,
    ) -> None:
        self.model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._timeout = timeout
        self._max_retries = max_retries
        self._json_mode = json_mode
        self._min_interval = min_interval
        self._last_call = 0.0

    def complete(self, system: str, user: str) -> str:
        body: dict[str, object] = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self._json_mode:
            body["response_format"] = {"type": "json_object"}
        return self._post_with_retries(body)

    def _post_with_retries(self, body: dict[str, object]) -> str:
        delay = 2.0
        for attempt in range(self._max_retries + 1):
            self._throttle()
            try:
                response = httpx.post(
                    self._url, json=body, headers=self._headers, timeout=self._timeout
                )
            except httpx.TransportError as error:  # dropped connection, timeout, DNS...
                if attempt == self._max_retries:
                    raise LLMError(f"LLM transport error: {error}") from error
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
                continue
            if response.status_code not in _RETRY_STATUS or _daily_quota_exhausted(response):
                break
            if attempt == self._max_retries:
                break
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
        if response.status_code != 200:
            raise LLMError(f"LLM HTTP {response.status_code}: {response.text[:300]}")
        content = response.json()["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise LLMError("LLM reply has no text content")
        return content

    def _throttle(self) -> None:
        wait = self._last_call + self._min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()
