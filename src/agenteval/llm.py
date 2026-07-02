from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class LLMUnavailable(RuntimeError):
    pass


@dataclass
class LLMConfig:
    provider: str = "deepseek"
    model: str = "deepseek-v4-pro"
    base_url: str = "https://api.deepseek.com"
    timeout_s: float = 60.0
    max_tokens: int = 2048
    temperature: float = 0.1

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"),
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
            timeout_s=float(os.environ.get("DEEPSEEK_TIMEOUT_S", "60")),
            max_tokens=int(os.environ.get("DEEPSEEK_MAX_TOKENS", "2048")),
            temperature=float(os.environ.get("DEEPSEEK_TEMPERATURE", "0.1")),
        )


class DeepSeekJSONClient:
    def __init__(self, api_key: str | None = None, config: LLMConfig | None = None):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.config = config or LLMConfig.from_env()

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def complete_json(self, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise LLMUnavailable("DEEPSEEK_API_KEY is not set.")
        body = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        request = urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_s) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as exc:
            raise LLMUnavailable(f"DeepSeek API request failed: {exc}") from exc

        parsed = json.loads(raw)
        content = parsed["choices"][0]["message"].get("content") or ""
        if not content.strip():
            raise LLMUnavailable("DeepSeek API returned empty JSON content.")
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMUnavailable(f"DeepSeek API returned non-JSON content: {content[:200]}") from exc


def truncate_text(value: Any, limit: int = 1200) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, str) else value
    return text if len(text) <= limit else text[:limit] + "...<truncated>"
