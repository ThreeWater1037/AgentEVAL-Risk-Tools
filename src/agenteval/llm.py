"""DeepSeek JSON 调用封装。

Tool1/Tool2 的 LLM 能力都通过这个轻量客户端进入系统。没有 API key 或返回
非 JSON 时会抛 LLMUnavailable，上层必须保留确定性路径或显式 skipped 状态。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class LLMUnavailable(RuntimeError):
    """LLM 不可用、请求失败或返回格式不满足 JSON 协议。"""

    pass


@dataclass
class LLMConfig:
    """从环境变量读取的 DeepSeek 调用配置。"""

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
    """只面向 JSON object 响应的 DeepSeek 客户端。"""

    def __init__(self, api_key: str | None = None, config: LLMConfig | None = None):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.config = config or LLMConfig.from_env()

    @property
    def available(self) -> bool:
        """只用 API key 判断可用性，避免初始化阶段发网络请求。"""
        return bool(self.api_key)

    def complete_json(self, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        """发送系统提示和结构化 payload，并强制解析为 JSON object。"""
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
    """控制传给 LLM 的上下文长度，避免 prompt 被大型对象撑爆。"""
    text = json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, str) else value
    return text if len(text) <= limit else text[:limit] + "...<truncated>"
