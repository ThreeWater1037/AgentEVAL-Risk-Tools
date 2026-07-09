"""Prompt 资源加载器。

Prompt 文本集中放在 src/agenteval/prompts/*.txt，业务代码只按名字读取。
"""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """按 prompt 名读取同名 txt，并缓存避免重复 IO。"""
    prompt_path = files(__package__).joinpath(f"{name}.txt")
    return prompt_path.read_text(encoding="utf-8").strip()
