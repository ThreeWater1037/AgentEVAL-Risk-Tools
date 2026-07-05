from __future__ import annotations

from functools import lru_cache
from importlib.resources import files


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    prompt_path = files(__package__).joinpath(f"{name}.txt")
    return prompt_path.read_text(encoding="utf-8").strip()
