"""JSON 文件读写工具。

所有输出统一走 to_jsonable，以便 dataclass 和嵌套对象稳定落盘。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import to_jsonable


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在并返回 Path。"""
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def load_json(path: str | Path) -> Any:
    """按 UTF-8 读取 JSON。"""
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str | Path, value: Any) -> Path:
    """写入带缩进的 UTF-8 JSON，并保留末尾换行。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(value), fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return target
