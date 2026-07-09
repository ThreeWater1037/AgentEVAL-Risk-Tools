"""JSON/YAML/JSONL 结构化文件工具。

主要供目标清单、导入结果和后续批量实验数据复用。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .schemas import to_jsonable


def load_structured_file(path: str | Path) -> Any:
    """按扩展名读取 YAML 或 JSON；未安装 PyYAML 时 YAML 会回落到 JSON 解析。"""
    resolved = Path(path)
    text = resolved.read_text(encoding="utf-8")
    if resolved.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore

            return yaml.safe_load(text)
        except ModuleNotFoundError:
            pass
    return json.loads(text)


def write_structured_file(path: str | Path, data: Any) -> Path:
    """以 JSON 形式写出结构化数据。"""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(to_jsonable(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return resolved


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSON Lines，每个非空行是一条记录。"""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def write_jsonl(path: str | Path, records: Iterable[Any]) -> Path:
    """写出 JSON Lines，适合追加式实验记录后续兼容。"""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(to_jsonable(record), ensure_ascii=False, sort_keys=True) + "\n")
    return resolved
