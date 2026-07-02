from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .schemas import to_jsonable


def load_structured_file(path: str | Path) -> Any:
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
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(to_jsonable(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return resolved


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def write_jsonl(path: str | Path, records: Iterable[Any]) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(to_jsonable(record), ensure_ascii=False, sort_keys=True) + "\n")
    return resolved
