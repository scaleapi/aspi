"""Shared I/O helpers.

Consolidates utilities that were previously duplicated across modules:
    - parse_json: was in run_pipeline.parse_json AND aspi_judge._parse_json
    - jsonl read/write
    - error logging to append-only JSONL files
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Iterator


def parse_json(text: str) -> dict:
    """Extract and parse the first JSON object from LLM output.

    Handles two common cases:
      1. The model wraps its response in a ``` markdown fence.
      2. The model prepends/appends prose to the JSON.

    If the first parse attempt fails, falls back to a regex search for
    the first ``{ ... }`` block.
    """
    text = text.strip()

    # Strip markdown code fence if present
    if text.startswith("```"):
        lines = text.split("\n")
        # drop the opening ``` line and (optionally) the closing ``` line
        if lines and lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines)

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: find the first {...} substring by brace matching
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    # Last-resort regex for a JSON-looking blob
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from text (first 400 chars):\n{text[:400]}")


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of dicts. Empty lines are skipped."""
    path = Path(path)
    out: list[dict] = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """Stream-read a JSONL file one record at a time."""
    path = Path(path)
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, records: Iterable[dict], mode: str = "w") -> None:
    """Write an iterable of dicts to a JSONL file (one JSON object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode) as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def append_jsonl(path: str | Path, record: dict) -> None:
    """Append a single record to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def log_error(pair_key: str, stage: str, error_msg: str, errors_path: str | Path) -> None:
    """Append a structured error record to an error-log JSONL."""
    append_jsonl(errors_path, {
        "pair_key": pair_key,
        "stage": stage,
        "error": error_msg,
    })


def completed_keys_from_jsonl(
    path: str | Path,
    key_fields: tuple[str, ...] = ("suite", "user_task_id", "injection_task_id"),
) -> set[tuple]:
    """Read an output JSONL and return the set of record keys already present.

    Used for resume support: instead of maintaining a separate `.progress_*.txt`
    file, we read the output file itself to determine which records are done.
    """
    path = Path(path)
    if not path.exists():
        return set()
    keys: set[tuple] = set()
    for r in iter_jsonl(path):
        keys.add(tuple(r.get(f, "") for f in key_fields))
    return keys
