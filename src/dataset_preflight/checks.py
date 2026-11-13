"""Line-oriented validation and normalized-text overlap detection."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataset_preflight import __version__


@dataclass(frozen=True)
class InputFile:
    path: Path
    split: str


def _reject_constant(value: str) -> None:
    raise ValueError("Non-standard JSON constant")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Repeated object key")
        result[key] = value
    return result


def normalized_text(text: str) -> str:
    """Normalize Unicode NFC and whitespace, preserving case and punctuation."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def validate_inputs(inputs: list[InputFile]) -> None:
    if not inputs:
        raise ValueError("At least one input file is required")
    paths = set()
    for item in inputs:
        if item.split not in {"train", "eval"}:
            raise ValueError("Split must be train or eval")
        path = item.path.resolve()
        if path in paths:
            raise ValueError("An input path was supplied more than once")
        paths.add(path)
        if not path.is_file():
            raise ValueError("Every input must be a regular file")


def check_dataset(
    inputs: list[InputFile],
    *,
    text_field: str = "text",
    id_field: str | None = None,
    max_findings: int = 100,
) -> dict[str, Any]:
    validate_inputs(inputs)
    if not text_field or (id_field is not None and not id_field):
        raise ValueError("Field names must not be empty")
    if type(max_findings) is not int or max_findings < 1:
        raise ValueError("max_findings must be a positive integer")

    files = []
    findings = []
    counts: Counter[str] = Counter()
    seen_text: dict[str, dict[str, dict[str, Any]]] = {}
    seen_ids: dict[str, dict[str, dict[str, Any]]] = {}

    def add(code: str, location: dict[str, Any], first: dict[str, Any] | None = None):
        counts[code] += 1
        if len(findings) < max_findings:
            finding = {"code": code, **location}
            if first:
                finding["first_seen"] = first
            findings.append(finding)

    def index_record(
        index: dict[str, dict[str, dict[str, Any]]],
        digest: str,
        kind: str,
        location: dict[str, Any],
    ) -> None:
        per_split = index.setdefault(digest, {})
        split = location["split"]
        if split in per_split:
            add(f"duplicate_{kind}", location, per_split[split])
        other = "eval" if split == "train" else "train"
        if other in per_split:
            add(f"cross_split_{kind}", location, per_split[other])
        per_split.setdefault(split, location)

    for item in inputs:
        fingerprint = hashlib.sha256()
        total_bytes = 0
        lines = 0
        valid = 0
        with item.path.open("rb") as stream:
            for number, raw in enumerate(stream, start=1):
                fingerprint.update(raw)
                total_bytes += len(raw)
                lines += 1
                location = {"split": item.split, "file": str(item.path), "line": number}
                try:
                    line = raw.decode("utf-8")
                except UnicodeDecodeError:
                    add("invalid_utf8", location)
                    continue
                if not line.strip():
                    add("blank_line", location)
                    continue
                try:
                    record = json.loads(
                        line,
                        parse_constant=_reject_constant,
                        object_pairs_hook=_unique_object,
                    )
                except (ValueError, RecursionError):
                    add("invalid_json", location)
                    continue
                if not isinstance(record, dict):
                    add("not_object", location)
                    continue
                if text_field not in record:
                    add("missing_text", location)
                    continue
                text = record[text_field]
                if not isinstance(text, str):
                    add("text_not_string", location)
                    continue
                normalized = normalized_text(text)
                if not normalized:
                    add("empty_text", location)
                    continue
                try:
                    text_bytes = normalized.encode("utf-8")
                except UnicodeEncodeError:
                    add("invalid_text_unicode", location)
                    continue
                identity = record.get(id_field) if id_field is not None else None
                if id_field is not None and (
                    type(identity) not in {str, int}
                    or (isinstance(identity, str) and not identity.strip())
                ):
                    add("invalid_id", location)
                    continue
                id_bytes = None
                if id_field is not None:
                    encoded = json.dumps(
                        [type(identity).__name__, identity], ensure_ascii=False
                    )
                    try:
                        id_bytes = encoded.encode("utf-8")
                    except UnicodeEncodeError:
                        add("invalid_id", location)
                        continue
                valid += 1
                index_record(
                    seen_text, hashlib.sha256(text_bytes).hexdigest(), "text", location
                )
                if id_bytes is not None:
                    index_record(
                        seen_ids, hashlib.sha256(id_bytes).hexdigest(), "id", location
                    )
        if not lines:
            add(
                "empty_file",
                {"split": item.split, "file": str(item.path), "line": None},
            )
        files.append(
            {
                "split": item.split,
                "file": str(item.path),
                "sha256": fingerprint.hexdigest(),
                "bytes": total_bytes,
                "lines": lines,
                "valid_records": valid,
                "invalid_records": lines - valid,
            }
        )
    return {
        "schema_version": 1,
        "tool_version": __version__,
        "config": {
            "text_field": text_field,
            "id_field": id_field,
            "normalization": "Unicode NFC and collapsed whitespace; case-sensitive",
            "max_findings": max_findings,
        },
        "summary": {
            "files": len(files),
            "lines": sum(file["lines"] for file in files),
            "valid_records": sum(file["valid_records"] for file in files),
            "invalid_records": sum(file["invalid_records"] for file in files),
            "finding_count": sum(counts.values()),
            "counts": dict(sorted(counts.items())),
            "findings_truncated": sum(counts.values()) > len(findings),
        },
        "files": files,
        "findings": findings,
        "ok": not counts,
    }
