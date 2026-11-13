"""JSON report output without exposing dataset contents."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from dataset_preflight import __version__
from dataset_preflight.checks import InputFile, check_dataset, validate_inputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check local JSONL text datasets.")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--train", type=Path, action="append", required=True)
    parser.add_argument("--eval", type=Path, action="append", default=[])
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--id-field", default=None)
    parser.add_argument("--max-findings", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    inputs = [InputFile(path, "train") for path in args.train]
    inputs.extend(InputFile(path, "eval") for path in args.eval)
    try:
        validate_inputs(inputs)
        if args.output:
            if args.output.is_dir():
                raise ValueError("Output must be a file")
            output = args.output.resolve()
            for item in inputs:
                source = item.path.resolve()
                if (
                    output == source
                    or output in source.parents
                    or source in output.parents
                ):
                    raise ValueError("Output must not overlap an input path")
        report = check_dataset(
            inputs,
            text_field=args.text_field,
            id_field=args.id_field,
            max_findings=args.max_findings,
        )
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=".dataset-preflight-", dir=args.output.parent
            ) as tmp:
                staged = Path(tmp) / "report.json"
                staged.write_text(encoded, encoding="utf-8")
                staged.replace(args.output)
        sys.stdout.write(encoded)
        return 0 if report["ok"] else 1
    except (ValueError, OSError, RuntimeError):
        print(
            "Cannot check dataset: inspect paths, field names, and limits.",
            file=sys.stderr,
        )
        return 2
    except KeyboardInterrupt:
        print("Dataset check interrupted.", file=sys.stderr)
        return 130
