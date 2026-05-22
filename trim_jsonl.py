#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trim a JSONL file to an inclusive document range."
    )
    parser.add_argument("input", type=Path, help="Path to the input JSONL file")
    parser.add_argument("output", type=Path, help="Path to write the trimmed JSONL file")
    parser.add_argument(
        "--start-doc",
        type=int,
        required=True,
        help="1-based document number to start from, inclusive",
    )
    parser.add_argument(
        "--end-doc",
        type=int,
        required=True,
        help="1-based document number to end at, inclusive",
    )
    return parser.parse_args()


def trim_jsonl(input_path: Path, output_path: Path, start_doc: int, end_doc: int) -> int:
    if start_doc < 1:
        raise ValueError("--start-doc must be >= 1")
    if end_doc < start_doc:
        raise ValueError("--end-doc must be >= --start-doc")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    written = 0
    doc_index = 0

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for raw_line in src:
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue

            doc_index += 1
            if doc_index < start_doc:
                continue
            if doc_index > end_doc:
                break

            dst.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
            written += 1

    return written


def main() -> int:
    args = parse_args()

    try:
        written = trim_jsonl(args.input, args.output, args.start_doc, args.end_doc)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Wrote {written} document(s) from {args.input} "
        f"to {args.output} (docs {args.start_doc}..{args.end_doc})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
