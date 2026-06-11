#!/usr/bin/env python3
"""Filter queries.jsonl down to the query ids present in test.tsv."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_QUERIES = Path("rag/evals/queries.jsonl")
DEFAULT_QRELS = Path("rag/evals/test.tsv")
DEFAULT_OUTPUT = Path("rag/evals/queries-test.jsonl")


def load_test_query_ids(path: Path) -> set[str]:
    query_ids: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"{path} is missing a header row")
        for line_number, row in enumerate(reader, start=2):
            query_id = row.get("query-id")
            if not isinstance(query_id, str) or not query_id.strip():
                raise ValueError(f"Invalid query-id on line {line_number} in {path}")
            query_ids.add(query_id)
    return query_ids


def filter_queries(queries_path: Path, query_ids: set[str]) -> list[dict[str, object]]:
    filtered: list[dict[str, object]] = []
    with queries_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL on line {line_number} in {queries_path}") from exc

            query_id = record.get("_id")
            if not isinstance(query_id, str) or not query_id.strip():
                raise ValueError(f"Missing or invalid _id on line {line_number} in {queries_path}")

            if query_id in query_ids:
                filtered.append(record)

    return filtered


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create queries-test.jsonl from queries.jsonl using query ids from test.tsv."
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES,
        help="Path to the source queries JSONL file (default: rag/evals/queries.jsonl).",
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        default=DEFAULT_QRELS,
        help="Path to the qrels TSV file containing the query ids (default: rag/evals/test.tsv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the filtered JSONL file (default: rag/evals/queries-test.jsonl).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    query_ids = load_test_query_ids(args.qrels)
    filtered_queries = filter_queries(args.queries, query_ids)
    write_jsonl(args.output, filtered_queries)

    print(f"Source queries: {args.queries}")
    print(f"Qrels: {args.qrels}")
    print(f"Output: {args.output}")
    print(f"Matched queries: {len(filtered_queries)}")
    print(f"Matched ids: {len(query_ids)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
