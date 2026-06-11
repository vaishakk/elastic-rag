#!/usr/bin/env python3
"""Generate a retrieval run from queries and evaluate it against qrels.

The checked-in evaluation files are:

- ``rag/evals/test.tsv``: qrels with ``query-id``, ``corpus-id``, and ``score``
- ``rag/evals/queries.jsonl``: query text keyed by ``_id``

This script reads the queries, retrieves documents from Elasticsearch, writes a
run file, and then computes precision/recall against the qrels.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rag import ElasticsearchVectorDB, LlamaIndexChunker, OpenAIEmbeddingModel
from rag.core.evals import (
    PrecisionRecall,
    PrecisionRecallMetrics,
    PrecisionRecallQueryMetrics,
    QrelSample,
)


DEFAULT_QRELS = Path("rag/evals/test.tsv")
DEFAULT_QUERIES = Path("rag/evals/queries-test.jsonl")
DEFAULT_RUN = Path("rag/evals/run.tsv")
DEFAULT_INDEX_NAME = "test-documents"


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"{path} is missing a header row")
        return list(reader)


def load_queries(path: Path) -> list[Query]:
    queries: list[Query] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL on line {line_number} in {path}") from exc
            query_id = record.get("_id")
            text = record.get("text")
            if not isinstance(query_id, str) or not query_id:
                raise ValueError(f"Missing or invalid _id on line {line_number} in {path}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Missing or invalid text on line {line_number} in {path}")
            queries.append(Query(query_id=query_id, text=text))
    if not queries:
        raise ValueError(f"{path} is empty")
    return queries


def load_qrels(path: Path, *, min_relevance: int = 1) -> list[QrelSample]:
    qrels_by_id: dict[str, list[str]] = {}
    for row in _read_tsv(path):
        try:
            score = int(row["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid qrels row in {path}: {row}") from exc
        if score >= min_relevance:
            query_id = row["query-id"]
            qrels_by_id.setdefault(query_id, []).append(row["corpus-id"])

    return [QrelSample(id=query_id, query=query_id, doc_ids=doc_ids) for query_id, doc_ids in qrels_by_id.items()]


def load_run(path: Path) -> list[QrelSample]:
    run_by_id: dict[str, list[tuple[str, float]]] = {}
    for row in _read_tsv(path):
        try:
            score = float(row["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid run row in {path}: {row}") from exc
        query_id = row["query-id"]
        run_by_id.setdefault(query_id, []).append((row["corpus-id"], score))

    run_samples: list[QrelSample] = []
    for query_id, entries in run_by_id.items():
        entries.sort(key=lambda item: (-item[1], item[0]))
        run_samples.append(
            QrelSample(
                id=query_id,
                query=query_id,
                doc_ids=[doc_id for doc_id, _ in entries],
            )
        )

    return run_samples


def write_run(path: Path, rows: Iterable[tuple[str, str, float, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["query-id", "corpus-id", "score", "rank"])
        for query_id, corpus_id, score, rank in rows:
            writer.writerow([query_id, corpus_id, score, rank])


def build_vector_db(*, index_name: str = DEFAULT_INDEX_NAME) -> ElasticsearchVectorDB:
    embed_model = OpenAIEmbeddingModel()
    chunker = LlamaIndexChunker()
    db = ElasticsearchVectorDB(model=embed_model, chunker=chunker, index_name=index_name)
    return db


def generate_run_rows(
    queries: Iterable[Query],
    db: ElasticsearchVectorDB,
    *,
    k: int,
) -> list[tuple[str, str, float, int]]:
    rows: list[tuple[str, str, float, int]] = []
    for query in queries:
        hits = db.search(query.text, top_k=k)
        for rank, hit in enumerate(hits, start=1):
            score = float(hit.metadata.get("score", 0.0) if hit.metadata else 0.0)
            rows.append((query.query_id, hit.doc_id, score, rank))
    return rows


def evaluate(qrels: list[QrelSample], run: list[QrelSample], *, k: int) -> list[PrecisionRecallQueryMetrics]:
    evaluator = PrecisionRecall(test_qrels=qrels, run_qrels=run, top_k=k)
    test_by_id = {sample.id: sample for sample in qrels}

    metrics: list[PrecisionRecallQueryMetrics] = []
    for run_sample in run:
        test_sample = test_by_id[run_sample.id]
        metrics.append(evaluator.evaluate_query(test_sample, run_sample, top_k=k))
    return metrics


def aggregate(metrics: list[PrecisionRecallQueryMetrics]) -> PrecisionRecallMetrics:
    evaluator = PrecisionRecall(test_qrels=[], run_qrels=[], top_k=0)
    return evaluator.aggregate(metrics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a run file and evaluate precision and recall against qrels.")
    parser.add_argument(
        "--qrels",
        type=Path,
        default=DEFAULT_QRELS,
        help="Path to the qrels TSV file (default: rag/evals/test.tsv).",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES,
        help="Path to the queries JSONL file (default: rag/evals/queries.jsonl).",
    )
    parser.add_argument(
        "--index-name",
        default=DEFAULT_INDEX_NAME,
        help='Elasticsearch index name to search (default: "test-documents").',
    )
    parser.add_argument(
        "--run",
        type=Path,
        default=DEFAULT_RUN,
        help="Path to write the generated run TSV (default: rag/evals/run.tsv).",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=10,
        help="Cutoff for evaluation. Use a positive integer to score top-k results.",
    )
    parser.add_argument(
        "--min-relevance",
        type=int,
        default=1,
        help="Minimum qrels score treated as relevant (default: 1).",
    )
    parser.add_argument(
        "--per-query",
        action="store_true",
        help="Print per-query precision and recall values.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    queries = load_queries(args.queries)
    db = build_vector_db(index_name=args.index_name)
    run_rows = generate_run_rows(queries, db, k=args.k)
    write_run(args.run, run_rows)

    qrels = load_qrels(args.qrels, min_relevance=args.min_relevance)
    run = load_run(args.run)
    evaluator = PrecisionRecall(test_qrels=qrels, run_qrels=run, top_k=args.k)
    summary = evaluator.evaluate_run()
    metrics = evaluate(qrels, run, k=args.k)

    print(f"Qrels: {args.qrels}")
    print(f"Queries: {args.queries}")
    print(f"Index: {args.index_name}")
    print(f"Run:   {args.run}")
    print(f"Cutoff: {args.k}")
    print(f"Relevant threshold: {args.min_relevance}")
    print()
    print(f"Macro precision@{args.k}: {summary.macro_precision:.4f}")
    print(f"Macro recall@{args.k}:    {summary.macro_recall:.4f}")
    print(f"Micro precision@{args.k}: {summary.micro_precision:.4f}")
    print(f"Micro recall@{args.k}:    {summary.micro_recall:.4f}")

    if args.per_query:
        print()
        print("query-id\trelevant\tretrieved\thits\tprecision\trecall")
        for item in metrics:
            print(
                f"{item.query_id}\t{item.relevant}\t{item.retrieved}\t{item.hits}\t"
                f"{item.precision:.4f}\t{item.recall:.4f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
