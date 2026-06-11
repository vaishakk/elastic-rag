#!/usr/bin/env python3
"""Index corpus.jsonl into Elasticsearch using ElasticsearchVectorDB."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from elasticsearch import Elasticsearch

from rag import DocumentStackFromJSONLFile, ElasticsearchVectorDB, LlamaIndexChunker, OpenAIEmbeddingModel


DEFAULT_CORPUS = Path("rag/evals/corpus.jsonl")
DEFAULT_INDEX_NAME = "nfcorp-documents"
DEFAULT_ES_URL = "https://localhost:9200"


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def build_client() -> Elasticsearch:
    es_url = os.environ.get("ES_URL", DEFAULT_ES_URL)
    password = os.environ.get("ELASTIC_PASSWORD")
    if not password:
        raise RuntimeError("ELASTIC_PASSWORD must be set")

    ca_cert = os.environ.get("ES_CA_CERT")
    if ca_cert:
        ca_path = Path(ca_cert)
        if not ca_path.is_absolute():
            ca_path = _project_root() / ca_path
    else:
        local_ca = _project_root() / "http_ca.crt"
        if local_ca.exists():
            ca_path = local_ca
        else:
            es_home = os.environ.get("ES_HOME")
            if not es_home:
                raise RuntimeError("Set ES_CA_CERT or ES_HOME, or place http_ca.crt in the project root")
            ca_path = Path(es_home) / "config/certs/http_ca.crt"

    if not ca_path.exists():
        raise RuntimeError(f"CA cert not found: {ca_path}")

    return Elasticsearch(
        es_url,
        basic_auth=("elastic", password),
        ca_certs=str(ca_path),
        request_timeout=30,
    )


def index_corpus(client: Elasticsearch, corpus_path: Path, index_name: str) -> int:
    stack = DocumentStackFromJSONLFile(corpus_path)
    db = ElasticsearchVectorDB(
        model=OpenAIEmbeddingModel(),
        chunker=LlamaIndexChunker(),
        client=client,
        index_name=index_name,
    )
    db.create_db(stack)
    return len(stack.documents)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index corpus.jsonl into Elasticsearch.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help="Path to the corpus JSONL file (default: rag/evals/corpus.jsonl).",
    )
    parser.add_argument(
        "--index-name",
        default=DEFAULT_INDEX_NAME,
        help='Elasticsearch index name (default: "nfcorp-documents").',
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    client = build_client()
    indexed = index_corpus(client, args.corpus, args.index_name)

    print(f"Corpus: {args.corpus}")
    print(f"Index: {args.index_name}")
    print(f"Indexed documents: {indexed}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
