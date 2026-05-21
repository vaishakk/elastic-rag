#!/usr/bin/env python3

import os
import sys
from elasticsearch import Elasticsearch


INDEX_NAME = "my-index"


def get_client() -> Elasticsearch:
    es_url = os.environ.get("ES_URL", "https://192.168.1.53:9200")
    password = os.environ["ELASTIC_PASSWORD"]
    ca_cert = os.environ.get("ES_CA_CERT", os.path.expandvars("$ES_HOME/config/certs/http_ca.crt"))

    return Elasticsearch(
        es_url,
        basic_auth=("elastic", password),
        ca_certs=ca_cert,
        request_timeout=30,
    )


def search(es: Elasticsearch, query_text: str) -> None:
    response = es.search(
        index=INDEX_NAME,
        query={
            "multi_match": {
                "query": query_text,
                "fields": ["title^2", "content"],
            }
        },
        size=10,
    )

    hits = response["hits"]["hits"]

    print(f"Found {len(hits)} result(s)\n")

    for hit in hits:
        source = hit["_source"]
        score = hit["_score"]

        print(f"ID: {hit['_id']}")
        print(f"Score: {score}")
        print(f"Title: {source.get('title')}")
        print(f"Status: {source.get('status')}")
        print(f"Content: {source.get('content')}")
        print("-" * 60)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python search_docs.py '<search text>'")
        sys.exit(1)

    query_text = " ".join(sys.argv[1:])

    es = get_client()
    try:
        search(es, query_text)
    finally:
        es.close()


if __name__ == "__main__":
    main()
