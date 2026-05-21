#!/usr/bin/env python3

import os
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


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


def create_index(es: Elasticsearch) -> None:
    if es.indices.exists(index=INDEX_NAME):
        print(f"Index already exists: {INDEX_NAME}")
        return

    es.indices.create(
        index=INDEX_NAME,
        mappings={
            "properties": {
                "title": {"type": "text"},
                "content": {"type": "text"},
                "status": {"type": "keyword"},
                "created_at": {"type": "date"},
            }
        },
    )

    print(f"Created index: {INDEX_NAME}")


def index_documents(es: Elasticsearch) -> None:
    documents = [
        {
            "_index": INDEX_NAME,
            "_id": "1",
            "_source": {
                "title": "Hello Elasticsearch",
                "content": "This is my first document indexed from Python.",
                "status": "published",
                "created_at": "2026-05-19",
            },
        },
        {
            "_index": INDEX_NAME,
            "_id": "2",
            "_source": {
                "title": "Python Search Example",
                "content": "This document explains how to search Elasticsearch using Python.",
                "status": "published",
                "created_at": "2026-05-19",
            },
        },
        {
            "_index": INDEX_NAME,
            "_id": "3",
            "_source": {
                "title": "Draft Document",
                "content": "This is a draft and should not appear in published-only filters.",
                "status": "draft",
                "created_at": "2026-05-19",
            },
        },
    ]

    success_count, errors = bulk(es, documents, refresh=True)

    print(f"Indexed documents: {success_count}")

    if errors:
        print("Errors:")
        print(errors)


def main() -> None:
    es = get_client()

    try:
        info = es.info()
        print(f"Connected to cluster: {info['cluster_name']}")

        create_index(es)
        index_documents(es)
    finally:
        es.close()


if __name__ == "__main__":
    main()
