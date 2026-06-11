from __future__ import annotations

from pathlib import Path

from make_queries_test import filter_queries, load_test_query_ids, write_jsonl


def test_make_queries_test_filters_queries_by_qrels_ids(tmp_path: Path):
    queries_path = tmp_path / "queries.jsonl"
    qrels_path = tmp_path / "test.tsv"
    output_path = tmp_path / "queries-test.jsonl"

    queries_path.write_text(
        "\n".join(
            [
                '{"_id": "q1", "text": "first", "metadata": {"url": "https://example.com/1"}}',
                '{"_id": "q2", "text": "second", "metadata": {"url": "https://example.com/2"}}',
                '{"_id": "q3", "text": "third", "metadata": {"url": "https://example.com/3"}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    qrels_path.write_text(
        "\n".join(
            [
                "query-id\tcorpus-id\tscore",
                "q3\td1\t1",
                "q1\td2\t2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    query_ids = load_test_query_ids(qrels_path)
    filtered = filter_queries(queries_path, query_ids)
    write_jsonl(output_path, filtered)

    assert query_ids == {"q1", "q3"}
    assert [record["_id"] for record in filtered] == ["q1", "q3"]
    assert output_path.read_text(encoding="utf-8").splitlines() == [
        '{"_id": "q1", "text": "first", "metadata": {"url": "https://example.com/1"}}',
        '{"_id": "q3", "text": "third", "metadata": {"url": "https://example.com/3"}}',
    ]
