from __future__ import annotations

from pathlib import Path

from precision_recall_eval import (
    Query,
    aggregate,
    evaluate,
    generate_run_rows,
    load_queries,
    load_qrels,
    load_run,
    write_run,
)


def test_precision_recall_eval(tmp_path: Path):
    qrels_path = tmp_path / "qrels.tsv"
    run_path = tmp_path / "run.tsv"

    qrels_path.write_text(
        "\n".join(
            [
                "query-id\tcorpus-id\tscore",
                "q1\td1\t2",
                "q1\td2\t1",
                "q2\td3\t1",
                "q2\td4\t1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    run_path.write_text(
        "\n".join(
            [
                "query-id\tcorpus-id\tscore",
                "q1\td1\t0.9",
                "q1\td9\t0.8",
                "q1\td2\t0.7",
                "q2\td4\t0.6",
                "q2\td8\t0.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    qrels = load_qrels(qrels_path)
    run = load_run(run_path)
    assert [sample.id for sample in qrels] == ["q1", "q2"]
    assert [sample.doc_ids for sample in qrels] == [["d1", "d2"], ["d3", "d4"]]
    assert [sample.id for sample in run] == ["q1", "q2"]
    assert [sample.doc_ids for sample in run] == [["d1", "d9", "d2"], ["d4", "d8"]]

    metrics = evaluate(qrels, run, k=2)
    summary = aggregate(metrics)

    assert summary.macro_precision == 0.5
    assert summary.macro_recall == 0.5
    assert summary.micro_precision == 0.5
    assert summary.micro_recall == 0.5


def test_run_generation_from_queries(tmp_path: Path):
    queries_path = tmp_path / "queries.jsonl"
    run_path = tmp_path / "run.tsv"

    queries_path.write_text(
        "\n".join(
            [
                '{"_id": "q1", "text": "first query"}',
                '{"_id": "q2", "text": "second query"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class _FakeDB:
        @staticmethod
        def search(query: str, top_k: int):
            if query == "first query":
                return [
                    type("Hit", (), {"doc_id": "d1", "metadata": {"score": 1.5}})(),
                    type("Hit", (), {"doc_id": "d2", "metadata": {"score": 1.0}})(),
                ]
            return [
                type("Hit", (), {"doc_id": "d3", "metadata": {"score": 2.0}})(),
            ]

    queries = load_queries(queries_path)
    assert queries == [
        Query(query_id="q1", text="first query"),
        Query(query_id="q2", text="second query"),
    ]

    rows = generate_run_rows(queries, _FakeDB(), k=2)
    write_run(run_path, rows)

    assert run_path.read_text(encoding="utf-8").splitlines() == [
        "query-id\tcorpus-id\tscore\trank",
        "q1\td1\t1.5\t1",
        "q1\td2\t1.0\t2",
        "q2\td3\t2.0\t1",
    ]
