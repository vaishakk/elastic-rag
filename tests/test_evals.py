from __future__ import annotations

from rag.core.evals.precision_recall import (
    PrecisionRecall,
    PrecisionRecallQueryMetrics,
    QrelSample,
)


def test_evaluate_query_respects_rank_order_and_top_k():
    evaluator = PrecisionRecall(test_qrels=[], run_qrels=[], top_k=2)
    test_sample = QrelSample(id="q1", query="query", doc_ids=["d1", "d3"])
    run_sample = QrelSample(id="q1", query="query", doc_ids=["d2", "d1", "d3"])

    metrics = evaluator.evaluate_query(test_sample, run_sample, top_k=2)

    assert metrics.query_id == "q1"
    assert metrics.relevant == 2
    assert metrics.retrieved == 2
    assert metrics.hits == 1
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5


def test_aggregate_computes_macro_and_micro_values():
    evaluator = PrecisionRecall(test_qrels=[], run_qrels=[], top_k=10)
    metrics = [
        PrecisionRecallQueryMetrics(
            query_id="q1",
            relevant=2,
            retrieved=2,
            hits=1,
            precision=0.5,
            recall=0.5,
        ),
        PrecisionRecallQueryMetrics(
            query_id="q2",
            relevant=1,
            retrieved=1,
            hits=1,
            precision=1.0,
            recall=1.0,
        ),
    ]

    aggregate = evaluator.aggregate(metrics)

    assert aggregate.macro_precision == 0.75
    assert aggregate.macro_recall == 0.75
    assert aggregate.micro_precision == 2 / 3
    assert aggregate.micro_recall == 2 / 3


def test_aggregate_returns_zero_metrics_for_empty_input():
    evaluator = PrecisionRecall(test_qrels=[], run_qrels=[], top_k=10)

    aggregate = evaluator.aggregate([])

    assert aggregate.macro_precision == 0.0
    assert aggregate.macro_recall == 0.0
    assert aggregate.micro_precision == 0.0
    assert aggregate.micro_recall == 0.0


def test_evaluate_run_indexes_test_qrels_by_id():
    test_qrels = [
        QrelSample(id="q2", query="query 2", doc_ids=["d4", "d5"]),
        QrelSample(id="q1", query="query 1", doc_ids=["d1", "d2"]),
    ]
    run_qrels = [
        QrelSample(id="q1", query="query 1", doc_ids=["d2", "d9", "d1"]),
        QrelSample(id="q2", query="query 2", doc_ids=["d5", "d4", "d8"]),
    ]

    evaluator = PrecisionRecall(test_qrels=test_qrels, run_qrels=run_qrels, top_k=2)

    aggregate = evaluator.evaluate_run()

    assert aggregate.macro_precision == 0.75
    assert aggregate.macro_recall == 0.75
    assert aggregate.micro_precision == 0.75
    assert aggregate.micro_recall == 0.75
