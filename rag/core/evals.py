from dataclasses import dataclass
from typing import Set

@dataclass(frozen=True)
class PrecisionRecallQueryMetrics:
    query_id: str
    relevant: int
    retrieved: int
    hits: int
    precision: float
    recall: float

@dataclass(frozen=True)
class PrecisionRecallMetrics:
    macro_precision: float
    macro_recall: float
    micro_precision: float
    micro_recall: float

@dataclass
class QrelSample:
    id: str
    query: str
    doc_ids: list[str]

class PrecisionRecall:

    def __init__(self, test_qrels: list[QrelSample], run_qrels: list[QrelSample], top_k: int) -> None:
        self.test_qrels = test_qrels
        self.run_qrels = run_qrels
        self.top_k = top_k

    def evaluate_query(
            self,
            test_sample: QrelSample,
            run_sample: QrelSample,
            top_k: int
    ) -> PrecisionRecallQueryMetrics:

        ranked = run_sample.doc_ids
        relevant_docs = set(test_sample.doc_ids)
        if top_k > 0:
            ranked = ranked[:top_k] 

        hits = sum(1 for doc_id in ranked if doc_id in relevant_docs)
        retrieved = len(ranked)
        relevant = len(relevant_docs)

        precision = hits / retrieved if retrieved else 0.0
        recall = hits / relevant if relevant else 0.0

        return PrecisionRecallQueryMetrics(
            query_id=run_sample.id,
            relevant=relevant,
            retrieved=retrieved,
            hits=hits,
            precision=precision,
            recall=recall,
        )

    def evaluate_run(self) -> PrecisionRecallMetrics:
        test_by_id = {sample.id: sample for sample in self.test_qrels}

        metrics = []
        for run_sample in self.run_qrels:
            test_sample = test_by_id[run_sample.id]
            metrics.append(self.evaluate_query(test_sample, run_sample, top_k=self.top_k))

        return self.aggregate(metrics)

    def aggregate(self, metrics: list[PrecisionRecallQueryMetrics]) -> PrecisionRecallMetrics:
        if not metrics:
            return PrecisionRecallMetrics(0.0, 0.0, 0.0, 0.0)

        total_hits = sum(item.hits for item in metrics)
        total_retrieved = sum(item.retrieved for item in metrics)
        total_relevant = sum(item.relevant for item in metrics)

        macro_precision = sum(item.precision for item in metrics) / len(metrics)
        macro_recall = sum(item.recall for item in metrics) / len(metrics)
        micro_precision = total_hits / total_retrieved if total_retrieved else 0.0
        micro_recall = total_hits / total_relevant if total_relevant else 0.0

        return PrecisionRecallMetrics(macro_precision, macro_recall, micro_precision, micro_recall)






