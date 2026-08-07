import statistics
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Tuple

from rag import DocumentChunk, RAG

@dataclass
class Query:
    query_id: str
    text: str

@dataclass
class UmbrellaMetrics:

    query_id: str
    avg_score: float
    precision: float
    mrr: float

class LLMJudge(ABC):
    @abstractmethod
    def judge_query(self, query: str, context: List[DocumentChunk]) -> List[int]:
        """Score retrieved chunks for a query.

        Args:
            query: The user query being evaluated.
            context: Retrieved document chunks for the query.

        Returns:
            A list of relevance scores, one score for each chunk.
        """
        pass

class Umbrella:

    def __init__(self, rag: RAG, llmjudge: LLMJudge, relevant_threshold: float, queries: List[Query]):
        """Create an umbrella evaluator.

        Args:
            rag: Retrieval system used to fetch context for each query.
            llmjudge: Judge that scores retrieved chunks for relevance.
            relevant_threshold: Minimum score treated as relevant.
        """
        self.rag = rag
        self.llmjudge = llmjudge
        self.relevant_threshold = relevant_threshold
        self.queries = queries
        self.scores = {}

    def evaluate_query(self, query: Query) -> UmbrellaMetrics:
        """Evaluate a query end to end and return aggregate metrics.

        The query is retrieved against the configured RAG backend, scored by
        the judge, and converted into precision and MRR metrics.
        """
        context = self.rag.retrieve(query=query.text)
        scores = self.llmjudge.judge_query(query.text, context)
        avg_score, precision, mrr = self._scores_to_metrics(scores)
        return UmbrellaMetrics(query.query_id, avg_score, precision, mrr)

    def _scores_to_metrics(self, scores: List[int]) -> Tuple[float, float, float]:
        """Convert raw relevance scores into umbrella metrics."""
        binary_relevance = [
            1 if score >= self.relevant_threshold else 0
            for score in scores
        ]
        return (statistics.mean(scores),
                sum(binary_relevance) / len(scores) if len(scores) > 0 else 0,
                self._calculate_mrr(binary_relevance)
                )

    def evaluate(self):
        for query in self.queries:
            self.scores[query.query_id] = self.evaluate_query(query=query)

    # Adapted from the Open RAG Eval project.
    # Original code: https://github.com/vectara/open-rag-eval/
    # License: Apache-2.0
    def _calculate_mrr(self, binary_relevance: list[int]) -> float:
        """Calculate Mean Reciprocal Rank from binary relevance scores.

        Args:
            binary_relevance: List of 1s and 0s indicating relevant and non-relevant items

        Returns:
            float: MRR score
        """
        for i, is_relevant in enumerate(binary_relevance, start=1):
            if is_relevant == 1:
                return 1.0 / i
        return 0.0
