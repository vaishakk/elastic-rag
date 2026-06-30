from rag.core.evals.umbrella import *

class MedianJudge(LLMJudge):

    def judge_query(self, query: str, context: List[DocumentChunk]) -> List[int]:
        return [3]*len(context)


def test_scores_to_metrics(rag):
    umbrella = Umbrella(
        rag=rag,
        llmjudge=MedianJudge(),
        queries=[Query(query_id='1', text='query1'), Query(query_id='2', text='query2')],
        relevant_threshold=3
    )
    context = umbrella.rag.retrieve(query='query1')
    scores = umbrella.llmjudge.judge_query('query1', context)
    avg_score, precision, mrr = umbrella._scores_to_metrics(scores)
    assert avg_score == 3
    assert precision == 1
    assert mrr == 1

def test_evaluate(rag):
    umbrella = Umbrella(
        rag=rag,
        llmjudge=MedianJudge(),
        queries=[Query(query_id='1', text='query1'), Query(query_id='2', text='query2')],
        relevant_threshold=3
    )
    umbrella.evaluate()
    assert '1' in umbrella.scores
    assert '2' in umbrella.scores
    assert umbrella.scores['1'].avg_score == 3
    assert umbrella.scores['2'].avg_score == 3