from rag.core.evals.umbrella import *
from umbrella_eval import OpenAIJudge

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

def test_openai_judge():
    judge = OpenAIJudge()
    scores = judge.judge_query(
        query='2 + 2 = ?',
        context=[
            DocumentChunk(id='1', doc_id='doc1', text='2 + 2 = 4'),
            DocumentChunk(id='2', doc_id='doc1', text='2 + 3 = 5'),
            DocumentChunk(id='3', doc_id='doc1', text='Hello!')
        ]
    )
    assert len(scores) == 3
    assert scores[0] == 3
    assert scores[1] < 3
    assert scores[2] == 0