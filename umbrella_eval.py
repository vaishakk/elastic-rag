import csv
import json
import os
from pathlib import Path

from tqdm import tqdm

from rag import ElasticsearchVectorDB, OpenAIEmbeddingModel, LlamaIndexChunker, DocumentStackFromJSONLFile, \
    DictDocumentRepository
from rag.core.evals.umbrella import *
from rag.rag.LLMs.open_ai import *

from pydantic import BaseModel

DEFAULT_QRELS = Path("rag/evals/test.tsv")
DEFAULT_QUERIES = Path("rag/evals/queries-test.jsonl")
DEFAULT_RUN = Path("rag/evals/run.tsv")
DEFAULT_INDEX_NAME = "test-documents"
SYSTEM_PROMPT_PATH = Path("rag/umbrella_prompt.txt")

class UmbrellaScore(BaseModel):
    score: int

def read_system_prompt():
    if not os.path.exists(SYSTEM_PROMPT_PATH):
        raise FileNotFoundError('Umbrella Prompt file not found')
    with open(SYSTEM_PROMPT_PATH) as f:
        return f.read()

@dataclass(frozen=True)
class Query:
    query_id: str
    text: str

class OpenAIJudge(LLMJudge):

    def __init__(self):
        super().__init__()
        self.system_prompt = read_system_prompt()
        self.client = OpenAI()

    def judge_query(self, query: str, context: List[DocumentChunk]) -> List[int]:
        scores: List[int] = []
        for chunk in context:
            message = self.system_prompt.format(query=query, passage=chunk)
            str_score = chat_completion(
                client=self.client,
                query=message,
                system_prompt='',
                response_format=UmbrellaScore
            )
            try:
                score = int(str_score['score'])
            except ValueError:
                print(f'Cannot extract score from {str_score}.')
            else:
                scores.append(score)
        return scores

def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"{path} is missing a header row")
        return list(reader)

def build_rag(*, index_name: str = DEFAULT_INDEX_NAME) -> RAG:
    embed_model = OpenAIEmbeddingModel()
    chunker = LlamaIndexChunker()
    db = ElasticsearchVectorDB(model=embed_model, chunker=chunker, index_name=index_name)
    repo = DictDocumentRepository()
    stack = DocumentStackFromJSONLFile(file_url='rag/evals/corpus.jsonl', repo=repo)
    rag = RAG(doc_stack=stack, embed_model=embed_model, db=db)
    return rag

def load_queries(path: Path) -> list[Query]:
    queries: list[Query] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL on line {line_number} in {path}") from exc
            query_id = record.get("_id")
            text = record.get("text")
            if not isinstance(query_id, str) or not query_id:
                raise ValueError(f"Missing or invalid _id on line {line_number} in {path}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Missing or invalid text on line {line_number} in {path}")
            queries.append(Query(query_id=query_id, text=text))
    if not queries:
        raise ValueError(f"{path} is empty")
    return queries

def run_eval(queries: list[Query], rag: RAG, judge: LLMJudge) -> dict[str,UmbrellaMetrics]:
    umbrella = Umbrella(rag, judge, 3, queries)
    rows = []
    if os.path.isfile('umbrella_metrics.csv'):
        os.remove('umbrella_metrics.csv')
    print('Calculating Umbrella Metrics...')
    for _ in tqdm(umbrella.evaluate()):
        metric = next(reversed(umbrella.scores.values()))
        rows.append((metric.query_id,metric.avg_score,metric.precision,metric.mrr))
    with open('umbrella_metrics.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(('Query ID', 'Avg Score', 'Precision', 'MRR'))
        writer.writerows(rows)
    return umbrella.scores

def main():
    queries = load_queries(DEFAULT_QUERIES)
    rag = build_rag(index_name=DEFAULT_INDEX_NAME)
    judge = OpenAIJudge()
    metrics = run_eval(queries=queries, rag=rag, judge=judge)
    avg_precision, avg_mrr = 0.0, 0.0

    for metric in tqdm(metrics.values()):
        print('Calculating averages...')
        avg_precision += metric.precision / len(queries)
        avg_mrr += metric.mrr / len(queries)
    print(f'Average precision: {avg_precision:.4f}, MRR: {avg_mrr:.4f}')

if __name__ == "__main__":
    main()
