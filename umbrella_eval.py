import csv
import json
from pathlib import Path

from rag import ElasticsearchVectorDB, OpenAIEmbeddingModel, LlamaIndexChunker, DocumentStackFromJSONLFile
from rag.core.evals.umbrella import *

DEFAULT_QRELS = Path("rag/evals/test.tsv")
DEFAULT_QUERIES = Path("rag/evals/queries-test.jsonl")
DEFAULT_RUN = Path("rag/evals/run.tsv")
DEFAULT_INDEX_NAME = "test-documents"

@dataclass(frozen=True)
class Query:
    query_id: str
    text: str

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
    stack = DocumentStackFromJSONLFile(file_url='rag/evals/corpus.jsonl')
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

def run_eval(queries: list[Query], rag: RAG, judge: LLMJudge) -> List[UmbrellaMetrics]:
    metrics = []
    umbrella = Umbrella(rag, judge, 3)
    for query in queries:
        metrics.append(umbrella.evaluate_query(query.text))
    return metrics