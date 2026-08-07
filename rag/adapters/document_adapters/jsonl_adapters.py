import json
from pathlib import Path
from typing import List, Tuple, Any, Generator

from rag import DocumentError, Document, DocumentStack
from rag.core.document import DocumentExtractor, DocumentRepo


class JSONLExtractor(DocumentExtractor):
    """
    Default extractor for JSONL records.

    Expects each record to contain a ``text`` field and optionally a
    ``title`` field.
    """

    def extract_text(self, url: str) -> Generator[tuple[str, str], Any, None]:
        if not Path(url).is_file():
            raise DocumentError("File url {} does not exist".format(url))
        if Path(url).suffix.lower() != ".jsonl":
            raise DocumentError("Only JSONL files are supported.")

        try:
            with Path(url).open("r", encoding="utf-8") as f:
                for line_number, raw_line in enumerate(f, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise DocumentError(f"Invalid JSONL on line {line_number}.") from exc
                    yield self.extract_one(record)


        except FileNotFoundError as exc:
            raise DocumentError("File not found.") from exc


    def extract_one(self, record: dict, id_field=None) -> Tuple[str, str, dict]:
        if not isinstance(record, dict):
            raise DocumentError("Each JSONL line must be a JSON object.")
        metadata = None
        if id_field:
            metadata = {
                id_field: record[id_field]
            }
        text = record.get("text", "")
        title = record.get("title", "")

        if not isinstance(text, str):
            raise DocumentError("JSONL field 'text' must be a string.")
        if not isinstance(title, str):
            raise DocumentError("JSONL field 'title' must be a string.")

        return title, text, metadata


class DocumentFromJSONL(Document):
    """
    Loads a Document instance from one JSONL record.
    """

    def __init__(
        self,
        id: str,
        file_url: Path,
        summary: str,
        extractor: JSONLExtractor,
        record: dict,
    ):
        title, text, metadata = extractor.extract_text(record, file_url, id_field="_id")
        super().__init__(id=metadata.get('_id', id), path=file_url, title=title, text=text, summary=summary)
        self.extractor = extractor


class DocumentStackFromJSONLFile(DocumentStack):
    """
    Builds a DocumentStack from a single JSONL file.
    """

    def __init__(self, file_url: str, repo: DocumentRepo):
        self.file_url = file_url
        super().__init__(repo=repo, doc_extractor=JSONLExtractor())
        for title, text, metadata in self.doc_extractor.extract_text(self.file_url):
            self.add(
                Document(
                    id=str(len(self.documents)),
                    path=Path(self.file_url),
                    summary="",
                    title=title,
                    text=text,
                )
            )
