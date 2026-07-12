import json
from pathlib import Path
from typing import List

from rag import DocumentError, TextExtractor, Document, DocumentStack
from rag.core.document import DocumentExtractor


class JSONLExtractor(DocumentExtractor):
    """
    Default extractor for JSONL records.

    Expects each record to contain a ``text`` field and optionally a
    ``title`` field.
    """

    def extract_text(self, record: dict, source: Path, id_field=None) -> (str, str):
        if not isinstance(record, dict):
            raise DocumentError("Each JSONL line must be a JSON object.")
        metadata = None
        if id_field:
            metadata = {
                id_field: record[id_field]
            }
        text = record.get("text", "")
        title = record.get("title", source.stem)

        if not isinstance(text, str):
            raise DocumentError("JSONL field 'text' must be a string.")
        if not isinstance(title, str):
            raise DocumentError("JSONL field 'title' must be a string.")

        return text, title, metadata


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
        text, title, metadata = extractor.extract_text(record, file_url, id_field="_id")
        super().__init__(id=metadata.get('_id', id), path=file_url, title=title, text=text, summary=summary)
        self.extractor = extractor


class DocumentStackFromJSONLFile(DocumentStack):
    """
    Builds a DocumentStack from a single JSONL file.
    """

    def __init__(self, file_url: str | Path):
        self.extractor = JSONLExtractor()
        self.file_url = Path(file_url)
        if not self.file_url.is_file():
            raise DocumentError("File url {} does not exist".format(self.file_url))
        if self.file_url.suffix.lower() != ".jsonl":
            raise DocumentError("Only JSONL files are supported.")

        docs: List[Document] = []
        try:
            with self.file_url.open("r", encoding="utf-8") as f:
                for line_number, raw_line in enumerate(f, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise DocumentError(f"Invalid JSONL on line {line_number}.") from exc

                    docs.append(
                        DocumentFromJSONL(
                            id=str(len(docs)),
                            file_url=self.file_url,
                            summary="",
                            extractor=self.extractor,
                            record=record,
                        )
                    )
        except FileNotFoundError as exc:
            raise DocumentError("File not found.") from exc

        if not docs:
            raise DocumentError("JSONL file is empty.")

        super().__init__(docs)
