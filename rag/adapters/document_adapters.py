from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import List

from sqlalchemy.testing.suite.test_reflection import metadata

from rag import DocumentError
from rag.core.document import Document, DocumentStack
import os

class TextExtractor(ABC):
    """
    Abstract base class for PDF text extractors.

    Subclasses must implement extract_text to read a PDF and return its content.
    """

    @abstractmethod
    def extract_text(self, url: Path, **kwargs) -> (str, str):
        """
        Extract the full text and title from a PDF file.

        Args:
        url (Path): Path or URL to the PDF file.

        Returns:
            tuple(str, str): A tuple containing:
                - text (str): The extracted text content.
                - title (str): The extracted title or metadata from the PDF.
        """
        raise NotImplementedError


class DocumentFromPDF(Document):
    """
    Loads a Document instance by extracting text and title from a PDF source.

    Attributes:
        extractor (PDFTextExtractor): The extractor used to read PDF content.
    """

    def __init__(
        self,
        id: str,
        file_url: Path,
        summary: str,
        extractor: TextExtractor
    ):
        """
        Initialize a DocumentFromPDF.

        Uses the extractor to read the PDF at file_url and populate the Document fields.

        Args:
            id (str): Unique identifier for this document.
            file_url (Path): Path or URL to the PDF file.
            summary (str): Optional summary of the document.
            extractor (PDFTextExtractor): Instance to extract text and title.
        """
        text, title = extractor.extract_text(file_url)
        super().__init__(id=id, path=file_url, title=title, text=text, summary=summary)
        self.extractor = extractor


class DocumentStackFromPDFFolder(DocumentStack):
    """
    Builds a DocumentStack from all PDF files in a given folder.
    """

    def __init__(self, folder_url: str, extractor: TextExtractor):
        """
        Initialize the stack by scanning the folder and loading each PDF.

        Args:
            folder_url (str): Path or URL to the folder containing PDF files.
            extractor (PDFTextExtractor): Instance used to extract PDF content.
        """
        self.extractor = extractor
        if not os.path.isdir(folder_url):
            raise DocumentError('Folder url {} does not exist'.format(folder_url))
        try:
            files = self.list_pdf_dir(folder_url)
        except DocumentError as e:
            raise DocumentError(e)
        docs: List[Document] = []
        for idx, file in enumerate(files):
            docs.append(
                DocumentFromPDF(
                    id=str(idx),
                    file_url=Path(folder_url) / file,
                    summary='',
                    extractor=self.extractor
                )
            )
        super().__init__(docs)

    def list_pdf_dir(self, folder_url: str) -> List[str]:
        """
        List PDF files within the specified folder.

        Args:
            folder_url (str): Path or URL to the folder.

        Returns:
            List[str]: Filenames of all PDF files found.
        """
        pdf_files = [file for file in os.listdir(folder_url) if file.endswith('.pdf')]
        if not pdf_files:
            raise DocumentError('No PDF files found in {}'.format(folder_url))
        return pdf_files


class JSONLExtractor(TextExtractor):
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

