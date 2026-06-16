from pathlib import Path
from typing import ClassVar

from rag.core.document import Document, DocumentRepository, DocumentStack
from rag.core.exceptions import DocumentError
from rag.rag.document_extractors.pdf_document_reader import PyPDFExtractor


class DictDocumentRepository(DocumentRepository):
    """Simple in-memory repository backed by a class-level dictionary."""

    _docs: ClassVar[dict[str, Path]] = {}

    def __init__(self, stack: DocumentStack):
        super().__init__(stack)
        self.extractor = PyPDFExtractor()
        self._register_stack(stack)

    def _register_stack(self, stack: DocumentStack):
        type(self)._docs = {doc.id: doc.path for doc in stack.documents}

    def get_doc_by_id(self, id: str) -> Document:
        if id not in self._docs:
            raise DocumentError(f"Document with id {id} not found")

        url = self._docs[id]

        if url.suffix.lower() != '.pdf':
            raise DocumentError('Only PDF documents are supported')

        text, title = self.extractor.extract_text(url)

        return Document(
            id=id,
            path=url,
            title=title,
            text=text,
            summary=''
        )
