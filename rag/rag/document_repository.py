from pathlib import Path
from typing import ClassVar

from rag.core.document import Document, DocumentRepo
from rag.core.exceptions import DocumentError

class DictDocumentRepository(DocumentRepo):
    """Simple in-memory repository backed by a class-level dictionary."""

    _docs: ClassVar[dict[str, Document]] = {}

    def __init__(self):
        super().__init__()

    def add_doc(self, doc: Document):
        self._docs[doc.id] = doc

    def get_doc_by_id(self, id: str) -> Document:
        if id not in self._docs:
            raise DocumentError(f"Document with id {id} not found")

        return self._docs[id]
