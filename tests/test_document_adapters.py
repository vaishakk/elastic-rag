from __future__ import annotations

from pathlib import Path

from rag import DocumentStackFromFolder
from rag.core.document import DocumentRepo, Document, DocumentExtractor


class _FakeExtractor(DocumentExtractor):
    def extract_text(self, url: Path):
        return f"text:{url.name}", f"title:{url.name}"

class InMemoryRepo(DocumentRepo):

    store: dict[str,Document] = {}

    def add_doc(self, document: Document) -> bool:
        if self.store is None:
            return False
        self.store[document.id] = document
        return True

    def get_doc_by_id(self, id:str):
        if not self.store:
            return None
        return self.store[id]


def test_document_stack_from_pdf_folder_recurses_into_subfolders(tmp_path: Path):
    (tmp_path / "root.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "ignore.txt").write_text("not a pdf", encoding="utf-8")

    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "child.pdf").write_bytes(b"%PDF-1.4")

    deeper = nested / "deeper"
    deeper.mkdir()
    (deeper / "grandchild.pdf").write_bytes(b"%PDF-1.4")

    stack = DocumentStackFromFolder(str(tmp_path), InMemoryRepo(), _FakeExtractor())

    assert {doc.path for doc in stack.documents} == {
        tmp_path / "root.pdf",
        nested / "child.pdf",
        deeper / "grandchild.pdf",
    }
    assert len(stack.documents) == 3
    assert [doc.title for doc in stack.documents] == [
        "title:child.pdf",
        "title:grandchild.pdf",
        "title:root.pdf",
    ]
