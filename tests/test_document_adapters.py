from __future__ import annotations

from pathlib import Path

from rag import DocumentStackFromFolder
from rag.core.document import DocumentRepo, Document, DocumentExtractor


class _FakeExtractor(DocumentExtractor):
    def extract_text(self, url: Path):
        return f"title:{url.name}", f"text:{url.name}"

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


def test_document_stack_from_folder_recurses_into_subfolders(tmp_path: Path):
    (tmp_path / "root.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "textfile.txt").write_text("not a pdf", encoding="utf-8")

    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "child.pdf").write_bytes(b"%PDF-1.4")

    deeper = nested / "deeper"
    deeper.mkdir()
    (deeper / "grandchild.pdf").write_bytes(b"%PDF-1.4")

    stack = DocumentStackFromFolder(str(tmp_path), InMemoryRepo(), _FakeExtractor())
    assert len(stack.documents) == 4
    titles = []
    paths = []
    for doc_id in stack.documents:
        doc = stack.get_doc_by_id(doc_id)
        titles.append(doc.title)
        paths.append(doc.path)
    assert sorted(titles) == [
        "title:child.pdf",
        "title:grandchild.pdf",
        "title:root.pdf",
        "title:textfile.txt"
    ]
    print(sorted(paths))
    assert sorted(paths) == [
        nested / "child.pdf",
        deeper / "grandchild.pdf",
        tmp_path / "root.pdf",
        tmp_path / "textfile.txt"
    ]