from rag.core.document import *

class InMemoryRepo(DocumentRepo):

    store: dict[str,Document] = {}

    def add_doc(self, document: Document) -> bool:
        print(self.store)
        if self.store is None:
            return False
        self.store[document.id] = document
        return True

    def get_doc_by_id(self, id:str):
        if not self.store:
            return None
        return self.store[id]

def test_document_repo():
    repo = InMemoryRepo()
    documents = [
        Document(id="doc-1", path=Path("doc-1.txt"), title="Doc 1", text="x"),
        Document(id="doc-2", path=Path("doc-2.txt"), title="Doc 2", text="y"),
        Document(id="doc-3", path=Path("doc-3.txt"), title="Doc 3", text="z"),
    ]
    for document in documents:
        repo.add_doc(document)

    assert repo.get_doc_by_id("doc-1") == documents[0]
    assert repo.get_doc_by_id("doc-2") == documents[1]
    assert repo.get_doc_by_id("doc-3") == documents[2]
    assert repo.get_doc_by_id("doc-3").id == "doc-3"

def test_add_doc():
    repo = InMemoryRepo()
    documents = [
        Document(id="doc-1", path=Path("doc-1.txt"), title="Doc 1", text="x"),
        Document(id="doc-2", path=Path("doc-2.txt"), title="Doc 2", text="y"),
        Document(id="doc-3", path=Path("doc-3.txt"), title="Doc 3", text="z"),
    ]
    stack = DocumentStack(repo, [])
    assert len(stack) == 0
    stack.add(documents[0])
    assert len(stack) == 1
    assert stack.get_doc_by_id("doc-1") == documents[0]