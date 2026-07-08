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

class DummyExtract(DocumentExtractor):

    def extract_text(self, document: Document):
        return 'extracted title', 'extracted text'

def test_document_init():
    doc = Document(
        id = '1',
        title = 'title',
        text='text',
        summary='summary'
    )
    assert doc.id == '1'
    assert doc.title == 'title'
    assert doc.text == 'text'
    assert doc.summary == 'summary'

def test_document_repo():
    repo = InMemoryRepo(doc_extractor=DummyExtract())
