import pytest
from huggingface_hub.cli import repos

from rag.core import DocumentError
from rag.core.document import *

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

class DummyExtract(DocumentExtractor):

    def extract_text(self, url: str):
        return 'extracted title', 'extracted text'

def dummy_file_reader(path: Path):
    if 'existing_file' in str(path):
        return 'file title', 'file text'
    raise DocumentError('File not found')

class DummyFileExtract(DocumentExtractor):

    def extract_text(self, url: str):
        try:
            title, text = dummy_file_reader(Path(url))
        except DocumentError:
            raise DocumentError('File not found')
        return title, text


def test_document_init():
    doc = Document(
        id = '1',
        title = 'title',
        text='text',
        summary='summary',
        path = Path('path')
    )
    assert doc.id == '1'
    assert doc.title == 'title'
    assert doc.text == 'text'
    assert doc.summary == 'summary'

def test_doc_stack_add_doc():

    repo = InMemoryRepo()
    extract = DummyExtract()
    stack = DocumentStack(repo, extract)
    stack.add(
        Document(
            id = '1',
            title = 'title',
            text = 'text',
            summary = 'summary',
            path=Path('path')
        )
    )
    assert stack.get_doc_by_id('1').id == '1'

def test_doc_stack_extract_doc():
    repo = InMemoryRepo()
    extract = DummyExtract()
    stack = DocumentStack(repo, extract)
    title, text = stack.doc_extractor.extract_text('path')
    assert title == 'extracted title'
    assert text == 'extracted text'

def test_extract_doc_existing_file():
    extractor = DummyFileExtract()
    title, text = extractor.extract_text('existing_file.pdf')
    assert title == 'file title'
    assert text == 'file text'

def test_extract_doc_non_existing_file():
    extractor = DummyFileExtract()
    with pytest.raises(DocumentError) as e:
        title, text = extractor.extract_text('abscent_file.pdf')
    assert "File not found" in str(e.value)

def test_doc_stack_get_doc_by_id():
    repo = InMemoryRepo()
    extract = DummyExtract()
    stack = DocumentStack(repo, extract)
    stack.add(
        Document(
            id = '1',
            title = 'title',
            text = 'text',
            summary = 'summary',
            path = Path('path')
        )
    )
    doc = stack.get_doc_by_id('1')
    assert doc.id == '1'
    assert doc.title == 'title'
    assert doc.text == 'text'