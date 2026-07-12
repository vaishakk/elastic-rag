from pathlib import Path

from rag import DictDocumentRepository
from rag.rag.document_extractors.markdown_extractors import PlainTextExtractor, DocumentStackFromMarkdownFolder


def test_plain_text_extract():
    extractor = PlainTextExtractor()
    title, text = extractor.extract_text(url=Path('README.md'))
    assert title is not None
    assert text is not None

def test_markdown_folder_stack():
    stack = DocumentStackFromMarkdownFolder(url='./docs/Markdowns', doc_repo=DictDocumentRepository())
    assert len(stack.documents) != 0
    assert stack.get_doc_by_id('1') is not None

