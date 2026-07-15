from pathlib import Path
from docling.document_converter import DocumentConverter

from rag import DocumentError, DocumentStackFromFolder
from rag.core.document import DocumentRepo, DocumentExtractor


class DoclingTextExtractor(DocumentExtractor):

    def __init__(self):
        super().__init__()

    def extract_text(self, url: Path, **kwargs):
        converter = DocumentConverter()
        try:
            result = converter.convert(url)
            with open(str(url) + '.md', 'w') as f:
                f.write(result.document.export_to_markdown())
        except Exception as e:
            raise DocumentError(e)
        return '', result.document.export_to_markdown()

class PlainTextExtractor(DocumentExtractor):

    def __init__(self):
        super().__init__()

    def extract_text(self, url: Path, **kwargs):
        try:
            with open(url, 'r') as f:
                text = f.read()
            title = text.split('\n')[0]
            if len(title) > 25:
                title = title[:25]
        except Exception as e:
            raise DocumentError(e)
        return title, text

class DocumentStackFromMarkdownFolder(DocumentStackFromFolder):

    def __init__(self, url: str, doc_repo: DocumentRepo):
        super().__init__(url, doc_repo, PlainTextExtractor())




