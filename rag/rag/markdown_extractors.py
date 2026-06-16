from pathlib import Path

from docling.document_converter import DocumentConverter

from rag import DocumentError
from rag.adapters.document_adapters import MarkDownExtractor


class DoclingTextExtractor(MarkDownExtractor):

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
