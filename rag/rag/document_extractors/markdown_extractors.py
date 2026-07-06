import os
from pathlib import Path
from typing import List

from docling.document_converter import DocumentConverter

from rag import DocumentError, DocumentStack, Document, TextExtractor, MarkDownExtractor

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

class PlainTextExtractor(TextExtractor):

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

class DocumentStackFromMarkdownFolder(DocumentStack):

    def __init__(self, url: str, extractor: PlainTextExtractor):
        if not os.path.isdir(url):
            raise DocumentError('Folder url {} does not exist'.format(url))
        self.extractor = extractor
        root = Path(url)
        self.md_files = sorted(
            file for file in root.rglob("*")
            if file.is_file() and file.suffix.lower() == ".md"
        )
        if not self.md_files:
            raise DocumentError('No markdown files found in {}'.format(url))
        docs: List[Document] = []
        for idx, md_file in enumerate(self.md_files):
            title, text = self.extract_text(md_file)
            docs.append(
                Document(
                    id=str(idx),
                    title=title,
                    text=text,
                    path=md_file
                )
            )
        super().__init__(docs, url)

    def extract_text(self, url: Path, **kwargs):

        return self.extractor.extract_text(url, **kwargs)





