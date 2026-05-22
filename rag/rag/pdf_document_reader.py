from pathlib import Path

from pypdf.errors import PdfReadError, PdfStreamError

from rag import DocumentError
from rag.adapters.document_adapters import *
import os
from pypdf import PdfReader

class PyPDFExtractor(PDFTextExtractor):

    def __init__(self):
        super().__init__()

    def extract_text(self, url: Path):
        if url.suffix.lower() != '.pdf':
            raise DocumentError('Only PDF is supported.')
        try:
            reader = PdfReader(str(url))
        except FileNotFoundError:
            raise DocumentError('File not found.')
        except PdfStreamError:
            raise DocumentError('File is not a PDF.')
        text = ''
        title = ''
        for page in reader.pages:
            text += page.extract_text()
        if '/Title' in reader.metadata:
            title = reader.metadata['/Title']
        return text, title

# docs = DocumentStackFromPDFFolder(docs_folder, PyPDFExtractor())
# print([doc.title for doc in docs.documents])
# print(len(docs))
