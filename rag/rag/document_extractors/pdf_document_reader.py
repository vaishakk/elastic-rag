from pypdf.errors import PdfStreamError

from rag.adapters.document_adapters.document_adapters import *
from pypdf import PdfReader

from rag.core.document import DocumentExtractor


class PyPDFExtractor(DocumentExtractor):

    def __init__(self):
        super().__init__()

    def extract_text(self, url: Path, **kwargs):
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
        return title, text

