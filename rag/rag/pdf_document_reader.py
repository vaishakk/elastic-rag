from rag.adapters.document_adapters import *
import os
from pypdf import PdfReader

class PyPDFExtractor(PDFTextExtractor):

    def __init__(self):
        super().__init__()

    def extract_text(self, url):
        reader = PdfReader(url)
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