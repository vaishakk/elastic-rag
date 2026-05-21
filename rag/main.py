from pathlib import Path
import pytest
from olrag.adapters.document_adapters import *
from olrag.rag.pdf_document_reader import *
from olrag.core.document import *

stack = DocumentStackFromPDFFolder('tests', PyPDFExtractor())
print(stack.documents[0].text)