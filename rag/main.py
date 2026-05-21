from pathlib import Path
import pytest
from rag.adapters.document_adapters import *
from rag.rag.embedding import OpenAIEmbeddingModel
from rag.rag.pdf_document_reader import *
from rag.core.document import *

stack = DocumentStackFromPDFFolder('tests', PyPDFExtractor())
print(stack.documents[0].text)