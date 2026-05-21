from olrag.core.document import *
from olrag.core.embedding import *
from olrag.core.vectordb import *
from abc import ABC, abstractmethod

class InferenceModel(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def infer(self, query: str, context: list[DocumentChunk]) -> str:
        raise NotImplementedError