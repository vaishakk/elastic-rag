from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List
from olrag.core.document import DocumentChunk

@dataclass
class Embedding:
    """
    Represents the embedding vector for a document chunk.

    Attributes:
        chunk_id (str): Identifier of the source DocumentChunk.
        embedding (List[float]): Embedding vector values.
    """
    chunk_id: str
    embedding: List[float]

class EmbeddingModel(ABC):
    """
    Abstract base class for embedding models.

    Subclasses must implement the embed method to convert a DocumentChunk into an Embedding.
    """

    def __init__(self):
        """
        Initialize the embedding model.
        """
        pass

    @abstractmethod
    def embed(self, chunk: DocumentChunk) -> Embedding:
        """
        Generate an embedding for the given DocumentChunk.

        Args:
            chunk (DocumentChunk): The document chunk to embed.

        Returns:
            Embedding: The resulting embedding object for the chunk.
        """
        raise NotImplementedError
