from abc import ABC, abstractmethod
from rag.core.document import DocumentChunk, DocumentStack, Document, Chunker
from rag.core.embedding import EmbeddingModel, Embedding

class VectorDB(ABC):
    """
    Abstract base class for a vector database that stores embeddings for document chunks.

    Attributes:
        chunker (Chunker): Responsible for splitting documents into chunks.
        model (EmbeddingModel): Responsible for embedding chunks into vectors.
    """

    def __init__(self, model: EmbeddingModel):
        """
        Initialize the VectorDB with a chunker and embedding model.

        Args:
            model (EmbeddingModel): Instance used to embed document chunks.
        """
        super().__init__()
        self.model = model

    @abstractmethod
    def create_db(self, stack: DocumentStack):
        """
        Create and initialize the vector database with a batch of documents.

        Args:
            stack (DocumentStack): A collection of Document instances to index in bulk.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def add_stack(self, stack: DocumentStack):
        """
        Add all documents in a DocumentStack to the vector database.

        Args:
            stack (DocumentStack): A collection of Document instances.
        """
        raise NotImplementedError

    def add_doc(self, document: Document):
        """
        Add a single Document to the vector database.

        Args:
            document (Document): The document to add.
        """
        stack = DocumentStack([document])
        self.add_stack(stack=stack)

    @abstractmethod
    def add_chunk(self, chunk: DocumentChunk):
        """
        Add a single DocumentChunk to the vector database.

        Implementations should embed the chunk and persist the embedding
        along with any necessary metadata.

        Args:
            chunk (DocumentChunk): The document chunk to add.

        Returns:
            NotImplemented: If the method is not implemented by a subclass.
        """
        return NotImplemented
