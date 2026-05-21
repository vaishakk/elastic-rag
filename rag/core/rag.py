from rag.core.document import *
from rag.core.embedding import *
from rag.core.vectordb import *
from rag.core.inference import *
from abc import ABC, abstractmethod

class RAG(ABC):

    def __init__(self, doc_stack: DocumentStack, embed_model: EmbeddingModel, db: VectorDB, infer_model: InferenceModel):
        self.doc_stack = doc_stack
        self.embed_model = embed_model
        if getattr(db, "model", None) is not None and db.model is not embed_model:
            raise ValueError("VectorDB must be initialised with the same EmbeddingModel instance passed to RAG")
        self.db = db
        self.infer_model = infer_model

    def create_db(self):
        """
        Create and initialize the vector database with a batch of documents.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        self.db.create_db(self.doc_stack)

    def add_stack(self, stack: DocumentStack):
        """
        Add all documents in a DocumentStack to the vector database.

        Args:
            stack (DocumentStack): A collection of Document instances.
        """
        self.db.add_stack(stack=stack)

    def add_doc(self, document: Document):
        """
        Add a single Document to the vector database.

        Args:
            document (Document): The document to add.
        """
        self.db.add_doc(document=document)

    def add_chunk(self, chunk: DocumentChunk):
        """
        Add a single DocumentChunk to the vector database.

        Implementations should embed the chunk and persist the embedding
        along with any necessary metadata.

        Args:
            chunk (DocumentChunk): The document chunk to add.

        Returns:
            NotImplemented: If the method is not implemented by the vectordb.
        """
        if type(self.db.add_chunk) is VectorDB.add_chunk:
            return NotImplemented
        return self.db.add_chunk(chunk)

    @abstractmethod
    def retrieve(self, query: str) -> list[DocumentChunk]:
        raise NotImplementedError
    
    def infer(self, query: str, context: list[DocumentChunk]) -> str:
        return self.infer_model.infer(query=query, context=context)

    

