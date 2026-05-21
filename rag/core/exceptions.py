"""
Core exception classes for the olrag library.
Defines a structured hierarchy for categorizing and handling errors
across document loading, chunking, embedding, vector storage, retrieval,
and inference stages.
"""

class OlragError(Exception):
    """
    Base class for all exceptions raised by the olrag library.
    """
    pass

class DocumentError(OlragError):
    """
    Raised when there is an error loading or parsing a document.
    """
    pass

class ChunkingError(OlragError):
    """
    Raised when splitting documents into chunks fails.
    """
    pass

class EmbeddingError(OlragError):
    """
    Raised when generating embeddings for document chunks fails.
    """
    pass

class VectorDBError(OlragError):
    """
    Raised when storing or indexing embeddings in the vector database fails.
    """
    pass

class RetrievalError(OlragError):
    """
    Raised when retrieving relevant chunks from the vector database fails
    or yields no results.
    """
    pass

class InferenceError(OlragError):
    """
    Raised when generating responses from the inference (LLM) model fails.
    """
    pass

class AdapterError(OlragError):
    """
    Raised when an adapter (e.g., PDFTextExtractor) encounters an I/O or parsing error.
    """
    pass
