from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class Document:
    """
    Represents a source document with metadata and content.

    Attributes:
        id (str): Unique identifier for the document.
        path (Path): File path or source of the document.
        title (str): Title or name of the document.
        text (str): Full text content of the document.
        summary (str, optional): Generated summary of the document.
    """
    id: str
    path: Path
    title: str
    text: str = field(repr=False)
    summary: str = None

class DocumentExtractor(ABC):

    @abstractmethod
    def extract_text(self, url: str) -> Tuple[str, str]:
        pass

class DocumentRepo(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get_doc_by_id(self, id: str) -> Document:
        pass

    @abstractmethod
    def add_doc(self, document: Document) -> bool:
        pass

@dataclass
class DocumentStack:
    """
    Holds a collection of Document objects.

    Attributes:
        documents (List[str]): List of Document ids.
        repo: Document repository that holds all the documents in the stack.
    """
    repo: DocumentRepo
    documents: List[str]  # List of doc ids

    def __init__(self, repo: DocumentRepo, doc_extractor: DocumentExtractor):
        self.repo = repo
        self.doc_extractor = doc_extractor
        self.documents = []

    def add(self, document: Document):
        """
        Add a new Document to the stack and update the combined text.

        Args:
            document (Document): The document to add.
        """
        self.repo.add_doc(document)
        self.documents.append(document.id)

    def get_doc_by_id(self, id: str) -> Document:
        return self.repo.get_doc_by_id(id)

    def __len__(self):
        """
        Return the number of documents in the stack.
        """
        return len(self.documents)

@dataclass
class DocumentChunk:
    """
    Represents a fragment or chunk of a document.

    Attributes:
        id (str): Unique identifier for the chunk.
        doc_id (str): Identifier of the parent document.
        text (str): Text content of the chunk.
        metadata (dict): Dictionary of metadata for the chunk.
    """
    id: str
    doc_id: str
    text: str
    metadata: Optional[dict] = None

class Chunker(ABC):
    """
    Abstract base class for document chunkers.

    Implementations should split a DocumentStack into smaller DocumentChunks.
    """

    def __init__(self):
        """
        Initialize the chunker.
        """
        super().__init__()

    @abstractmethod
    def split_doc(self, documents: DocumentStack) -> List[DocumentChunk]:
        """
        Split the provided DocumentStack into a list of DocumentChunks.

        Args:
            documents (DocumentStack): The stack of documents to split.

        Returns:
            List[DocumentChunk]: A list of document chunks generated from the documents.
        """
        raise NotImplementedError
    
class DocumentRepository(ABC):
    """
    High-level accessor for documents persisted in a stack or backing store.

    Concrete implementations wrap the data source that produced the
    `DocumentStack` (e.g., in-memory list, filesystem, database) and expose a
    uniform interface for lookup operations. This indirection lets the RAG
    runtime fetch full document metadata at answer time without coupling to the
    ingestion mechanism.
    """

    def __init__(self, doc_stack: DocumentStack):
        pass

    @abstractmethod
    def get_doc_by_id(self, id: str) -> Document:
        """
        Retrieve a Document by its identifier.

        Args:
            id (str): Identifier of the document to fetch.

        Returns:
            Document: The matching document instance.

        Raises:
            DocumentError: If the document cannot be found or retrieved.
        """
        raise NotImplementedError
