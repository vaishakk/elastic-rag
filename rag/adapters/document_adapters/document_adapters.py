from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import List, Union, Sequence

from rag import DocumentError
from rag.core.document import Document, DocumentStack
import os

class TextExtractor(ABC):
    """
    Abstract base class for PDF text extractors.

    Subclasses must implement extract_text to read a PDF and return its content.
    """

    @abstractmethod
    def extract_text(self, url: Path, **kwargs) -> Union[Sequence[str], DocumentError]:
        """
        Extract the full text and title from a PDF file.

        Args:
        url (Path): Path or URL to the PDF file.

        Returns:
            tuple(str, str): A tuple containing:
                - text (str): The extracted text content.
                - title (str): The extracted title or metadata from the PDF.
        """
        raise NotImplementedError

