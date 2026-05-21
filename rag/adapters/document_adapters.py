from abc import ABC, abstractmethod
from typing import List
from rag.core.document import Document, DocumentStack
import os

class PDFTextExtractor(ABC):
    """
    Abstract base class for PDF text extractors.

    Subclasses must implement extract_text to read a PDF and return its content.
    """

    @abstractmethod
    def extract_text(self, url: str) -> (str, str):
        """
        Extract the full text and title from a PDF file.

        Args:
            url (str): Path or URL to the PDF file.

        Returns:
            tuple(str, str): A tuple containing:
                - text (str): The extracted text content.
                - title (str): The extracted title or metadata from the PDF.
        """
        raise NotImplementedError


class DocumentFromPDF(Document):
    """
    Loads a Document instance by extracting text and title from a PDF source.

    Attributes:
        extractor (PDFTextExtractor): The extractor used to read PDF content.
    """

    def __init__(
        self,
        id: str,
        file_url: str,
        summary: str,
        extractor: PDFTextExtractor
    ):
        """
        Initialize a DocumentFromPDF.

        Uses the extractor to read the PDF at file_url and populate the Document fields.

        Args:
            id (str): Unique identifier for this document.
            file_url (str): Path or URL to the PDF file.
            summary (str): Optional summary of the document.
            extractor (PDFTextExtractor): Instance to extract text and title.
        """
        text, title = extractor.extract_text(file_url)
        super().__init__(id=id, path=file_url, title=title, text=text, summary=summary)
        self.extractor = extractor


class DocumentStackFromPDFFolder(DocumentStack, ABC):
    """
    Builds a DocumentStack from all PDF files in a given folder.

    Subclasses must implement list_pdf_dir to list PDF filenames in the folder.
    """

    def __init__(self, folder_url: str, extractor: PDFTextExtractor):
        """
        Initialize the stack by scanning the folder and loading each PDF.

        Args:
            folder_url (str): Path or URL to the folder containing PDF files.
            extractor (PDFTextExtractor): Instance used to extract PDF content.
        """
        self.extractor = extractor
        files = self.list_pdf_dir(folder_url)
        docs: List[Document] = []
        for file in files:
            docs.append(
                DocumentFromPDF(
                    id=str(len(docs)),
                    file_url=f"{folder_url}/{file}",
                    summary='',
                    extractor=self.extractor
                )
            )
        super().__init__(docs)

    def list_pdf_dir(self, folder_url: str) -> List[str]:
        """
        List PDF files within the specified folder.

        Args:
            folder_url (str): Path or URL to the folder.

        Returns:
            List[str]: Filenames of all PDF files found.
        """
        return [file for file in os.listdir(folder_url) if '.pdf' in file]
