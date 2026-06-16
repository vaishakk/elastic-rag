import os
from pathlib import Path
from typing import List

from rag import Document, TextExtractor, DocumentStack, DocumentError


class DocumentFromPDF(Document):
    """
    Loads a Document instance by extracting text and title from a PDF source.

    Attributes:
        extractor (PDFTextExtractor): The extractor used to read PDF content.
    """

    def __init__(
        self,
        id: str,
        file_url: Path,
        summary: str,
        extractor: TextExtractor
    ):
        """
        Initialize a DocumentFromPDF.

        Uses the extractor to read the PDF at file_url and populate the Document fields.

        Args:
            id (str): Unique identifier for this document.
            file_url (Path): Path or URL to the PDF file.
            summary (str): Optional summary of the document.
            extractor (PDFTextExtractor): Instance to extract text and title.
        """
        text, title = extractor.extract_text(file_url)
        super().__init__(id=id, path=file_url, title=title, text=text, summary=summary)
        self.extractor = extractor


class DocumentStackFromPDFFolder(DocumentStack):
    """
    Builds a DocumentStack from all PDF files in a given folder and its subfolders.
    """

    def __init__(self, folder_url: str, extractor: TextExtractor):
        """
        Initialize the stack by scanning the folder and loading each PDF.

        Args:
            folder_url (str): Path or URL to the folder containing PDF files.
            extractor (PDFTextExtractor): Instance used to extract PDF content.
        """
        self.extractor = extractor
        if not os.path.isdir(folder_url):
            raise DocumentError('Folder url {} does not exist'.format(folder_url))
        try:
            files = self.list_pdf_dir(folder_url)
        except DocumentError as e:
            raise DocumentError(e)
        docs: List[Document] = []
        for idx, file in enumerate(files):
            docs.append(
                DocumentFromPDF(
                    id=str(idx),
                    file_url=file,
                    summary='',
                    extractor=self.extractor
                )
            )
        super().__init__(docs)

    def list_pdf_dir(self, folder_url: str) -> List[Path]:
        """
        List PDF files within the specified folder and all nested subfolders.

        Args:
            folder_url (str): Path or URL to the folder.

        Returns:
            List[Path]: Paths to all PDF files found.
        """
        root = Path(folder_url)
        pdf_files = sorted(
            file for file in root.rglob("*")
            if file.is_file() and file.suffix.lower() == ".pdf"
        )
        if not pdf_files:
            raise DocumentError('No PDF files found in {}'.format(folder_url))
        return pdf_files
