import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union, Sequence

from rag import Document, DocumentError, DocumentStack
from rag.core.document import DocumentExtractor


class MarkDownExtractor(DocumentExtractor):


    def extract_text(self, url: Path, **kwargs) -> Union[Sequence[str], DocumentError]:
        """
        Extract the full text and title from a supported file.

        Args:
        url (Path): Path or URL to the file.

        Returns:
            tuple(str, str): A tuple containing:
                - text (str): The extracted markdown text content.
                - title (str): The extracted title or metadata from the PDF.
        """
        raise NotImplementedError

class MDDocumentStackFromFolder(DocumentStack):

    def __init__(self, folder_url: str, extractor: MarkDownExtractor):
        """
        Initialize the stack by scanning the folder and loading each PDF.

        Args:
            folder_url (str): Path or URL to the folder containing PDF files.
            extractor (PDFTextExtractor): Instance used to extract PDF content.
        """
        self.extractor = extractor
        if not os.path.isdir(folder_url):
            raise DocumentError('Folder url {} does not exist'.format(folder_url))
        docs = self.extract_files(folder_url)
        super().__init__(docs)

    def extract_files(self, folder_url: str) -> List[Document]:
        """
        Extract all supported files within the specified folder and all nested subfolders.

        Args:
            folder_url (str): Path or URL to the folder.

        Returns:
            DocumentStack: DocumentStack containing all supported files.
        """
        stack: list[Document] = []
        root = Path(folder_url)
        files = sorted(
            file for file in root.rglob("*")
            if file.is_file()
        )

        for idx, file in enumerate(files):
            print(f'[MDDocumentStackFromFolder] Extracting file: {file}')
            try:
                title, text = self.extractor.extract_text(file)
            except DocumentError as e:
                print(f'{file} not indexed because of error: {e}')
                continue
            else:
                stack.append(Document(
                    id=str(idx),
                    path=file,
                    title=title,
                    text=text)
                )

        return stack

