import os
from pathlib import Path
from typing import List

from rag.core.document import DocumentRepo, DocumentExtractor
from rag import Document, DocumentStack, DocumentError

class DocumentStackFromFolder(DocumentStack):
    """
    Builds a DocumentStack from all PDF files in a given folder and its subfolders.
    """

    def __init__(self, folder_url: str, doc_repo: DocumentRepo, extractor: DocumentExtractor):
        """
        Initialize the stack by scanning the folder and loading each PDF.

        Args:
            folder_url (str): Path or URL to the folder containing PDF files.
            extractor (PDFTextExtractor): Instance used to extract PDF content.
        """
        self.extractor = extractor
        self.doc_repo = doc_repo
        super().__init__(repo=doc_repo, doc_extractor=extractor)
        if not os.path.isdir(folder_url):
            raise DocumentError('Folder url {} does not exist'.format(folder_url))
        try:
            files = list_dir(folder_url)
        except DocumentError as e:
            raise DocumentError(e)
        for idx, file in enumerate(files):
            try:
                title, text = self.extractor.extract_text(str(file))
            except:
                continue
            doc = Document(
                    id=str(idx),
                    path=file,
                    title=title,
                    text=text,
                    summary=''
                )
            self.add(doc)


def list_dir(folder_url: str) -> List[Path]:
    """
    List PDF files within the specified folder and all nested subfolders.

    Args:
        folder_url (str): Path or URL to the folder.

    Returns:
        List[Path]: Paths to all PDF files found.
    """
    root = Path(folder_url)
    files = sorted(
        file for file in root.rglob("*")
    )
    if not files:
        raise DocumentError('No files found in {}'.format(folder_url))
    return files
