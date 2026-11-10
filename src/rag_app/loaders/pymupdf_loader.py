from .base import BasePDFLoader
from langchain_community.document_loaders import PyMuPDFLoader


class PyMuPDFDocumentLoader(BasePDFLoader):
    """PDF loader using the PyMuPDF library (faster than pypdf).

    Args:
        file_path: Path to one PDF file.
    Raises:
        ValueError: If the file does not exist.
    """

    name = "pymupdf"

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.loader = PyMuPDFLoader(self.file_path)

    def load(self):
        """Read the PDF with PyMuPDF."""
        return self.loader.load()
