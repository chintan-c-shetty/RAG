from .base import BasePDFLoader
from langchain_community.document_loaders import PyPDFLoader


class PyPDFDocumentLoader(BasePDFLoader):
    """PDF loader using the pypdf library.

    Args:
        file_path: Path to one PDF file.
    Raises:
        ValueError: If the file does not exist.
    """

    name = "pypdf"

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.loader = PyPDFLoader(self.file_path)

    def load(self):
        """Read the PDF with pypdf."""
        return self.loader.load()
