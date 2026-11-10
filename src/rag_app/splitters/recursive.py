from langchain_text_splitters import RecursiveCharacterTextSplitter
from .base import BaseSplitter


class RecursiveSplitter(BaseSplitter):
    """Split by paragraphs, then lines, then words, measured in characters.

    Args:
        chunk_size: Max characters per chunk.
        chunk_overlap: Characters repeated between neighbouring chunks.
    Raises:
        ValueError: If chunk_overlap >= chunk_size.
    """

    name = "recursive"

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 80):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
            raise TypeError("chunk_size must be an integer")

        if not isinstance(chunk_overlap, int) or isinstance(chunk_overlap, bool):
            raise TypeError("chunk_overlap must be an integer")

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def split_text(self, text):
        """Split a string by characters."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        return self.splitter.split_text(text)

    def split_documents(self, documents):
        """Split pages by characters."""
        return self.splitter.split_documents(documents)
