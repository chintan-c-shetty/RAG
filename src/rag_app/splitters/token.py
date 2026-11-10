from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .base import BaseSplitter


class TokenSplitter(BaseSplitter):
    """Same splitting rules as RecursiveSplitter, but measured in model tokens.

    Use this to guarantee chunks fit the embedder's token limit
    (e.g. 384 for all-mpnet-base-v2).

    Args:
        tokenizer_name: Hugging Face id; should match the embedder's model.
        chunk_size: Max tokens per chunk.
        chunk_overlap: Tokens repeated between neighbouring chunks.
    """

    name = "token"

    def __init__(
        self,
        tokenizer_name: str = "sentence-transformers/all-mpnet-base-v2",
        chunk_size: int = 256,
        chunk_overlap: int = 32,
    ):
        self.tokenizer_name = tokenizer_name
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

        # Load the Hugging Face tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Create a recursive splitter whose length is measured in tokens.
        self.splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self.tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_text(self, text):
        """Split a string by tokens."""
        if not text:
            return []

        return self.splitter.split_text(text)


    def split_documents(self, documents):
        """Split pages by tokens."""
        if not documents:
            return []

        return self.splitter.split_documents(documents)
