from typing import Optional
from .base import BaseEmbedder
from langchain_huggingface import HuggingFaceEmbeddings


class HuggingFaceEmbedder(BaseEmbedder):
    """Embedder that runs a sentence-transformers model locally.

    Args:
        model_name: Hugging Face model id.
        model_kwargs: Passed to the model on load (e.g. {"device": "cpu"}).
        encode_kwargs: Options used when embedding documents.
        query_encode_kwargs: Options used when embedding queries.
    """

    name = "huggingface"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        model_kwargs: Optional[dict] = None,
        encode_kwargs: Optional[dict] = None,
        query_encode_kwargs: Optional[dict] = None,
    ):
        self.model_name = model_name

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=model_kwargs or {},
            encode_kwargs=encode_kwargs or {},
            query_encode_kwargs=query_encode_kwargs or {},
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks with the loaded model."""
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a query with the loaded model."""
        return self.embeddings.embed_query(text)
