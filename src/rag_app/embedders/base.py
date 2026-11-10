from abc import abstractmethod
from langchain_core.embeddings import Embeddings


class BaseEmbedder(Embeddings):
    """Contract that every embedder must follow.

    Subclasses wrap one embedding library and set `name`.
    """

    name: str = "base"

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many document chunks.

        Args:
            texts: Chunk texts to embed.
        Returns:
            One vector per text, in the same order.
        """
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query.

        Args:
            text: The user's question.
        Returns:
            One vector.
        """
        ...
