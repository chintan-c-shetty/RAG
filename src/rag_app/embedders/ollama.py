from .base import BaseEmbedder
from langchain_ollama import OllamaEmbeddings


class OllamaEmbedder(BaseEmbedder):
    """Embedder that calls a model served by a local Ollama server.

    Args:
        model_name: Ollama model tag (must be pulled first).
        base_url: Ollama server URL.
        keep_alive: Seconds the model stays loaded after a call.
    """

    name = "ollama"

    def __init__(
        self,
        model_name: str = "qwen3-embedding:0.6b",
        base_url: str = "http://localhost:11434",
        keep_alive: int = 300,
    ):
        self.model_name = model_name

        self.embeddings = OllamaEmbeddings(
            model=self.model_name,
            base_url=base_url,
            keep_alive=keep_alive,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks through Ollama."""
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a query through Ollama."""
        return self.embeddings.embed_query(text)
