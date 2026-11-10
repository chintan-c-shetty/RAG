from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All app settings in one place.

    Values come from (highest priority first): environment variables
    prefixed RAG_ (e.g. RAG_TOP_K=5), then a .env file, then these defaults.
    """

    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # data
    data_dir: str = "data"
    loader_name: str = "pymupdf"

    # splitting
    splitter_name: str = "recursive"
    chunk_size: int = 800
    chunk_overlap: int = 80

    # embedding
    embedder_name: str = "huggingface"
    embedder_model: str = "sentence-transformers/all-mpnet-base-v2"

    # vector store
    vectorstore_name: str = "qdrant"
    qdrant_url: str = "http://localhost:6333"
    collection_prefix: str = "pdf_docs"

    # llm
    llm_name: str = "ollama"
    llm_model: str = "qwen3:4b-instruct"
    temperature: float = 0.0

    # retrieval
    top_k: int = 3

    @property
    def collection_name(self) -> str:
        """Collection name that includes the embedding model (one collection per model)."""
        # TODO 1: build e.g. "pdf_docs__all-mpnet-base-v2" from collection_prefix + last part of embedder_model
        #         (Qdrant names: keep letters, digits, -, _ ; replace "/" and ":")
        ...


settings = Settings()
