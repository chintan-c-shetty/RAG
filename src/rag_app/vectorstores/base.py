from abc import ABC, abstractmethod
from langchain_core.documents import Document


class BaseVectorStore(ABC):
    """Contract that every vector store must follow.

    Subclasses wrap one vector database and set `name`.
    One instance works on one collection; server-level operations
    (list / delete all collections) are also included.
    """

    name: str = "base"

    # ---------- write ----------

    @abstractmethod
    def add_documents(self, documents: list[Document], ids: list[str]) -> None:
        """Store chunks; an existing id is overwritten, not duplicated.

        Args:
            documents: Chunks to embed and store.
            ids: One stable id per chunk, same order as documents.
        Raises:
            ValueError: If documents and ids have different lengths.
        """
        ...

    # ---------- read ----------

    @abstractmethod
    def search(self, query: str, k: int = 3) -> list[Document]:
        """Find the chunks most similar to a query.

        Args:
            query: The user's question.
            k: How many chunks to return.
        Returns:
            Up to k Documents, most similar first.
        """
        ...

    @abstractmethod
    def search_with_scores(self, query: str, k: int = 3) -> list[tuple[Document, float]]:
        """Same as search, but also return the similarity score.

        Args:
            query: The user's question.
            k: How many chunks to return.
        Returns:
            Up to k (Document, score) pairs, highest score first.
        """
        ...

    @abstractmethod
    def get_by_ids(self, ids: list[str]) -> list[Document]:
        """Fetch stored chunks by their ids.

        Args:
            ids: Chunk ids as given to add_documents.
        Returns:
            The Documents found; missing ids are skipped.
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the number of chunks in this collection."""
        ...

    # ---------- delete ----------

    @abstractmethod
    def delete_by_ids(self, ids: list[str]) -> None:
        """Delete selected chunks.

        Args:
            ids: Chunk ids as given to add_documents.
        """
        ...

    @abstractmethod
    def delete_by_source(self, source: str) -> None:
        """Delete every chunk that came from one file.

        Args:
            source: Value of metadata["source"], e.g. "data/file.pdf".
        """
        ...

    @abstractmethod
    def delete_collection(self) -> None:
        """Delete this collection and all its chunks."""
        ...

    # ---------- server level ----------

    @abstractmethod
    def list_collections(self) -> list[str]:
        """Return the names of all collections on the server."""
        ...

    @abstractmethod
    def delete_all_collections(self, confirm: bool = False) -> list[str]:
        """Delete every collection on the server.

        Args:
            confirm: Must be True, to prevent accidental wipes.
        Returns:
            Names of the deleted collections.
        Raises:
            ValueError: If confirm is not True.
        """
        ...
