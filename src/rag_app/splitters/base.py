from abc import ABC, abstractmethod
from langchain_core.documents import Document


class BaseSplitter(ABC):
    """Contract that every splitter must follow.

    Subclasses wrap one splitting strategy and set `name`.
    Chunks keep the metadata of the page they came from.
    """

    name: str = "base"

    # ---------- must implement ----------

    @abstractmethod
    def split_text(self, text: str) -> list[str]:
        """Split one plain string into chunks.

        Args:
            text: Any text.
        Returns:
            Chunk strings, in reading order.
        """
        ...

    @abstractmethod
    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split pages into chunks, keeping each page's metadata.

        Args:
            documents: Pages from a loader.
        Returns:
            Chunk Documents, in reading order.
        """
        ...

    # ---------- shared (already works for every subclass) ----------

    def split_with_ids(self, documents: list[Document]) -> tuple[list[Document], list[str]]:
        """Split pages and give every chunk a stable id "source:page:index".

        Args:
            documents: Pages from a loader.
        Returns:
            (chunks, ids), ready for BaseVectorStore.add_documents.
        Raises:
            KeyError: If a page has no "source" or "page" metadata.
        """
        chunks = self.split_documents(documents)
        ids: list[str] = []
        previous_source = None
        previous_page = None
        chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata["source"]
            page = chunk.metadata["page"]

            # Reset index when source or page changes.
            if (source, page) != (previous_source, previous_page):
                chunk_index = 0
                previous_source = source
                previous_page = page

            chunk_id = f"{source}:{page}:{chunk_index}"

            # Store ID in chunk metadata.
            chunk.metadata["id"] = chunk_id

            ids.append(chunk_id)
            chunk_index += 1

        return chunks, ids

        
