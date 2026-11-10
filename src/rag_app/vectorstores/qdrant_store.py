import uuid
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore
from .base import BaseVectorStore

SOURCE_KEY = "metadata.source"   # LangChain nests metadata under "metadata" in the payload


class QdrantStore(BaseVectorStore):
    """Vector store backed by a Qdrant server.

    How a chunk is stored (one chunk = one Qdrant "point"):
        id      -> UUID made from the chunk id "source:page:index" (see _to_uuid)
        vector  -> embedder.embed_documents([chunk.page_content])[0]
        payload -> {"page_content": "...", "metadata": {"source": ..., "page": ..., "id": ...}}

    Args:
        embedder: Any BaseEmbedder; its vector size fixes the collection size.
        collection_name: Qdrant collection to use (created if missing).
        url: Qdrant server URL.
    Raises:
        QdrantVectorStoreError: If the collection exists but was built with a
            different vector size (i.e. a different embedding model).
    """

    name = "qdrant"

    def __init__(self, embedder, collection_name: str, url: str = "http://localhost:6333"):
        self.collection_name = collection_name
        self.client = QdrantClient(url=url)

        # 1. Create the collection on first use.
        #    The vector size comes from the embedder itself (embed one word and
        #    measure it), so switching models never needs a code change.
        if not self.client.collection_exists(collection_name):
            vector_size = len(embedder.embed_query("dimension probe"))
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,   # higher score = more similar
                ),
            )

        # 2. Index metadata.source so filtering by file (delete_by_source) stays
        #    fast on big collections. Only created if it isn't there yet.
        payload_schema = self.client.get_collection(collection_name).payload_schema
        if SOURCE_KEY not in payload_schema:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=SOURCE_KEY,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

        # 3. LangChain wrapper: embeds text and converts points <-> Documents.
        #    It also checks the existing collection's vector size matches the
        #    embedder and raises if not.
        self.store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=embedder,
        )

    @staticmethod
    def _to_uuid(chunk_id: str) -> str:
        """Turn a text id into a stable UUID (Qdrant only accepts int/UUID ids).

        uuid5 is deterministic: the same chunk id always gives the same UUID,
        so re-ingesting a file overwrites its points instead of duplicating them.

        Example:
            "data/a.pdf:0:3" -> "6f1c...-...." (always the same value)
        """
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    # ---------- write ----------

    def add_documents(self, documents: list[Document], ids: list[str]) -> None:
        """Store chunks in Qdrant (upsert: an existing id is overwritten)."""
        if len(documents) != len(ids):
            raise ValueError(
                f"documents and ids must have the same length "
                f"(got {len(documents)} documents and {len(ids)} ids)"
            )
        if not documents:
            return

        self.store.add_documents(documents, ids=[self._to_uuid(i) for i in ids])

    # ---------- read ----------

    def search(self, query: str, k: int = 3) -> list[Document]:
        """Similarity search in Qdrant; most similar first."""
        return self.store.similarity_search(query, k=k)

    def search_with_scores(self, query: str, k: int = 3) -> list[tuple[Document, float]]:
        """Similarity search in Qdrant, with cosine scores (higher = more similar)."""
        return self.store.similarity_search_with_score(query, k=k)

    def get_by_ids(self, ids: list[str]) -> list[Document]:
        """Fetch chunks from Qdrant by their chunk ids; missing ids are skipped."""
        if not ids:
            return []
        return self.store.get_by_ids([self._to_uuid(i) for i in ids])

    def count(self) -> int:
        """Count points in the Qdrant collection (exact count)."""
        return self.client.count(collection_name=self.collection_name, exact=True).count

    # ---------- delete ----------

    def delete_by_ids(self, ids: list[str]) -> None:
        """Delete selected points from Qdrant; unknown ids are ignored."""
        if not ids:
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=[self._to_uuid(i) for i in ids]),
        )

    def delete_by_source(self, source: str) -> None:
        """Delete all points whose metadata.source matches.

        Use before re-ingesting a changed file, so chunks that no longer
        exist in the new version don't stay behind as stale results.
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=SOURCE_KEY,
                            match=models.MatchValue(value=source),
                        )
                    ]
                )
            ),
        )

    def delete_collection(self) -> None:
        """Drop this Qdrant collection.

        After this, the instance is unusable; create a new QdrantStore to
        recreate the collection.
        """
        self.client.delete_collection(collection_name=self.collection_name)

    # ---------- server level ----------

    def list_collections(self) -> list[str]:
        """List collection names on the Qdrant server."""
        return [c.name for c in self.client.get_collections().collections]

    def delete_all_collections(self, confirm: bool = False) -> list[str]:
        """Drop every collection on the Qdrant server (including this one)."""
        if confirm is not True:
            raise ValueError("Refusing to delete all collections: pass confirm=True.")

        names = self.list_collections()
        for name in names:
            self.client.delete_collection(collection_name=name)
        return names
