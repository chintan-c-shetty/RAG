from .qdrant_store import QdrantStore

VECTOR_STORES = {
    "qdrant": QdrantStore,
}


def get_vector_store(name: str, **kwargs):
    """Create a vector store by name.

    Args:
        name: Key in VECTOR_STORES (case-insensitive).
        **kwargs: Passed to the store's constructor.
    Returns:
        A ready-to-use BaseVectorStore.
    Raises:
        ValueError: If the name is not registered.
    """
    name = name.strip().lower()

    store_class = VECTOR_STORES.get(name)

    if store_class is None:
        available = ", ".join(VECTOR_STORES.keys())
        raise ValueError(
            f"Unknown vector store '{name}'. Available vector stores: {available}"
        )

    return store_class(**kwargs)
