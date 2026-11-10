from .huggingface import HuggingFaceEmbedder
from .ollama import OllamaEmbedder

EMBEDDERS = {
    "huggingface": HuggingFaceEmbedder,
    "ollama": OllamaEmbedder,
}


def get_embedder(name: str, **kwargs):
    """Create an embedder by name.

    Args:
        name: Key in EMBEDDERS (case-insensitive).
        **kwargs: Passed to the embedder's constructor.
    Returns:
        A ready-to-use BaseEmbedder.
    Raises:
        ValueError: If the name is not registered.
    """
    embedder_class = EMBEDDERS.get(name.lower())

    if embedder_class is None:
        available = ", ".join(EMBEDDERS.keys())
        raise ValueError(
            f"Unknown embedder '{name}'. "
            f"Available embedders: {available}"
        )

    return embedder_class(**kwargs)
