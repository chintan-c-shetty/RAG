from .ollama_llm import OllamaChatModel

LLMS = {
    "ollama": OllamaChatModel,
}


def get_llm(name: str, **kwargs):
    """Create an LLM by name.

    Args:
        name: Key in LLMS (case-insensitive).
        **kwargs: Passed to the LLM's constructor.
    Returns:
        A ready-to-use BaseLLM.
    Raises:
        ValueError: If the name is not registered.
    """
    name = name.strip().lower()

    llm_class = LLMS.get(name)

    if llm_class is None:
        available = ", ".join(LLMS.keys())
        raise ValueError(
            f"Unknown LLM '{name}'. Available LLMs: {available}"
        )

    return llm_class(**kwargs)