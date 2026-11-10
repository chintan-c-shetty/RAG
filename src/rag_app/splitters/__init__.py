from .recursive import RecursiveSplitter
from .token import TokenSplitter

SPLITTERS = {
    "recursive": RecursiveSplitter,
    "token": TokenSplitter,
}


def get_splitter(name: str, **kwargs):
    """Create a splitter by name.

    Args:
        name: Key in SPLITTERS (case-insensitive).
        **kwargs: Passed to the splitter's constructor.
    Returns:
        A ready-to-use BaseSplitter.
    Raises:
        ValueError: If the name is not registered.
    """
    name = name.strip().lower()

    splitter_class = SPLITTERS.get(name)

    if splitter_class is None:
        available = ", ".join(SPLITTERS.keys())
        raise ValueError(
            f"Unknown splitter '{name}'. Available splitters: {available}"
        )

    return splitter_class(**kwargs)