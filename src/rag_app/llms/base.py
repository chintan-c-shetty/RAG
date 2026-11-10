from abc import ABC, abstractmethod
from collections.abc import Iterator


class BaseLLM(ABC):
    """Contract that every text-generation model must follow.

    Subclasses wrap one LLM provider and set `name`.
    """

    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a full answer.

        Args:
            prompt: Complete prompt text.
        Returns:
            The model's answer text.
        """
        ...

    @abstractmethod
    def stream(self, prompt: str) -> Iterator[str]:
        """Generate an answer piece by piece (for live output).

        Args:
            prompt: Complete prompt text.
        Returns:
            Iterator of text pieces; joined, they equal generate()'s answer.
        """
        ...
