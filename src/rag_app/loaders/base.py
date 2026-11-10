from abc import ABC, abstractmethod
from langchain_core.documents import Document


class BasePDFLoader(ABC):
    """Contract that every PDF loader must follow.

    Subclasses wrap one PDF library and set `name`.
    """

    name: str = "base"

    @abstractmethod
    def load(self) -> list[Document]:
        """Read the PDF.

        Returns:
            One Document per page, with "source" and "page" in metadata.
        """
        ...
