from dataclasses import dataclass, field
from collections.abc import Iterator
from langchain_core.documents import Document

from ..llms.base import BaseLLM
from ..vectorstores.base import BaseVectorStore
from ..prompts import build_rag_prompt


@dataclass
class RAGAnswer:
    """Result of one question."""

    answer: str
    sources: list[str] = field(default_factory=list)       # chunk ids used
    chunks: list[Document] = field(default_factory=list)   # full retrieved chunks


class RAGChain:
    """Answer questions: search store -> build prompt -> ask LLM.

    Args:
        store: Any BaseVectorStore (already filled by ingest).
        llm: Any BaseLLM.
        top_k: How many chunks to retrieve per question.
        min_score: Drop chunks below this similarity (0 = keep all).
    """

    def __init__(self, store: BaseVectorStore, llm: BaseLLM, top_k: int = 3, min_score: float = 0.0):
        self.store = store
        self.llm = llm
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(self, question: str) -> list[Document]:
        """Get relevant chunks for a question.

        Returns:
            Chunks with score >= min_score, most relevant first.
        """
        # TODO 1: self.store.search_with_scores(...), keep only score >= self.min_score
        ...

    def ask(self, question: str) -> RAGAnswer:
        """Answer a question from the stored documents.

        Returns:
            RAGAnswer with the answer text, source ids and chunks.
        """
        # TODO 2: chunks = self.retrieve(question)
        # TODO 3: if no chunks -> return RAGAnswer("I don't know.") WITHOUT calling the LLM
        # TODO 4: prompt = build_rag_prompt(...); answer = self.llm.generate(prompt)
        # TODO 5: sources = [c.metadata.get("id") for c in chunks]; return RAGAnswer(...)
        ...

    def ask_stream(self, question: str) -> tuple[Iterator[str], list[Document]]:
        """Like ask(), but stream the answer for live output.

        Returns:
            (iterator of answer pieces, retrieved chunks)
        """
        # TODO 6: same as ask(), using self.llm.stream(prompt)
        ...
