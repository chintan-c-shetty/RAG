from langchain_core.documents import Document

RAG_TEMPLATE = """Answer the question using ONLY the context below.
If the context does not contain the answer, reply exactly: I don't know.
Cite the sources you used by their [number].

Context:
{context}

Question: {question}

Answer:"""


def format_context(documents: list[Document]) -> str:
    """Join chunks into one numbered context block.

    Args:
        documents: Chunks from the vector store, most relevant first.
    Returns:
        Text like "[1] (data/file.pdf, page 0)\\n<chunk text>" separated by blank lines.
    """
    # TODO 1: number from 1; use metadata "source" and "page"
    ...


def build_rag_prompt(question: str, documents: list[Document]) -> str:
    """Fill RAG_TEMPLATE with the question and formatted context.

    Args:
        question: The user's question.
        documents: Retrieved chunks.
    Returns:
        The complete prompt text for the LLM.
    """
    # TODO 2: RAG_TEMPLATE.format(...)
    ...
