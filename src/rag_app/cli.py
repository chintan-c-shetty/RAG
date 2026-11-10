"""Command line entry point.

Usage:
    rag ingest              # index every PDF in settings.data_dir
    rag ask "question"      # answer one question
    rag chat                # ask questions in a loop
"""
import argparse

from .config import settings


def build_components():
    """Create splitter, embedder, store and llm from settings (the only place that wires modules)."""
    # TODO 1: use get_splitter / get_embedder / get_vector_store / get_llm with values from settings
    #         return splitter, store, llm
    ...


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="index PDFs from the data folder")
    ask = sub.add_parser("ask", help="answer one question")
    ask.add_argument("question")
    sub.add_parser("chat", help="interactive questions")
    args = parser.parse_args()

    # TODO 2: splitter, store, llm = build_components()
    # TODO 3: "ingest" -> IngestPipeline(...).ingest_folder(settings.data_dir); print the report
    # TODO 4: "ask"    -> RAGChain(...).ask(args.question); print answer + sources
    # TODO 5: "chat"   -> loop input() until empty line / Ctrl+C, stream answers


if __name__ == "__main__":
    main()
