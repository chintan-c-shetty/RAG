from dataclasses import dataclass, field
from pathlib import Path

from ..loaders import get_pdf_loader
from ..splitters.base import BaseSplitter
from ..vectorstores.base import BaseVectorStore


@dataclass
class IngestReport:
    """Summary of one ingest run."""

    files_ok: dict[str, int] = field(default_factory=dict)      # path -> chunks stored
    files_failed: dict[str, str] = field(default_factory=dict)  # path -> error message

    @property
    def total_chunks(self) -> int:
        return sum(self.files_ok.values())


class IngestPipeline:
    """Index PDFs: load -> split -> replace old chunks -> store.

    Args:
        splitter: Any BaseSplitter.
        store: Any BaseVectorStore.
        loader_name: Name for get_pdf_loader (a loader is made per file).
    """

    def __init__(self, splitter: BaseSplitter, store: BaseVectorStore, loader_name: str = "pymupdf"):
        self.splitter = splitter
        self.store = store
        self.loader_name = loader_name

    def ingest_file(self, file_path: str) -> int:
        """Index one PDF, replacing any chunks it had before.

        Args:
            file_path: Path to a PDF.
        Returns:
            Number of chunks stored.
        Raises:
            ValueError: If the file can't be read or has no text.
        """
        # TODO 1: pages = get_pdf_loader(self.loader_name, file_path=file_path).load()
        # TODO 2: chunks, ids = self.splitter.split_with_ids(pages)
        # TODO 3: raise ValueError if no chunks (scanned PDF without text?)
        # TODO 4: self.store.delete_by_source(...)   <- why before adding? (see vectorstores guide)
        # TODO 5: self.store.add_documents(chunks, ids); return len(chunks)
        ...

    def ingest_folder(self, folder: str, pattern: str = "*.pdf") -> IngestReport:
        """Index every matching PDF in a folder; one bad file doesn't stop the rest.

        Args:
            folder: Directory to scan.
            pattern: Glob pattern (use "**/*.pdf" to include subfolders).
        Returns:
            IngestReport with per-file results.
        Raises:
            FileNotFoundError: If the folder doesn't exist.
        """
        # TODO 6: loop sorted(Path(folder).glob(pattern));
        #         try ingest_file -> report.files_ok[path] = n
        #         except Exception as e -> report.files_failed[path] = str(e)
        ...
