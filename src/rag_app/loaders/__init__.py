from .pypdf_loader import PyPDFDocumentLoader
from .pymupdf_loader import PyMuPDFDocumentLoader


PDF_LOADERS = {
    "pypdf": PyPDFDocumentLoader,
    "pymupdf": PyMuPDFDocumentLoader,
}


def get_pdf_loader(name: str, **kwargs):
    """Create a PDF loader by name.

    Args:
        name: Key in PDF_LOADERS (case-insensitive).
        **kwargs: Passed to the loader's constructor (e.g. file_path).
    Returns:
        A ready-to-use BasePDFLoader.
    Raises:
        ValueError: If the name is not registered.
    """
    loader_class = PDF_LOADERS.get(name.lower())

    if loader_class is None:
        available = ", ".join(PDF_LOADERS.keys())
        raise ValueError(
            f"Unknown PDF loader '{name}'. "
            f"Available loaders: {available}"
        )

    return loader_class(**kwargs)
