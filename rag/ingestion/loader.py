import logging
from pathlib import Path

from pypdf import PdfReader
from rag.models import Document

logger = logging.getLogger(__name__)


def load_pdf(path: Path) -> Document:
    if not path.exists() or path.suffix.lower() != ".pdf":
        raise ValueError(f"Invalid PDF path: {path}")

    reader = PdfReader(path)
    pages_text: list[str] = []

    for page_number, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        cleaned = text.strip()

        if cleaned:
            pages_text.append(cleaned)
        else:
            logger.warning(
                "Empty text on page %d in %s",
                page_number,
                path.name,
            )

    full_text = "\n\n".join(pages_text)

    if not full_text:
        raise ValueError(f"No extractable text found in {path.name}")

    return Document(
        id=path.stem,
        text=full_text,
        source=path,
        metadata={
            "filename": path.name,
            "num_pages": len(reader.pages),
        },
    )


def load_pdfs_from_dir(directory: Path) -> list[Document]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    documents: list[Document] = []

    for pdf_path in sorted(directory.glob("*.pdf")):
        try:
            documents.append(load_pdf(pdf_path))
            logger.info("Loaded PDF: %s", pdf_path.name)
        except Exception as exc:
            logger.error("Failed to load %s: %s", pdf_path.name, exc)

    if not documents:
        raise RuntimeError("No valid PDF documents loaded")

    return documents
