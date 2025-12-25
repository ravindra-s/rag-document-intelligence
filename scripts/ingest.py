import logging
from pathlib import Path

from rag.config import settings
from rag.ingestion.loader import load_pdfs_from_dir
from rag.ingestion.serializer import save_documents
from rag.logging_config import configure_logging


def main() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)

    raw_dir: Path = settings.data_raw_dir
    output_file: Path = settings.data_processed_dir / "documents.jsonl"

    logger.info("Loading PDFs from %s", raw_dir)
    documents = load_pdfs_from_dir(raw_dir)

    logger.info("Saving %d documents to %s", len(documents), output_file)
    save_documents(documents, output_file)

    logger.info("Ingestion completed successfully")


if __name__ == "__main__":
    main()
