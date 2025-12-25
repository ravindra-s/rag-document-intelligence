import json
import logging
from pathlib import Path

from rag.config import settings
from rag.ingestion.chunker import chunk_documents
from rag.logging_config import configure_logging
from rag.models import Document


def _load_documents(path: Path) -> list[Document]:
    documents: list[Document] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            documents.append(
                Document(
                    id=record["id"],
                    text=record["text"],
                    source=Path(record["source"]),
                    metadata=record["metadata"],
                )
            )

    return documents


def main() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)

    input_path = settings.data_processed_dir / "documents.jsonl"
    output_path = settings.data_processed_dir / "chunks.jsonl"

    documents = _load_documents(input_path)
    chunks = chunk_documents(documents)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(
                json.dumps(
                    {
                        "id": chunk.id,
                        "document_id": chunk.document_id,
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    logger.info(
        "Chunking complete: %d documents → %d chunks",
        len(documents),
        len(chunks),
    )


if __name__ == "__main__":
    main()
