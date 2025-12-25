import json
import logging
from pathlib import Path

from rag.config import settings
from rag.embeddings.embedder import Embedder
from rag.logging_config import configure_logging
from rag.retrieval.store import FaissVectorStore


def main() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)

    chunks_path = settings.data_processed_dir / "chunks.jsonl"
    store_path = settings.data_processed_dir / "vector_store"

    texts: list[str] = []
    metadatas: list[dict[str, str]] = []

    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            texts.append(record["text"])
            metadatas.append(
                {
                    "chunk_id": record["id"],
                    "document_id": record["document_id"],
                    "source": record["metadata"]["source"],
                }
            )

    logger.info("Embedding %d chunks", len(texts))

    embedder = Embedder(
        model_name=settings.embedding_model_name
    )
    vectors = embedder.embed_texts(texts)

    store = FaissVectorStore(dimension=len(vectors[0]))
    store.add(vectors, metadatas)
    store.save(store_path)

    logger.info("Vector store saved to %s", store_path)


if __name__ == "__main__":
    main()
