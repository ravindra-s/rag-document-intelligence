import argparse
import json
import logging
from pathlib import Path

from rag.config import settings
from rag.embeddings.embedder import Embedder
from rag.logging_config import configure_logging
from rag.retrieval.store import FaissVectorStore


def _load_chunk_texts(chunks_path: Path) -> dict[str, str]:
    chunk_map: dict[str, str] = {}
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            chunk_map[record["id"]] = record["text"]
    return chunk_map


def main() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Semantic search over indexed document chunks"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of top results to return",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Show readable chunk text for each result",
    )

    args = parser.parse_args()

    store_path = settings.data_processed_dir / "vector_store"
    chunks_path = settings.data_processed_dir / "chunks.jsonl"

    if not store_path.exists():
        raise FileNotFoundError(
            "Vector store not found. Run `python scripts/embed.py` first."
        )

    embedder = Embedder(
        model_name=settings.embedding_model_name
    )
    store = FaissVectorStore.load(store_path)

    chunk_texts: dict[str, str] = {}
    if args.inspect:
        if not chunks_path.exists():
            raise FileNotFoundError(
                "chunks.jsonl not found. Run `python scripts/chunk.py` first."
            )
        chunk_texts = _load_chunk_texts(chunks_path)

    query = input("Enter query: ").strip()
    if not query:
        logger.warning("Empty query provided")
        return

    query_vector = embedder.embed_query(query)
    results = store.search(query_vector, k=args.k)

    print("\n=== Top Results ===")
    for rank, result in enumerate(results, start=1):
        print(f"\n[{rank}]")
        print(f"Chunk ID   : {result['chunk_id']}")
        print(f"Document  : {result['document_id']}")
        print(f"Source    : {result['source']}")

        if args.inspect:
            text = chunk_texts.get(result["chunk_id"], "")
            print("\n--- Chunk Text (truncated) ---")
            print(text[:800].strip())
            print("--- End ---")


if __name__ == "__main__":
    main()
