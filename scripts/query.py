import argparse
import json
import logging
from pathlib import Path

from rag.config import settings
from rag.embeddings.embedder import Embedder
from rag.generation.context_builder import build_context
from rag.generation.llm import LocalLLM
from rag.generation.prompts import grounded_qa_prompt
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
        description="RAG query: retrieval + context selection + local LLM generation"
    )
    parser.add_argument("--retrieval-k", type=int, default=8)
    parser.add_argument("--context-k", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    store_path = settings.data_processed_dir / "vector_store"
    chunks_path = settings.data_processed_dir / "chunks.jsonl"

    if not store_path.exists():
        raise FileNotFoundError(
            "Vector store not found. Run `python scripts/embed.py` first."
        )
    if not chunks_path.exists():
        raise FileNotFoundError(
            "chunks.jsonl not found. Run `python scripts/chunk.py` first."
        )

    embedder = Embedder(
        model_name=settings.embedding_model_name
    )
    store = FaissVectorStore.load(store_path)
    chunk_texts = _load_chunk_texts(chunks_path)

    question = input("Enter question: ").strip()
    if not question:
        logger.warning("Empty question provided")
        return

    # Retrieval (breadth)
    query_vec = embedder.embed_query(question)
    retrieved = store.search(query_vec, k=args.retrieval_k)

    # Context selection (precision)
    context, sources = build_context(
        retrieved,
        chunk_texts,
        context_k=args.context_k,
    )

    prompt = grounded_qa_prompt(context, question)

    llm = LocalLLM(model_name=settings.llm_model_name)
    answer = llm.generate(
        prompt,
        max_new_tokens=args.max_new_tokens,
    )

    logger.info(answer)

    #print("\n=== Sources (chunk_ids) ===")
    #for cid in sources:
    #    print("-", cid)


if __name__ == "__main__":
    main()
