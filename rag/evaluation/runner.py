import json
import logging
from pathlib import Path
from typing import Any

from rag.config import settings
from rag.embeddings.embedder import Embedder
from rag.generation.context_builder import build_context
from rag.generation.llm import LocalLLM
from rag.generation.prompts import grounded_qa_prompt
from rag.retrieval.store import FaissVectorStore


def _load_chunk_texts(chunks_path: Path) -> dict[str, str]:
    chunk_map: dict[str, str] = {}
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            chunk_map[rec["id"]] = rec["text"]
    return chunk_map


class EvaluationRunner:
    """
    Runs RAG for a batch of questions using current config.
    Reuses FAISS index and chunk texts; embeds queries on demand.
    """

    def __init__(
        self,
        *,
        retrieval_k: int,
        context_k: int,
        max_new_tokens: int,
    ) -> None:
        self.retrieval_k = retrieval_k
        self.context_k = context_k
        self.max_new_tokens = max_new_tokens

        self.store_path = settings.data_processed_dir / "vector_store"
        self.chunks_path = settings.data_processed_dir / "chunks.jsonl"

        if not self.store_path.exists():
            raise FileNotFoundError(
                "Vector store not found. Run `python scripts/embed.py` first."
            )
        if not self.chunks_path.exists():
            raise FileNotFoundError(
                "chunks.jsonl not found. Run `python scripts/chunk.py` first."
            )

        # Load heavy assets once
        self.embedder = Embedder(model_name=settings.embedding_model_name)
        self.store = FaissVectorStore.load(self.store_path)
        self.chunk_texts = _load_chunk_texts(self.chunks_path)
        self.llm = LocalLLM(model_name=settings.llm_model_name)

    def answer_one(self, question: str) -> str:
        # Retrieval (breadth)
        qvec = self.embedder.embed_query(question)
        retrieved = self.store.search(qvec, k=self.retrieval_k)

        # Context selection (precision)
        context, _sources = build_context(
            retrieved,
            self.chunk_texts,
            context_k=self.context_k,
        )

        prompt = grounded_qa_prompt(context, question)
        return self.llm.generate(
            prompt,
            max_new_tokens=self.max_new_tokens,
        ).strip()

    def run(self, questions: list[str]) -> list[str]:
        logger = logging.getLogger(__name__)

        total = len(questions)
        answers: list[str] = []

        for idx, q in enumerate(questions, start=1):
            logger.info("Processing question %d/%d", idx, total)

            if not q.strip():
                answers.append("")
                continue

            try:
                answer = self.answer_one(q)
                logger.info(f"Answer: {answer}")
            except Exception as exc:
                logger.error(
                    "Failed on question %d/%d: %s", idx, total, exc
                )
                answer = ""

            answers.append(answer)

        return answers

