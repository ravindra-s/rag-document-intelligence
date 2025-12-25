import hashlib
from pathlib import Path
from typing import Any

from rag.models import Chunk, Document


def _stable_chunk_id(document_id: str, index: int, text: str) -> str:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{document_id}_chunk_{index}_{h}"


def chunk_document(
    document: Document,
    *,
    max_chars: int = 1200,
    overlap_chars: int = 150,
) -> list[Chunk]:
    """
    Chunk a Document into overlapping text chunks.

    Strategy:
    - Split by paragraphs first
    - Merge paragraphs until max_chars is reached
    - Apply character overlap between consecutive chunks
    """

    paragraphs = [
        p.strip()
        for p in document.text.split("\n\n")
        if p.strip()
    ]

    chunks: list[Chunk] = []
    buffer: str = ""
    chunk_index = 0

    for paragraph in paragraphs:
        if len(buffer) + len(paragraph) + 2 <= max_chars:
            buffer = f"{buffer}\n\n{paragraph}" if buffer else paragraph
        else:
            chunk_text = buffer
            chunk_id = _stable_chunk_id(
                document.id, chunk_index, chunk_text
            )

            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=document.id,
                    text=chunk_text,
                    metadata={
                        **document.metadata,
                        "chunk_index": chunk_index,
                        "source": str(document.source),
                    },
                )
            )

            chunk_index += 1
            buffer = buffer[-overlap_chars:] + "\n\n" + paragraph

    if buffer.strip():
        chunk_id = _stable_chunk_id(document.id, chunk_index, buffer)
        chunks.append(
            Chunk(
                id=chunk_id,
                document_id=document.id,
                text=buffer,
                metadata={
                    **document.metadata,
                    "chunk_index": chunk_index,
                    "source": str(document.source),
                },
            )
        )

    return chunks


def chunk_documents(
    documents: list[Document],
    *,
    max_chars: int = 1200,
    overlap_chars: int = 150,
) -> list[Chunk]:
    all_chunks: list[Chunk] = []

    for doc in documents:
        doc_chunks = chunk_document(
            doc,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )
        all_chunks.extend(doc_chunks)

    return all_chunks
