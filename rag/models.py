from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    source: Path
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class EmbeddingRecord:
    chunk_id: str
    vector: list[float]
    metadata: dict[str, Any]
