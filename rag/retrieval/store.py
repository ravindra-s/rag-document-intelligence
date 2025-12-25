import json
from pathlib import Path

import faiss
import numpy as np


class FaissVectorStore:
    def __init__(self, dimension: int) -> None:
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: list[dict[str, str]] = []

    def add(
        self,
        vectors: list[list[float]],
        metadatas: list[dict[str, str]],
    ) -> None:
        array = np.array(vectors).astype("float32")
        self.index.add(array)
        self.metadata.extend(metadatas)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with (path / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(self.metadata, f)

    @classmethod
    def load(cls, path: Path) -> "FaissVectorStore":
        index = faiss.read_index(str(path / "index.faiss"))
        with (path / "metadata.json").open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        store = cls(index.d)
        store.index = index
        store.metadata = metadata
        return store

    def search(
        self,
        query_vector: list[float],
        k: int = 5,
    ) -> list[dict[str, str]]:
        array = np.array([query_vector]).astype("float32")
        scores, indices = self.index.search(array, k)

        results: list[dict[str, str]] = []
        for idx in indices[0]:
            if idx != -1:
                results.append(self.metadata[idx])

        return results