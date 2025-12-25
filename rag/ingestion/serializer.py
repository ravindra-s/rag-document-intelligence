import json
from pathlib import Path
from collections.abc import Iterable

from rag.models import Document


def save_documents(
    documents: Iterable[Document],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for doc in documents:
            record = {
                "id": doc.id,
                "text": doc.text,
                "source": str(doc.source),
                "metadata": doc.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
