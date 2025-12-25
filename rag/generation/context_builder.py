from typing import Any


def build_context(
    retrieved: list[dict[str, Any]],
    chunk_texts: dict[str, str],
    *,
    context_k: int,
) -> tuple[str, list[str]]:
    """
    Select top context_k chunks (truncation) and assemble context text.
    Returns (context_text, source_chunk_ids).
    """
    selected = retrieved[:context_k]

    parts: list[str] = []
    sources: list[str] = []

    for item in selected:
        cid = item["chunk_id"]
        text = chunk_texts.get(cid, "")
        if text:
            parts.append(text)
            sources.append(cid)

    context_text = "\n\n---\n\n".join(parts)
    return context_text, sources
