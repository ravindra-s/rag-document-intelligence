from sentence_transformers import SentenceTransformer

class Embedder:
    """
    Local CPU embedding wrapper.
    Model is auto-downloaded on first use.
    """

    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]