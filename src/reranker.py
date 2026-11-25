import os

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:

    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self):

        self.model = None

    def _load(self):

        if self.model is not None:
            return

        try:
            self.model = CrossEncoder(self.MODEL_NAME)
        except Exception:
            # Offline or missing model: fall back to identity (no re-ranking)
            self.model = "fallback"

    def rerank(self, query, candidates, top_n=5):

        self._load()

        if self.model == "fallback" or not candidates:
            return candidates[:top_n]

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)

        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [c for c, _ in scored[:top_n]]