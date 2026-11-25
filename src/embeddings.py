from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingModel:

    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def embed_documents(self, chunks):
        embedding_texts = []

        for chunk in chunks:
            metadata = chunk["metadata"]

            embedding_text = (
                f"Document: {metadata['source']}\n"
                f"Section: {metadata['section']}\n\n"
                f"{chunk['text']}"
            )

            embedding_texts.append(embedding_text)

        return self.model.encode(
            embedding_texts,
            normalize_embeddings=True
        )

    def embed_query(self, query):
        return self.model.encode(
            query,
            normalize_embeddings=True
        )