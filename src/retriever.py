import chromadb
from rank_bm25 import BM25Okapi

from .embeddings import EmbeddingModel


COLLECTION_NAME = "university_policies"


class Retriever:

    # Initial relevance threshold.
    # Lower Chroma distance = more similar.
    RELEVANCE_THRESHOLD = 1.25

    def __init__(self):

        self.embedding_model = EmbeddingModel()

        self.client = chromadb.PersistentClient(
            path="./chroma_db"
        )

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME
        )

        self._bm25 = None
        self._bm25_docs = None
        self._bm25_metadatas = None

    def _build_bm25_index(self):

        if self._bm25 is not None:
            return

        all_data = self.collection.get()
        documents = all_data["documents"]
        metadatas = all_data.get("metadatas", [{} for _ in documents])

        tokenized = [doc.split() for doc in documents]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_docs = documents
        self._bm25_metadatas = metadatas

    def retrieve_hybrid(self, query, top_k=10):

        self._build_bm25_index()

        # Vector search
        vector_results = self.retrieve(query, top_k=top_k)

        # BM25 search
        tokenized_query = query.split()
        bm25_scores = self._bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:top_k]

        bm25_results = []
        for i in top_indices:
            bm25_results.append(
                {
                    "text": self._bm25_docs[i],
                    "metadata": self._bm25_metadatas[i] or {},
                    "distance": 1.0,
                }
            )

        # Reciprocal Rank Fusion
        rrf_scores = {}

        for rank, result in enumerate(vector_results):
            text = result["text"]
            rrf_scores[text] = rrf_scores.get(text, 0.0) + 1.0 / (60 + rank + 1)

        for rank, result in enumerate(bm25_results):
            text = result["text"]
            rrf_scores[text] = rrf_scores.get(text, 0.0) + 1.0 / (60 + rank + 1)

        # Fuse and sort by RRF score descending
        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        all_results = {r["text"]: r for r in vector_results + bm25_results}

        final = []
        for text, score in fused[:top_k]:
            base = all_results[text]
            final.append(
                {
                    "text": base["text"],
                    "metadata": base.get("metadata", {}),
                    "distance": base.get("distance", 1.0),
                    "rrf_score": score,
                }
            )

        return final

    def retrieve(self, query, top_k=5):

        # Convert the user's question into an embedding vector.
        query_embedding = self.embedding_model.embed_query(query)

        # Search Chroma for the nearest chunks.
        search_results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )

        retrieved = []

        documents = search_results["documents"][0]
        metadatas = search_results["metadatas"][0]
        distances = search_results["distances"][0]

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):

            retrieved.append(
                {
                    "text": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

        return retrieved

    def is_relevant(self, results):

        if not results:
            return False

        best_distance = results[0]["distance"]

        return best_distance <= self.RELEVANCE_THRESHOLD

    def retrieve_with_gate(self, query, top_k=5):

        results = self.retrieve(
            query,
            top_k=top_k,
        )

        relevant = self.is_relevant(results)

        return {
            "relevant": relevant,
            "results": results,
        }


if __name__ == "__main__":

    retriever = Retriever()

    while True:

        question = input("\nAsk a question: ").strip()

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            break

        result = retriever.retrieve_with_gate(question)

        print("\n" + "=" * 80)

        if not result["relevant"]:

            print("NO RELEVANT EVIDENCE FOUND")
            print("=" * 80)

            print(
                "I could not find sufficiently relevant information "
                "in the provided documents."
            )

            continue

        print("RELEVANT EVIDENCE FOUND")
        print("=" * 80)

        for i, item in enumerate(
            result["results"],
            start=1,
        ):

            print(f"\n===== RESULT {i} =====")

            print(
                f"Source: {item['metadata']['source']}"
            )

            print(
                f"Page: {item['metadata']['page']}"
            )

            print(
                f"Section: {item['metadata']['section']}"
            )

            print(
                f"Distance: {item['distance']:.4f}"
            )

            print()

            print(item["text"])