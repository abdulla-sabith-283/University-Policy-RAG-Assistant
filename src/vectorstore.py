import chromadb

from .chunking import create_chunks
from .embeddings import EmbeddingModel


COLLECTION_NAME = "university_policies"


def create_vector_store():

    chunks = create_chunks()

    embedding_model = EmbeddingModel()

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.embed_documents(chunks)

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    metadatas = [chunk["metadata"] for chunk in chunks]

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    print(f"Stored {len(chunks)} chunks in Chroma.")

    return collection


if __name__ == "__main__":
    create_vector_store()