"""
Brique 1 — La base vectorielle persistante.
"""

import uuid

import chromadb
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME, CHROMA_DB_PATH, COLLECTION_NAME


class VectorDB:
    def __init__(
        self,
        db_path: str = CHROMA_DB_PATH,
        collection_name: str = COLLECTION_NAME,
        chunks: list[str] | None = None,
        sources: list[str] | None = None,
    ):
        self.db_path = db_path
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(path=self.db_path)

        collection_exists = collection_name in [
            c.name for c in self.client.list_collections()
        ]

        if collection_exists:
            self._reload_collection()
        elif chunks:
            self._create_collection(chunks, sources)
        else:
            raise ValueError(
                f"Aucune collection '{collection_name}' trouvée dans "
                f"'{db_path}', et aucun chunk fourni pour en créer une."
            )

    def _create_collection(self, chunks: list[str], sources: list[str] | None):
        if sources is not None and len(sources) != len(chunks):
            raise ValueError("sources doit avoir la même longueur que chunks.")

        print(
            f"[VectorDB] Création de la collection '{self.collection_name}' "
            f"({len(chunks)} chunks)..."
        )

        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "embedding_model": EMBEDDING_MODEL_NAME,
                "hnsw:space": "cosine",
            },
        )

        embeddings = self._encode(chunks)

        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [
            {"source": sources[i] if sources else f"chunk_{i}"}
            for i in range(len(chunks))
        ]

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

        print(
            f"[VectorDB] {len(chunks)} chunks indexés et persistés dans "
            f"'{self.db_path}'."
        )

    def _reload_collection(self):
        self.collection = self.client.get_collection(name=self.collection_name)

        stored_model_name = self.collection.metadata.get("embedding_model")
        if not stored_model_name:
            raise ValueError(
                "Collection existante mais sans métadonnée 'embedding_model'."
            )

        print(
            f"[VectorDB] Rechargement de la collection existante "
            f"'{self.collection_name}' "
            f"(modèle : {stored_model_name})."
        )

        self.embedding_model = SentenceTransformer(stored_model_name)

    def _encode(self, texts: list[str]):
        return self.embedding_model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def retrieve(self, question: str, n: int = 3) -> list[dict]:
        query_embedding = self._encode([question])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n,
        )

        chunks = []

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = 1 / (1 + dist)

            chunks.append(
                {
                    "text": doc,
                    "source": meta.get("source"),
                    "distance": dist,
                    "similarity": similarity,
                }
            )

        return chunks