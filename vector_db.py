"""
Brique 1 — La base vectorielle persistante.

Comportement voulu par le sujet (l'"aiguillage" dans le constructeur) :
  - si une base existe déjà sur disque à db_path -> on la recharge
  - sinon, si on nous donne des chunks -> on la crée
  - sinon -> erreur explicite, on ne devine rien
"""

import os
import uuid

import chromadb
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME, CHROMA_DB_PATH, COLLECTION_NAME


class VectorDB:
    def __init__(self, db_path: str = CHROMA_DB_PATH,
                 collection_name: str = COLLECTION_NAME,
                 chunks: list[str] | None = None,
                 sources: list[str] | None = None):
        self.db_path = db_path
        self.collection_name = collection_name

        # Client persistant : les données survivent à l'arrêt du programme.
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
                f"'{db_path}', et aucun chunk fourni pour en créer une. "
                "Impossible de démarrer : fournissez des chunks ou "
                "pointez vers une base existante."
            )

    # ------------------------------------------------------------------ #
    # Création
    # ------------------------------------------------------------------ #
    def _create_collection(self, chunks: list[str], sources: list[str] | None):
        print(f"[VectorDB] Création de la collection '{self.collection_name}' "
              f"({len(chunks)} chunks)...")

        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # Astuce du sujet : on grave le nom du modèle d'embedding utilisé
        # dans les métadonnées de la collection elle-même.
        #
        # Pourquoi c'est malin : si on ne le faisait pas, on relirait au
        # rechargement le nom du modèle depuis config.py -- qui peut avoir
        # changé entre-temps (quelqu'un met à jour EMBEDDING_MODEL_NAME
        # pour un projet suivant). On chargerait alors un modèle différent
        # de celui qui a servi à encoder les vecteurs stockés. Les nouveaux
        # vecteurs de requête ne vivraient plus dans le même espace
        # vectoriel que les vecteurs indexés : la similarité cosinus
        # perdrait tout son sens. Le pire, c'est que ça ne plante PAS
        # forcément (les dimensions peuvent coïncider par hasard) : le
        # système renvoie juste des résultats silencieusement mauvais,
        # sans aucune erreur -- un bug très difficile à diagnostiquer.
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"embedding_model": EMBEDDING_MODEL_NAME},
        )

        embeddings = self._encode(chunks)

        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [
            {"source": (sources[i] if sources else f"chunk_{i}")}
            for i in range(len(chunks))
        ]

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )
        print(f"[VectorDB] {len(chunks)} chunks indexés et persistés dans "
              f"'{self.db_path}'.")

    # ------------------------------------------------------------------ #
    # Rechargement
    # ------------------------------------------------------------------ #
    def _reload_collection(self):
        self.collection = self.client.get_collection(name=self.collection_name)

        # On relit le modèle qui a servi à créer CETTE collection, pas
        # celui écrit dans la config du jour -- voir l'explication ci-dessus.
        stored_model_name = self.collection.metadata.get("embedding_model")
        if not stored_model_name:
            raise ValueError(
                "Collection existante mais sans métadonnée 'embedding_model' : "
                "impossible de savoir quel modèle utiliser pour les requêtes."
            )

        print(f"[VectorDB] Rechargement de la collection existante "
              f"'{self.collection_name}' (modèle : {stored_model_name}).")
        self.embedding_model = SentenceTransformer(stored_model_name)

    # ------------------------------------------------------------------ #
    # Encodage (méthode partagée par l'indexation et la requête)
    # ------------------------------------------------------------------ #
    def _encode(self, texts: list[str]):
        # normalize_embeddings=True : on force des vecteurs de norme 1,
        # ce qui rend le produit scalaire équivalent à la similarité
        # cosinus -- exactement la métrique que Chroma utilise par défaut.
        return self.embedding_model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    # ------------------------------------------------------------------ #
    # Recherche
    # ------------------------------------------------------------------ #
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
            chunks.append({"text": doc, "source": meta.get("source"), "distance": dist})
        return chunks
