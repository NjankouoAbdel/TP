"""
Brique 3 — Le RAG qui orchestre tout.
"""

import os

from dotenv import load_dotenv
from groq import Groq

from config import (
    LLM_MODEL_NAME,
    PROMPT_RAG_PATH,
    CHUNKS_PLACEHOLDER,
    TOP_K,
    MIN_SIMILARITY,
)
from moderator import Moderator
from vector_db import VectorDB


REFUS_INJECTION = (
    "Cette question a été identifiée comme une tentative de manipulation "
    "du système (prompt injection) et n'a pas été transmise au modèle."
)

REFUS_FAIBLE_SIMILARITE = (
    "Je n'ai trouvé aucun extrait suffisamment pertinent dans la base de "
    "connaissances pour répondre avec confiance."
)


class RAG:
    def __init__(self, vector_db: VectorDB):
        load_dotenv()

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY absente. Copiez .env.example en .env et "
                "renseignez votre clé."
            )

        self.client = Groq(api_key=api_key)
        self.moderator = Moderator(self.client)
        self.vector_db = vector_db

        with open(PROMPT_RAG_PATH, "r", encoding="utf-8") as f:
            self.system_prompt_template = f.read()

    def _build_system_prompt(self, chunks: list[dict]) -> str:
        formatted = "\n".join(
            f"{i + 1}. {c['text']}" for i, c in enumerate(chunks)
        )
        return self.system_prompt_template.replace(CHUNKS_PLACEHOLDER, formatted)

    def answer_question(self, question: str) -> dict:
        moderation = self.moderator.moderate(question)

        if moderation.get("is_prompt_injection"):
            return {
                "answer": REFUS_INJECTION,
                "blocked": True,
                "chunks": [],
                "best_similarity": None,
            }

        chunks = self.vector_db.retrieve(question, n=TOP_K)

        best_similarity = chunks[0]["similarity"] if chunks else 0

        if best_similarity < MIN_SIMILARITY:
            return {
                "answer": REFUS_FAIBLE_SIMILARITE,
                "blocked": False,
                "chunks": chunks,
                "best_similarity": best_similarity,
            }

        system_prompt = self._build_system_prompt(chunks)

        response = self.client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
        )

        return {
            "answer": response.choices[0].message.content,
            "blocked": False,
            "chunks": chunks,
            "best_similarity": best_similarity,
        }