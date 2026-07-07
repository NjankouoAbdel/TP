"""
Brique 3 — Le RAG qui orchestre tout.

Pipeline de answer_question() :
  1. Modération de la question (agent modérateur, Brique 2)
  2. Si injection détectée -> refus immédiat, ON N'APPELLE JAMAIS le LLM
     principal avec cette question. L'ORDRE est une décision de
     sécurité : on ne veut surtout pas que le texte suspect atteigne
     le modèle qui a accès au contexte "sensible" (même minime ici).
  3. Sinon -> récupération des chunks (Brique 1)
  4. Construction du prompt système à trous (remplacement de {{Chunks}})
  5. Appel au LLM de génération avec messages system/user
"""

import os

from dotenv import load_dotenv
from groq import Groq

from config import (
    LLM_MODEL_NAME,
    PROMPT_RAG_PATH,
    CHUNKS_PLACEHOLDER,
    TOP_K,
)
from moderator import Moderator
from vector_db import VectorDB


REFUS_INJECTION = (
    "Cette question a été identifiée comme une tentative de manipulation "
    "du système (prompt injection) et n'a pas été transmise au modèle."
)


class RAG:
    def __init__(self, vector_db: VectorDB):
        load_dotenv()  # charge GROQ_API_KEY depuis .env

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY absente. Copiez .env.example en .env et "
                "renseignez votre clé (console.groq.com)."
            )

        self.client = Groq(api_key=api_key)
        self.moderator = Moderator(self.client)
        self.vector_db = vector_db

        with open(PROMPT_RAG_PATH, "r", encoding="utf-8") as f:
            self.system_prompt_template = f.read()

    def _build_system_prompt(self, chunks: list[dict]) -> str:
        # Les chunks sont déjà triés du plus au moins pertinent par
        # ChromaDB (résultats classés par distance croissante).
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
            }

        chunks = self.vector_db.retrieve(question, n=TOP_K)
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
        }
