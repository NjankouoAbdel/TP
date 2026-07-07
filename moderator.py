"""
Brique 2 — L'agent modérateur.

Pourquoi confier ça à un modèle dédié plutôt que d'écrire
"refuse les injections" dans le prompt du RAG lui-même ?
  - Séparation des responsabilités : le prompt du RAG reste concentré
    sur sa tâche (répondre depuis la base), il ne grossit pas avec des
    règles de sécurité qui n'ont rien à voir avec le retrieval.
    Une mégaprompt qui fait tout, tout le temps, est plus fragile.
  - Un modèle "safeguard" est entraîné spécifiquement pour cette
    classification et y est meilleur qu'une instruction ad-hoc noyée
    dans un prompt généraliste.
  - Sécurité en profondeur : si l'injection contourne quand même le
    prompt du RAG (les modèles généralistes ne sont pas infaillibles
    là-dessus), on a une deuxième ligne de défense indépendante, testée
    et modifiable séparément.
  - On peut appeler le modérateur AVANT de dépenser un appel au LLM
    principal : coupe court, plus rapide, moins cher en cas d'attaque.
"""

import json

from groq import Groq

from config import MODERATION_MODEL_NAME, PROMPT_MODERATOR_PATH


class Moderator:
    def __init__(self, client: Groq):
        self.client = client
        with open(PROMPT_MODERATOR_PATH, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

    def moderate(self, question: str) -> dict:
        response = self.client.chat.completions.create(
            model=MODERATION_MODEL_NAME,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
            ],
        )
        raw = response.choices[0].message.content

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Filet de sécurité : si le modèle ne renvoie pas un JSON
            # valide, on considère la question comme suspecte plutôt
            # que de planter ou de la laisser passer par défaut.
            return {"is_prompt_injection": True, "_parse_error": raw}
