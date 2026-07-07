"""
Constantes du projet.

Le principe : un seul endroit où vivent les noms de modèles.
"""

# Modèle d'embedding : multilingue, léger, tourne bien en local/CPU.
EMBEDDING_MODEL_NAME = "distiluse-base-multilingual-cased-v2"

# Modèle de génération du RAG.
LLM_MODEL_NAME = "openai/gpt-oss-120b"

# Modèle de modération / détection de prompt injection.
MODERATION_MODEL_NAME = "openai/gpt-oss-safeguard-20b"

# Emplacement de la base vectorielle persistée sur disque.
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "mon_premier_rag"

# Nombre de chunks récupérés à chaque question.
TOP_K = 3

# Seuil minimal de similarité pour considérer qu'un chunk est pertinent.
# Exemple :
# - 0.54 : question bien reliée au corpus
# - 0.38 : question hors corpus
MIN_SIMILARITY = 0.45

# Chemins des prompts système.
PROMPT_RAG_PATH = "prompts/system_rag.txt"
PROMPT_MODERATOR_PATH = "prompts/system_moderator.txt"

# Marqueur remplacé par les chunks retrouvés dans le prompt du RAG.
CHUNKS_PLACEHOLDER = "{{Chunks}}"