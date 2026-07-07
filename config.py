"""
Constantes du projet.

Le principe : un seul endroit où vivent les noms de modèles.
Si Groq déprécie un modèle demain, on change une ligne ici, pas dix
fichiers dispersés.

NB (juillet 2026) : le document du TP mentionne llama-3.3-70b-versatile
et meta-llama/llama-guard-4-12b. Les deux ont été dépréciés par Groq
depuis la rédaction du sujet. On les remplace ici par leurs successeurs
officiels ; vérifiez toujours https://console.groq.com/docs/models
et https://console.groq.com/docs/deprecations avant de lancer un vrai
projet, les noms de modèles changent vite sur cette plateforme.
"""

# Modèle d'embedding : multilingue, léger, tourne bien en local/CPU.
EMBEDDING_MODEL_NAME = "distiluse-base-multilingual-cased-v2"

# Modèle de génération (le "cerveau" du RAG).
# Remplace llama-3.3-70b-versatile (déprécié le 17/06/2026).
LLM_MODEL_NAME = "openai/gpt-oss-120b"

# Modèle de modération / détection de prompt injection.
# Remplace meta-llama/llama-guard-4-12b (déprécié le 10/02/2026).
MODERATION_MODEL_NAME = "openai/gpt-oss-safeguard-20b"

# Emplacement de la base vectorielle persistée sur disque.
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "mon_premier_rag"

# Nombre de chunks récupérés à chaque question.
TOP_K = 3

# Chemins des prompts système (texte, hors code -> se retravaille sans toucher au code).
PROMPT_RAG_PATH = "prompts/system_rag.txt"
PROMPT_MODERATOR_PATH = "prompts/system_moderator.txt"

# Marqueur remplacé par les chunks retrouvés dans le prompt du RAG.
CHUNKS_PLACEHOLDER = "{{Chunks}}"
