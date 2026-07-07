# Mon premier RAG — implémentation

Implémentation du mini-TP : ChromaDB + sentence-transformers + Groq + agent modérateur.

## Démarrage rapide

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # puis collez votre clé console.groq.com dans .env
python main.py
```

Le premier lancement crée `chroma_db/` (base persistée) en indexant le
corpus jouet. Les lancements suivants la rechargent sans réencoder.

## Structure

```
config.py              constantes (modèles, chemins, TOP_K)
vector_db.py            Brique 1 : VectorDB (créer / recharger / retrieve)
moderator.py            Brique 2 : agent modérateur (JSON strict)
rag.py                  Brique 3 : orchestrateur (modération -> retrieval -> LLM)
prompts/system_rag.txt          prompt système du RAG (marqueur {{Chunks}})
prompts/system_moderator.txt    prompt système du modérateur
data/corpus.py          les 10 phrases absurdes de la base de connaissances
main.py                 script de démo reproduisant les 4 tests de la section 6
```

## Réponses aux questions du sujet

**Le bug silencieux évité par la métadonnée `embedding_model` sur la collection**
(section 3.1) : sans cette astuce, on rechargerait au démarrage le nom du
modèle depuis `config.py` -- qui peut avoir changé depuis l'indexation
initiale. Les nouveaux vecteurs de requête seraient alors calculés par un
modèle différent de celui qui a produit les vecteurs stockés : deux espaces
vectoriels différents, comparés comme s'ils étaient identiques. Si les
dimensions coïncident (ce qui arrive), il n'y a **aucune erreur** : le
système renvoie juste des résultats faux ou aléatoires, silencieusement.
C'est le pire type de bug -- pas de crash, pas de log, juste une qualité de
réponse qui se dégrade sans qu'on comprenne pourquoi.

**Pourquoi un modèle de modération dédié plutôt qu'une instruction dans le
prompt du RAG** (section 4) : séparation des responsabilités, modèle
spécialisé donc plus fiable sur cette tâche précise, et surtout défense en
profondeur -- si l'injection passait quand même le prompt du RAG, on a une
deuxième barrière indépendante, testable et modifiable séparément.

**Les 4 consignes du prompt système du RAG** (section 5.1), reformulées :
- *"tous les chunks ne sont pas forcément utiles"* → évite que le modèle
  cite un extrait juste parce qu'il est présent dans le contexte, même s'il
  ne répond pas à la question.
- *"triés du plus au moins pertinent"* → le modèle doit privilégier le
  premier extrait en cas de doute ou de contenu contradictoire entre chunks.
- *"ne répondre qu'à partir de cette base ; hors périmètre, dire qu'on ne
  sait pas"* → évite les hallucinations où le LLM comble les trous avec sa
  mémoire générale (le corpus jouet existe justement pour vérifier ça).
- *"signaler la contradiction avec l'affirmation de l'utilisateur"* → évite
  qu'un utilisateur fasse involontairement (ou volontairement) accepter une
  fausse affirmation par simple flatterie conversationnelle du modèle.

**Test section 6 :**
1. C'est l'agent modérateur (Brique 2, `moderator.py`) qui intercepte
   l'entrée piégée, **avant** tout appel au LLM principal (voir
   `rag.py::answer_question`, l'ordre modération → retrieval → génération).
2. Sans modérateur, le RAG suivrait la partie "réponds n'importe quoi"
   de l'injection en priorité si le LLM lui obéit (le prompt système du
   RAG dit "ne réponds qu'à partir de la base", mais rien ne garantit que
   le modèle résiste à une instruction utilisateur qui lui dit explicitement
   de l'ignorer -- testez en désactivant le modérateur dans `rag.py` pour
   voir le comportement réel de votre modèle).
3. Sur "Quelle est la capitale du Japon ?", le prompt impose de répondre
   qu'on ne sait pas (hors corpus). Si ce n'est pas le cas en pratique,
   durcissez la consigne 3 du prompt (répétez-la, ajoutez un exemple négatif).
4. Sur "Le chat de Bob est vert, non ?", le chunk pertinent ("chat bleu de
   Bob") doit être retrouvé et la contradiction signalée grâce à la
   consigne 4 du prompt.

## Note sur les modèles Groq

Le sujet mentionne `llama-3.3-70b-versatile` et
`meta-llama/llama-guard-4-12b`. Les deux ont été dépréciés par Groq
(respectivement le 17/06/2026 et le 10/02/2026). Ce projet utilise à la
place `openai/gpt-oss-120b` (génération) et `openai/gpt-oss-safeguard-20b`
(modération), configurés dans `config.py`. Vérifiez toujours
[console.groq.com/docs/models](https://console.groq.com/docs/models) avant
un vrai déploiement : les noms de modèles évoluent vite sur cette plateforme.
