"""
Base de connaissances jouet.

L'idée du sujet : ces faits n'existent nulle part sur Internet.
Si le système répond juste, c'est forcément grâce au retrieval,
jamais grâce à la mémoire du LLM. C'est un excellent test pour
vérifier qu'un RAG fonctionne réellement.
"""

CORPUS = [
    "Le chat bleu de Bob s'appelle Henri.",
    "Henri, le chat de Bob, déteste profondément les mardis.",
    "La voiture de Sylvie est peinte en violet à pois orange.",
    "Le poisson rouge de Marc s'appelle Capitaine Nemo Junior.",
    "Dans le bureau de Claire, il y a une plante en plastique qui s'appelle Gérard.",
    "Le mot de passe secret du club de lecture de Fatima est 'ananas volant'.",
    "Le chien de Paul, un labrador, a peur des aspirateurs mais adore les tondeuses à gazon.",
    "La grand-mère de Julien collectionne les boîtes d'allumettes vides depuis 1987.",
    "Le café préféré de Nadia se prend avec trois glaçons et une pincée de cannelle.",
    "Le vélo de Karim a une sonnette qui joue l'hymne national du Luxembourg.",
]
