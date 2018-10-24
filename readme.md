Ce projet traite d'opinion mining et a été développé en M1 au pendant l'UE
d'extraction des connaissances. Il est question de predire si un avis est
positif ou negatif en fonction de son contenu et d'autres avis dont le label est
connu. Le but de ce projet est de comparer les prétraitements et les
classifieurs; de manière à déterminer le bon tuning dans le cadre du machine
learning.

# Prétraitement des données
Le prétraitement des données se de manière différée à l'éxécution des
classifieurs. Il s'agit de paramétrer le main de pretraitement.py et de généré
les versions transformées des données, ici déjà présentes.

# Évaluation des classifieurs
Il s'agit d'exécuter le fichier testclassifier.py. 2/3 des données
sont utilisées pour l'entrainement, 1/3 pour deviner les données.

# Résultats
Il ressort de ce projet que pour nos données et quelque, les avis bruts
(rivalisant avec les données sans stopword) sont en moyenne meilleur que les
autres types de prétraitements (lemmatisation, lemmatisation avec selection
morho-syntaxique).


# Entrainement des classifieurs et test sur un ensemble de données non étiquetté
Il faut pour ce but exécuter test_unknown_data.py. Sur un ensemble de donné inconnu et fourni au travers du déroulement de ce projet
F1-score de 92% d a pu être achevé grâce à la déterminisaison du meilleur tuning : meilleur pre-traitement, classifieurs et arguments:
