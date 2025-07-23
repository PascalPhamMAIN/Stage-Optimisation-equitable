# Stage-Optimisation-equitable

## Librairies utilisées et leurs versions :
- Gurobi : 12.0.1
- Numpy : 2.2.4

## Problème d'affectation :

## Problème d'affectation couple :
- affectation_couple.ipynb : fichier principal contenant l'ensemble des algorithmes appliqués au problème d'affectation couple
- affectation_couple_v2.ipynb ou affectation_couple_v2.py
- tests_affectation_couple.ipynb permettant de simuler T tests sur l'ensemble des algorithmes : la troisième cellule utilise affectation_couple_v2.py et non affectation_couple_v2.ipynb

## Problème Vertex-Cover OWA :

## Fichiers contenant les résultats des différents tests :
- test_exp_data_n_k.ods : contient le nombre de contraintes, imposant une borne inférieure à une composante L_k, pour chaque itération, dans le modèle exponentiel
- affectation_couple_data : fichier généré regroupant les différentes données mesurées de chaque algorithme pour le problème d'affectation couple
- donnees_arrondi_iter.ods : répertorie l'ensemble des résultats des tests observés pendant ce stage, les feuilles 1 et 2 contiennent des essais spécifiques ou des tests exploratoires, la feuille 3 contient tous les tests importants sur les 3 problèmes étudiés
