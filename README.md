# Stage - Optimisation équitable

Les codes ont été réalisés en Python à l'aide de Jupyter Notebook.

## Librairies utilisées et leurs versions

- `gurobipy` (v12.0.1) : Solveur d’optimisation linéaire
- `numpy` (v2.2.4)
- `pandas` (v2.3.0)
- `itertools` (standard Python) : Utilisé pour générer les contraintes liées aux composantes du vecteur **L** dans le modèle exponentiel
- `random` (standard Python)
- `time` (standard Python) : Mesure du temps d’exécution des algorithmes

---

## Problèmes traités

### Problème d’affectation

- `prob_lineaire.ipynb` : définition des composantes **L_k** et application du modèle d'Ogryczak au problème d'affectation
- `arrondi_iter_rdm.ipynb` : arrondi itératif sur le modèle d'Ogryczak avec une particularité : pour chaque itération, on ne cherche pas la plus grande composante de la solution optimale qui n'est pas fixée à 1. Ici, on fixe aléatoirement une de ses composantes à 1.
- `arrondi_iteratif.ipynb` : fichier principal regroupant tous les algorithmes pour le problème d’affectation
- `tests_3_algos.ipynb` : Lance **T tests** sur les algos contenus dans `arrondi_iteratif.ipynb`

### Problème d’affectation 1-1

- `affectation_couple.ipynb` : Fichier principal regroupant tous les algorithmes pour le problème d’affectation 1-1
- `affectation_couple_v2.ipynb` / `affectation_couple_v2.py`
- `tests_affectation_couple.ipynb` : Permet de lancer **T tests** sur les algorithmes :
  - Les deux premières cellules utilisent `affectation_couple.ipynb`
  - La troisième cellule fait appel à `affectation_couple_v2.py`

### Problème Vertex-Cover OWA

- `vertexcover.ipynb` : Introduction au problème Vertex-Cover avec des graphes simples
- `vertex_cover_equitable.ipynb` : Fichier principal contenant les algorithmes du Vertex-Cover OWA
- `tests_algo_vc.ipynb` : Lance **T tests** sur les algos contenus dans `vertex_cover_equitable.ipynb`

---

## Fichiers de résultats

- `test_exp_data_n_k.ods` : Indique, pour chaque itération du modèle exponentiel, le nombre de contraintes ajoutées sur les composantes **L_k**
- `affectation_couple_data` : Données mesurées pour chaque algorithme du problème d’affectation 1-1
- `donnees_arrondi_iter.ods` :
  - **Feuille 1 et 2** : Essais spécifiques ou explorations
  - **Feuille 3** : Feuille principale contenant l’ensemble des tests significatifs sur les **trois problèmes étudiés**
