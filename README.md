# Stage - Optimisation équitable

## Librairies utilisées et leurs versions

- `gurobipy` (v12.0.1) : Solveur d’optimisation linéaire
- `numpy` (v2.2.4) : Manipulation efficace de tableaux et matrices
- `itertools` (standard Python) : Utilisé pour générer les contraintes liées aux composantes du vecteur **L** dans le modèle exponentiel
- `random` (standard Python) : Génération aléatoire pour simulations de tests
- `time` (standard Python) : Mesure du temps d’exécution des algorithmes

---

## Problèmes traités

### Problème d’affectation

### Problème d’affectation couple

- `affectation_couple.ipynb` : Fichier principal regroupant tous les algorithmes pour le problème d’affectation couple
- `affectation_couple_v2.ipynb` / `affectation_couple_v2.py` : Version alternative du modèle
- `tests_affectation_couple.ipynb` : Permet de lancer **T tests** sur les algorithmes :
  - Les deux premières cellules utilisent `affectation_couple.ipynb`
  - La troisième cellule fait appel à `affectation_couple_v2.py`

### Problème Vertex-Cover OWA

- `vertexcover.ipynb` : Introduction au problème Vertex-Cover avec des graphes simples
- `vertex_cover_equitable.ipynb` : Fichier principal contenant les algorithmes du Vertex-Cover OWA
- `tests_algo_vc.ipynb` : Lance **T tests** sur les algos contenus dans `vertex_cover_equitable.ipynb`

---

## 📊 Fichiers de résultats

- `test_exp_data_n_k.ods` : Indique, pour chaque itération du modèle exponentiel, le nombre de contraintes ajoutées sur les composantes **L_k**
- `affectation_couple_data` : Données mesurées pour chaque algorithme du problème d’affectation couple
- `donnees_arrondi_iter.ods` :
  - **Feuille 1 et 2** : Essais spécifiques ou explorations
  - **Feuille 3** : Feuille principale contenant l’ensemble des tests significatifs sur les **trois problèmes étudiés**
