#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import random
import time
from gurobipy import *

# Paramètres du modèle
n = 5   # Nombre de lignes
m = 5 # Nombre de colonnes (avec n = m)
k = 5  # k <= n = m # NE PAS OUBLIER DE REFAIRE : jupyter nbconvert --to script affectation_couple_v2.ipynb (ou bien changer directement depuis le fichier py)
nb_ressources = k

# Cas n = m (voir bloc-notes reunion 23_05)
lower_agent = [1] * n # li
upper_agent = [1] * n # ui
lower_item = [1] * m  # lj'
upper_item = [1] * m  # uj'

def constr_model_dynamic(c, w_prime, valid_pairs):
    model = Model("affectation couple")
    model.Params.OutputFlag = 0

    # Variables
    x_affect = model.addVars(valid_pairs, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x_affect")
    x_obj = model.addVars(valid_pairs, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x_obj")
    z = model.addVars(n, vtype=GRB.CONTINUOUS, name="z")
    # l[0] = l1, l[1] = l2, etc
    l = model.addVars(n, vtype=GRB.CONTINUOUS, name="l")

    # Contraintes agents
    for i in range(n):
        involved_items = [j for j in range(m) if (i, j) in valid_pairs]
        if involved_items:
            model.addConstr(quicksum(x_affect[i, j] for j in involved_items) >= lower_agent[i])
            model.addConstr(quicksum(x_affect[i, j] for j in involved_items) <= upper_agent[i])

    # Contraintes items
    for j in range(m):
        valid_agents = [i for i in range(n) if (i, j) in valid_pairs]
        if valid_agents:
            model.addConstr(quicksum(x_affect[i, j] for i in valid_agents) >= lower_item[j])
            model.addConstr(quicksum(x_affect[i, j] for i in valid_agents) <= upper_item[j])

    sync_constr = {}
    #for i in range(n):
    #    for j in range(m):
    for (i,j) in valid_pairs:
        sync_constr[i, j] = model.addConstr(x_affect[i, j] == x_obj[i, j], name=f"sync_{i}_{j}")

    for i in range(n):
        model.addConstr(z[i] == quicksum(c[i][j] * x_obj[i, j] for j in range(m) if (i,j) in valid_pairs), name=f"z_{i}")

    model.setObjective(quicksum(w_prime[k] * l[k] for k in range(n)), GRB.MINIMIZE)
    model.update()
    return model, x_affect, x_obj, z, l, sync_constr

def iterative_rounding_dynamic_sorted1(
    model, x_affect, x_obj, z, l, sync_constr,
    valid_pairs, tol=1e-6
):
    iteration = 0
    it_frac   = 0
    min_max_val = 1
    l_sorted_constrs = []
    c_saturees_test  = []

    # Initialisation des compteurs d’agents / items
    # (si vous les utilisez plus bas)
    n = len(z)               # nombre d’agents
    m = max(j for (_,j) in x_affect.keys()) + 1
    somme_1 = [0]*n
    somme_2 = [0]*m

    while True:
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            print("Résolution non optimale.")
            break

        # Calcul des contraintes saturées
        c_saturees = 0

        # Extraction et tri des z[i]
        z_vals   = [(i, z[i].X) for i in range(n)]
        sorted_z = sorted(z_vals, key=lambda t: t[1], reverse=True)

        # Ajout des contraintes l[k] >= sum des k+1 plus grands z
        prefix = 0
        violation_found = False
        for k, (i_val, v) in enumerate(sorted_z):
            prefix += v
            if l[k].X < prefix - tol:
                # expr = somme des premières k+1 variables z
                expr = quicksum(z[j] for j,_ in sorted_z[:k+1])
                cons = model.addConstr(l[k] >= expr,
                                       name=f"l_sorted_{k}")
                l_sorted_constrs.append(cons)
                violation_found = True
        if violation_found:
            model.update()
            continue

        # Comptage des slacks <= tol
        for cons in l_sorted_constrs:
            if cons.getAttr(GRB.Attr.Slack) <= tol:
                c_saturees += 1

        c_saturees_test.append(c_saturees)
        print(f"Iter {iteration} -> c_saturees = {c_saturees}")

        # Sélection d’une variable fractionnaire la plus grande
        max_val = -1
        sel_i, sel_j = -1, -1
        # On parcourt x_affect.keys(), pas valid_pairs
        for (i,j) in x_affect.keys():
            if x_affect[i,j].LB != x_affect[i,j].UB:
                v = x_affect[i,j].X
                if v > max_val:
                    max_val = v
                    sel_i, sel_j = i, j

        if sel_i < 0:
            break

        if 0 < max_val < min_max_val:
            min_max_val = max_val
        if 0 < max_val < 1:
            it_frac += 1

        # On retire la sync et on fixe la variable
        model.remove(sync_constr[sel_i, sel_j])
        x_affect[sel_i, sel_j].LB = x_affect[sel_i, sel_j].UB = 1.0
        x_obj[sel_i, sel_j].LB    = x_obj[sel_i, sel_j].UB    = 0.5
        model.update()

        # Mise à jour des compteurs par agent et par item
        somme_1[sel_i] += 1
        if somme_1[sel_i] >= upper_agent[sel_i]:
            for (i2,j2) in x_affect.keys():
                if i2 == sel_i and x_affect[i2,j2].LB == 0:
                    x_affect[i2,j2].UB = 0
                    x_obj[i2,j2].UB    = 0

        somme_2[sel_j] += 1
        if somme_2[sel_j] >= upper_item[sel_j]:
            for (i2,j2) in x_affect.keys():
                if j2 == sel_j and x_affect[i2,j2].LB == 0:
                    x_affect[i2,j2].UB = 0
                    x_obj[i2,j2].UB    = 0

        iteration += 1

    print("Total constraints added:", len(l_sorted_constrs))
    return min_max_val, x_affect, iteration, it_frac, c_saturees_test

def init_prob():
    c = np.random.randint(1, 1000, size=(n, m))

    def fct_w(n):
        return np.array([n - k for k in range(n)])

    def fct_w_prime(w):
        n = len(w)
        w_prime = np.zeros(n)
        for k in range(n - 1):
            w_prime[k] = w[k] - w[k + 1]
        w_prime[n - 1] = w[n - 1]
        return w_prime

    w = fct_w(n)
    w_prime = fct_w_prime(w)

    #max_attempts = 10
    #solution_found = False
    it = 0

    while True: # en théorie tant que k est entre 1 et n, ca se finit en une itération
        # Pré-affectation aléatoire de certains arcs
        #arcs_fixes = []
        #col_count = {j: 0 for j in range(m)}

        #for i in range(n):
        #    candidate_cols = [j for j in range(m) if col_count[j] < k] # ca empeche de prendre bcp trop de fois le meme et donc ca permet de respecter les contraintes
        #    if candidate_cols:
        #        j_choisi = random.choice(candidate_cols)
        #        #print(j_choisi)
        #        arcs_fixes.append((i, j_choisi))
        #        col_count[j_choisi] += 1

        #if all(col_count[j] <= k for j in range(m)):
        modele = Model(f"Modele_Randomized_{it}")
        x = modele.addVars(n, m, lb=0, ub = 1, vtype=GRB.CONTINUOUS, name="x")

        # Définition de la fonction objectif : maximiser la somme de x[i,j]
        w1 = np.random.randint(1, 10, size=(n, m))
        modele.setObjective(quicksum(w1[i,j]*x[i, j] for i in range(n) for j in range(m)), GRB.MAXIMIZE)

        for i in range(n):
            modele.addConstr(quicksum(x[i, j] for j in range(m)) == k, name=f"Contrainte_ligne_{i}")

        for j in range(m):
            modele.addConstr(quicksum(x[i, j] for i in range(n)) == k, name=f"Contrainte_colonne_{j}")

        #for (i, j) in arcs_fixes:
        #    modele.addConstr(x[i, j] == 1, name=f"Fixe_{i}_{j}")
        modele.Params.OutputFlag = 0

        modele.optimize()
        valid_pairs = []

        if modele.Status == GRB.OPTIMAL:
            print(f"Solution trouvée à la tentative {it+1}")
            print("Valeur optimale de l'objectif :", modele.objVal) # donne aussi le nombre de pairs valides
            for i in range(n):
                for j in range(m):
                    val_x = round(x[i, j].X, 2)
                    #print(f"x[{i},{j}] = {val_x}")
                    if (val_x == 1):
                        valid_pairs.append((i,j))
            #solution_found = True
            break
        else:
            print(f"Pas de solution optimale à la tentative {it+1}, on recommence.")

        it += 1

    #print("Les pairs valides :", valid_pairs)
    #print("Nombre de pairs valides :", len(valid_pairs))
    return c, w_prime, valid_pairs

def run_one_test():
    c, w_prime, valid_pairs = init_prob()
    model, x_affect, x_obj, z, l, sync_constr = constr_model_dynamic(c, w_prime, valid_pairs)
    return iterative_rounding_dynamic_sorted1(model, x_affect, x_obj, z, l, sync_constr, valid_pairs)[-1]  # ne retourne que c_saturees_test

