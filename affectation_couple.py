#!/usr/bin/env python
# coding: utf-8

# \begin{align}
#     \text{Max} \quad & \sum_{i = 1}^{n} \ \sum_{j = 1}^{m} \ w_{ij} \ x_{ij} \\
#     \text{s.t.} \quad & 
#         \left\{
#             \begin{array}{ll}
#             n = m \\
#             \sum_{i = 1}^{n} x_{ij} = k \quad \forall j \\
#             \sum_{j = 1}^{m} x_{ij} = k \quad \forall i \\
#             x_{ij} \ge 0
#             \end{array}
#         \right.
# \end{align}

# In[72]:


import numpy as np
import random
import time
from gurobipy import *

# Paramètres du modèle
n = 10   # Nombre de lignes
m = 10  # Nombre de colonnes (avec n = m)
k = 5  # k <= n = m
nb_ressources = k

# Cas n = m (voir bloc-notes reunion 23_05)
lower_agent = [1] * n # li
upper_agent = [1] * n # ui
lower_item = [1] * m  # lj'
upper_item = [1] * m  # uj'

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


# In[73]:


def sol_equitable_opti_constrained(c, w_prime, n, m, l, u, l_prime, u_prime, valid_pairs):
    model = Model("min W constrained")
    model.Params.OutputFlag = 0

    # Création des variables SEULEMENT pour les paires valides
    x = model.addVars(valid_pairs, vtype=GRB.BINARY, name="x") # ca évite les cas des pairs invalides, et note : c'est une liste de tuples
    # donc, les x[i,j] qui ne sont pas dans valid_pairs n'existent pas et x[i,j] <=> x[(i,j)]
    b = model.addVars(n, n, vtype=GRB.CONTINUOUS, name="b", lb=0)
    r = model.addVars(n, vtype=GRB.CONTINUOUS, name="r")

    # Contraintes de couverture des agents
    for i in range(n):
        involved_items = [j for j in range(m) if (i, j) in valid_pairs]
        if involved_items:
            model.addConstr(quicksum(x[i, j] for j in involved_items) >= l[i], f"c1a_{i}")
            model.addConstr(quicksum(x[i, j] for j in involved_items) <= u[i], f"c1b_{i}")

    # Contraintes de couverture des items
    for j in range(m):
        involved_agents = [i for i in range(n) if (i, j) in valid_pairs]
        if involved_agents: # non vide
            model.addConstr(quicksum(x[i, j] for i in involved_agents) >= l_prime[j], f"c2a_{j}")
            model.addConstr(quicksum(x[i, j] for i in involved_agents) <= u_prime[j], f"c2b_{j}")

    # Contraintes de r[k] + b[i,k] >= coût total pour chaque i et k
    for i in range(n):
        for k in range(n):
            cost_sum = quicksum(c[i, j] * x[i, j] for j in range(m) if (i, j) in valid_pairs)
            model.addConstr(r[k] + b[i, k] >= cost_sum, f"c3_{i}_{k}")

    # Fonction objectif
    obj_fn = quicksum(w_prime[k] * ((k + 1) * r[k] + quicksum(b[i, k] for i in range(n))) for k in range(n))
    model.setObjective(obj_fn, GRB.MINIMIZE)

    return model, x

# Résolution
#####start = time.time()
#####model, x_vars = sol_equitable_opti_constrained(c, w_prime, n, m, lower_agent, upper_agent, lower_item, upper_item, valid_pairs)
#####model.optimize()
#####end = time.time()

#####time_exact_o = end - start
#####print("time_exact(O) :", time_exact_o)
#####val_exact_o = model.objVal
#####print("val_exact(O) :", val_exact_o)

#print("\nSolution finale :")
#cout_tot = 0
#for (i,j) in valid_pairs:
#    print("x[{}, {}] = {:.3f}".format(i, j, x_vars[i, j].X))
#    cout_tot += c[i,j]*x_vars[i, j].X
#print("Cout total :", cout_tot)


# In[74]:


######start = time.time()
model = Model("assignment")
model.Params.OutputFlag = 0

# Variables x uniquement pour les paires valides
x = model.addVars(valid_pairs, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
b = model.addVars(n, n, vtype=GRB.CONTINUOUS, lb=0, name="b")
r = model.addVars(n, vtype=GRB.CONTINUOUS, name="r")

# Contraintes agents
for i in range(n):
    involved_items = [j for j in range(m) if (i, j) in valid_pairs]
    if involved_items:
        model.addConstr(quicksum(x[i, j] for j in involved_items) >= lower_agent[i])
        model.addConstr(quicksum(x[i, j] for j in involved_items) <= upper_agent[i])

# Contraintes items
for j in range(m):
    valid_agents = [i for i in range(n) if (i, j) in valid_pairs]
    if valid_agents:
        model.addConstr(quicksum(x[i, j] for i in valid_agents) >= lower_item[j])
        model.addConstr(quicksum(x[i, j] for i in valid_agents) <= upper_item[j])

# Contrainte r[k] + b[i,k] >= coût
for i in range(n):
    for k in range(n):
        cost_expr = quicksum(c[i, j] * x[i, j] for j in range(m) if (i, j) in valid_pairs)
        model.addConstr(r[k] + b[i, k] >= cost_expr)

# Objectif
obj_expr = quicksum(w_prime[k]*((k+1)*r[k] + quicksum(b[i,k] for i in range(n))) for k in range(n))
model.setObjective(obj_expr, GRB.MINIMIZE)

model.update()

it_tot_o = 0
it_frac_o = 0
#it_demi_o = 0
max_iterations = len(valid_pairs)
somme_1 = [0 for i in range(n)]
somme_2 = [0 for j in range(m)]
#print("Valid pairs:", valid_pairs)

#print("Type of x:", type(x))
#print("Keys in x:", list(x.keys()))

#for (i, j) in valid_pairs:
#    if (i, j) in x:
#        print(f"x[{i},{j}] : LB={x[i, j].getAttr('LB')}, UB={x[i, j].getAttr('UB')}")

#for (i, j) in valid_pairs:
#    if (i, j) in x:
#        print(f"x[{i},{j}] : LB={x[i, j].LB}, UB={x[i, j].UB}")
min_max_val_o = 1

while it_tot_o <= max_iterations:
    #print("somme_2 :", somme_2)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        for (i,j) in valid_pairs:
            print("TEST :", i, j)
            if x[i, j].LB != x[i, j].UB:
                print("i,j =", i, j)
        print("Modèle non optimal ou infaisable")
        break

    # Recherche du max x[i,j] encore non fixé
    max_val = -1
    sel_i, sel_j = -1, -1
    for (i, j) in valid_pairs:
        if x[i, j].LB != x[i, j].UB:
            val = x[i, j].X
            if val > max_val:
                max_val = val
                sel_i, sel_j = i, j

    if max_val != -1 and max_val < min_max_val_o:
        min_max_val_o = max_val
    #print(max_val, sel_i, sel_j)

    if sel_i == -1 or sel_j == -1:
        break  # plus rien à fixer

    if max_val > 0 and max_val < 1:
        it_frac_o += 1

    #if max_val > 0 and max_val < 0.5:
    #    it_demi_o += 1

    x[sel_i, sel_j].lb = 1
    x[sel_i, sel_j].ub = 1
    model.update()

    somme_1[sel_i] += 1
    if somme_1[sel_i] >= upper_agent[sel_i]:
        for j in range(m):
            if (sel_i, j) in valid_pairs and x[sel_i, j].LB == 0:
                x[sel_i, j].ub = 0

    somme_2[sel_j] += 1
    if somme_2[sel_j] >= upper_item[sel_j]:
        for i in range(n):
            if (i, sel_j) in valid_pairs and x[i, sel_j].LB == 0:
                x[i, sel_j].ub = 0

    it_tot_o += 1

######end = time.time()

######time_approx_o = end - start
######print("time_approx(O) :", time_approx_o)
######val_approx_o = model.objVal
######print("val_approx(O):", val_approx_o)
######print("min_max_val(O):", min_max_val_o)

#print("Nombre total d'itérations(O) :", it_tot_o)
######print("it_frac(O) =", it_frac_o)
#print("it_demi(O) =", it_demi_o)

#cout_tot = 0
#for (i,j) in valid_pairs:
#    #print("x[{}, {}] = {:.3f}".format(i, j, x_vars[i, j].X))
#    cout_tot += c[i,j]*x[i, j].X
#print("Cout total :", cout_tot)


# In[75]:


def chassein_algo(c,w,n,m,l,u,l_prime,u_prime): # n = m
    opt_mod2 = Model(name = "C-MIP")
    opt_mod2.Params.OutputFlag = 0

    y = opt_mod2.addVars(n, vtype = GRB.CONTINUOUS, name = "y")
    #b = opt_mod2.addVars(n, n, vtype = GRB.CONTINUOUS, name = "b", lb = 0)
    #r = opt_mod2.addVars(n, name = 'r', vtype = GRB.CONTINUOUS)
    alpha = opt_mod2.addVars(n, name = 'alpha', vtype = GRB.CONTINUOUS)
    beta = opt_mod2.addVars(n, name = 'beta', vtype = GRB.CONTINUOUS)
    x = opt_mod2.addVars(valid_pairs, vtype = GRB.BINARY, name = "x")

    # Contraintes agents
    for i in range(n):
        involved_items = [j for j in range(m) if (i, j) in valid_pairs]
        if involved_items:
            opt_mod2.addConstr(quicksum(x[i, j] for j in involved_items) >= l[i])
            opt_mod2.addConstr(quicksum(x[i, j] for j in involved_items) <= u[i])

    # Contraintes items
    for j in range(m):
        valid_agents = [i for i in range(n) if (i, j) in valid_pairs]
        if valid_agents:
            opt_mod2.addConstr(quicksum(x[i, j] for i in valid_agents) >= l_prime[j])
            opt_mod2.addConstr(quicksum(x[i, j] for i in valid_agents) <= u_prime[j])

    #opt_mod2.addConstrs((l_prime[j] <= sum(x[i,j] for i in range(n)) for j in range(m)), name = 'c1a')
    #opt_mod2.addConstrs((sum(x[i,j] for i in range(n)) <= u_prime[j] for j in range(m)), name = 'c1b')
    #opt_mod2.addConstrs((l[i] <= sum(x[i,j] for j in range(m)) for i in range(n)), name = 'c2a')
    #opt_mod2.addConstrs((sum(x[i,j] for j in range(m)) <= u[i] for i in range(n)), name = 'c2b

    opt_mod2.addConstrs((y[i] == sum(c[i,j]*x[i,j] for j in range(m) if (i, j) in valid_pairs) for i in range(n)), name = 'c2')
    opt_mod2.addConstrs((alpha[i] + beta[j] >= w[j]*y[i] for i in range(n) for j in range(n) if (i, j) in valid_pairs), name = 'c3')
    #opt_mod2.addConstrs((r[k] + b[i,k] >= sum(c[i,j]*x[i,j] for j in range(m)) for i in range(n) for k in range(n)), name = 'c3')

    #w_prime = fct_w_prime(w)

    #obj_fn2 = quicksum(w_prime[k]*((k+1)*r[k] + quicksum(b[i,k] for i in range(n))) for k in range(n))
    obj_fn2 = quicksum(alpha[i] + beta[i] for i in range(n))
    opt_mod2.setObjective(obj_fn2, GRB.MINIMIZE)

    return opt_mod2, x

#####start = time.time()
#####opt_mod2, x1 = chassein_algo(c,w,n,m,lower_agent,upper_agent,lower_item,upper_item)
#####opt_mod2.optimize()
#####end = time.time()
#####time_exact_c = end-start
##print()
##print("RUNTIME (en s) :", opt_mod2.RUNTIME)
#####print("time_exact(C) :", time_exact_c)
#opt_mod2.write("sol_exacte_poly.lp")

#####val_exact_c = opt_mod2.objVal
#####print('val_exact(C) : %f' % val_exact_c)
#print("Nombre d'itérations :", opt_mod2.IterCount)
#print("Temps de résolution (s) :", opt_mod2.Runtime)

#cout_tot = 0
#for (i,j) in valid_pairs:
#    print("x[{}, {}] = {:.3f}".format(i, j, x1[i, j].X))
#    cout_tot += c[i,j]*x[i, j].X
#print("Cout total :", cout_tot)


# In[76]:


def chassein_iterative_rounding(c, w, n, m, l, u, l_prime, u_prime):
    model = Model(name = "C-MIP")
    model.Params.OutputFlag = 0

    y = model.addVars(n, vtype = GRB.CONTINUOUS, name = "y")
    #b = model.addVars(n, n, vtype = GRB.CONTINUOUS, name = "b", lb = 0)
    #r = model.addVars(n, name = 'r', vtype = GRB.CONTINUOUS)
    alpha = model.addVars(n, name = 'alpha', vtype = GRB.CONTINUOUS)
    beta = model.addVars(n, name = 'beta', vtype = GRB.CONTINUOUS)
    x = model.addVars(valid_pairs, vtype = GRB.CONTINUOUS, lb = 0, ub = 1, name = "x")

    # Contraintes agents
    for i in range(n):
        involved_items = [j for j in range(m) if (i, j) in valid_pairs]
        if involved_items:
            model.addConstr(quicksum(x[i, j] for j in involved_items) >= l[i])
            model.addConstr(quicksum(x[i, j] for j in involved_items) <= u[i])

    # Contraintes items
    for j in range(m):
        valid_agents = [i for i in range(n) if (i, j) in valid_pairs]
        if valid_agents:
            model.addConstr(quicksum(x[i, j] for i in valid_agents) >= l_prime[j])
            model.addConstr(quicksum(x[i, j] for i in valid_agents) <= u_prime[j])

    model.addConstrs((y[i] == sum(c[i,j]*x[i,j] for j in range(m) if (i, j) in valid_pairs) for i in range(n)), name = 'c2')
    model.addConstrs((alpha[i] + beta[j] >= w[j]*y[i] for i in range(n) for j in range(n) if (i, j) in valid_pairs), name = 'c3')
    #model.addConstrs((r[k] + b[i,k] >= sum(c[i,j]*x[i,j] for j in range(m)) for i in range(n) for k in range(n)), name = 'c3')

    #w_prime = fct_w_prime(w)

    #obj_fn2 = quicksum(w_prime[k]*((k+1)*r[k] + quicksum(b[i,k] for i in range(n))) for k in range(n))
    obj_fn2 = quicksum(alpha[i] + beta[i] for i in range(n))
    model.setObjective(obj_fn2, GRB.MINIMIZE)

    # ----- ARRONDI ITÉRATIF -----
    iteration = 0
    it_frac = 0
    #it_demi = 0
    max_iterations = len(valid_pairs)
    somme_1 = [0 for i in range(n)]
    somme_2 = [0 for j in range(m)]
    min_max_val_c = 1
    #print("Valid pairs:", valid_pairs)

    #print("Type of x:", type(x))
    #print("Keys in x:", list(x.keys()))

    #for (i, j) in valid_pairs:
    #    if (i, j) in x:
    #        print(f"x[{i},{j}] : LB={x[i, j].getAttr('LB')}, UB={x[i, j].getAttr('UB')}")

    #for (i, j) in valid_pairs:
    #    if (i, j) in x:
    #        print(f"x[{i},{j}] : LB={x[i, j].LB}, UB={x[i, j].UB}")

    while iteration <= max_iterations:
        #print("Iteration :", iteration)
        #print("somme_2 :", somme_2)
        model.optimize()

        if model.status != GRB.OPTIMAL:
            for (i,j) in valid_pairs:
                print("TEST :", i, j)
                if x[i, j].LB != x[i, j].UB:
                    print("i,j =", i, j)
            print("Modèle non optimal ou infaisable")
            break

        # Recherche du max x[i,j] encore non fixé
        max_val = -1
        sel_i, sel_j = -1, -1
        for (i, j) in valid_pairs:
            if x[i, j].LB != x[i, j].UB:
                val = x[i, j].X
                if val > max_val:
                    max_val = val
                    sel_i, sel_j = i, j
        #print(max_val, sel_i, sel_j)
        if max_val != -1 and max_val < min_max_val_c:
            min_max_val_c = max_val

        if sel_i == -1 or sel_j == -1:
            break  # plus rien à fixer

        if max_val > 0 and max_val < 1:
            it_frac += 1

        #if max_val > 0 and max_val < 0.5:
        #    it_demi += 1

        x[sel_i, sel_j].lb = 1
        x[sel_i, sel_j].ub = 1
        model.update()

        somme_1[sel_i] += 1
        if somme_1[sel_i] >= upper_agent[sel_i]:
            for j in range(m):
                if (sel_i, j) in valid_pairs and x[sel_i, j].LB == 0:
                    x[sel_i, j].ub = 0

        somme_2[sel_j] += 1
        if somme_2[sel_j] >= upper_item[sel_j]:
            for i in range(n):
                if (i, sel_j) in valid_pairs and x[i, sel_j].LB == 0:
                    x[i, sel_j].ub = 0

        iteration += 1

        # Construction de la solution finale binaire
        #solution = {(i, j): int(round(x[i, j].X)) for i in range(n) for j in range(m)}

    return model,x,min_max_val_c,iteration,it_frac #,it_demi

start = time.time()
opt_mod2, x1, min_max_val_c, it_tot_c, it_frac_c = chassein_iterative_rounding(c,w,n,m,lower_agent,upper_agent,lower_item,upper_item)
opt_mod2.optimize()
end = time.time()
time_approx_c = end-start
##print()
##print("RUNTIME (en s) :", opt_mod2.RUNTIME)
#####print("time_approx(C) :", time_approx_c)
#opt_mod2.write("sol_exacte_poly.lp")
#####mod_chassein_it = opt_mod2.objVal
#####print('val_approx(C) : %f' % mod_chassein_it)
#####print("min_max_val(C) :", min_max_val_c)
#print("Nombre total d'itérations(C) :", it_tot_c)
#####print("it_frac(C) :", it_frac_c)
#print("it_demi(C) :", it_demi_c)

#cout_tot = 0
#for (i,j) in valid_pairs:
#    #print("x[{}, {}] = {:.3f}".format(i, j, x_vars[i, j].X))
#    cout_tot += c[i,j]*x[i, j].X
#print("Cout total :", cout_tot)


# In[77]:


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

def iterative_rounding_dynamic_sorted(model, x_affect, x_obj, z, l, sync_constr, tol=1e-6):
    iteration = 0
    it_frac = 0
    #it_demi = 0
    #total_gurobi_iters = 0
    min_max_val = 1
    #somme_1 = [0 for i in range(n)]
    #somme_2 = [0 for j in range(m)]
    l_sorted_constrs = []
    c_saturees_test = []

    while True:
        model.optimize()
        #total_gurobi_iters += model.IterCount

        if model.Status != GRB.OPTIMAL:
            for (i,j) in valid_pairs:
                print("TEST :", i, j)
                if x_affect[i, j].LB != x_affect[i, j].UB:
                    print("i,j =", i, j)
            print("Résolution non optimale.")
            #print(iteration)
            break

        c_saturees = 0
        z_vals = [(i, z[i].X) for i in range(n)]
        sorted_z = sorted(z_vals, key=lambda tup: tup[1], reverse=True) # on trie en fonction du 2eme element du couple

        prefix_sum = 0
        violation_found = False
        # Pour k de 0 à n-1 correspond à la vérification sur les k+1 plus grandes valeurs
        for k, (i_val, val) in enumerate(sorted_z):
            prefix_sum += val
            if l[k].X < prefix_sum - tol:
                expr = quicksum(z[j] for j, _ in sorted_z[:k+1]) # NOTE : on est obligé d'utiliser z et pas z_sorted, car ce sont des variables
                # Or, à la prochaine itération, on aura un nouveau z et donc un autre sorted_z, d'ou l'intéret de garder z, sinon, ca ne marchera pas !
                cons = model.addConstr(l[k] >= expr, name=f"l_sorted_{k}")
                l_sorted_constrs.append(cons)
                violation_found = True
            #else:
            #    expr = quicksum(z[j] for j, _ in sorted_z[:k+1])
            #    if (l[k].X == expr):
            #        c_saturees += 1
        if violation_found:
            model.update()
            continue  # on relance l'optimisation pour tenir compte des nouvelles contraintes (donc on saute le reste)

        for cons in l_sorted_constrs:
            slack = cons.getAttr(GRB.Attr.Slack)
            if slack <= tol:
                c_saturees += 1
        c_saturees_test.append(c_saturees)

        #print("Itération :", iteration)
        #print("Iteration :", iteration, ", nombre de contraintes saturées :", c_saturees)
        print(c_saturees)
        #print("Nombre de contraintes ajoutées dans le modèle :", len(l_sorted_constrs))

        max_val = -1
        sel_i, sel_j = -1, -1

        #for i in range(n):
        #    for j in range(m):
        for (i,j) in valid_pairs:
            if x_affect[i, j].LB != x_affect[i, j].UB:  # variable non fixée
                val = x_affect[i, j].X
                if val > max_val:
                    max_val = val
                    sel_i, sel_j = i, j

        #print(max_val, sel_i, sel_j)

        if max_val > 0 and max_val < min_max_val:
            min_max_val = max_val

        if sel_i == -1 or sel_j == -1: #or max_val <= 1e-5: # tolerance
            #print("Terminé. Nombre total d'itérations Gurobi:", total_gurobi_iters)
            #print("Nombre tot d'iterations :", iteration)
            break

        if max_val > 0 and max_val < 1:
            it_frac += 1

        #if max_val > 0 and max_val < 0.5:
        #    it_demi += 1

        # On enlève la contrainte de synchronisation pour la variable fixée
        model.remove(sync_constr[sel_i, sel_j])

        x_affect[sel_i, sel_j].LB = 1.0
        x_affect[sel_i, sel_j].UB = 1.0

        x_obj[sel_i, sel_j].LB = 0.5
        x_obj[sel_i, sel_j].UB = 0.5

        model.update()

        somme_1[sel_i] += 1
        if somme_1[sel_i] >= upper_agent[sel_i]:
            for j in range(m):
                if (sel_i, j) in valid_pairs and x_affect[sel_i, j].LB == 0:
                    x_affect[sel_i, j].ub = 0
                    x_obj[sel_i, j].ub = 0

        somme_2[sel_j] += 1
        if somme_2[sel_j] >= upper_item[sel_j]:
            for i in range(n):
                if (i, sel_j) in valid_pairs and x_affect[i, sel_j].LB == 0:
                    x_affect[i, sel_j].ub = 0
                    x_obj[i, sel_j].ub = 0

        iteration += 1
    print("Nombre de contraintes ajoutées dans le modèle :", len(l_sorted_constrs))
    #print(l_sorted_constrs)
    return min_max_val, x_affect, iteration, it_frac, c_saturees_test #, it_demi

start = time.time()
model_dyn, x_affect, x_obj, z, l, sync_constr = constr_model_dynamic(c, w_prime, valid_pairs)
#model_dyn.write("modele_dynamique.lp")
min_max_val_3e_v2, x_affect, it_tot_3e, it_frac_3e, c_saturees_test = iterative_rounding_dynamic_sorted(model_dyn, x_affect, x_obj, z, l, sync_constr)
end = time.time()
val_new_mod_v2 = model_dyn.objVal
#model_dyn.write("modele_dynamique_end.lp")
time_approx_3e_v2 = end-start
print("time_approx_3e_v2 :", time_approx_3e_v2)
print("min_max_val_3e_v2 :", min_max_val_3e_v2)

# Affichage de la solution finale
print("val_new_mod_v2 :", val_new_mod_v2)
val_new_mod_v2_2 = 2*val_new_mod_v2
print('2*val_new_mod : %f' % val_new_mod_v2_2)

#print("Nombre total d'itérations 3e mod :", it_tot_3e)
print("it_frac(3e) =", it_frac_3e)
#print("it_demi(3e) =", it_demi_3e)

#cout_tot = 0
#for (i,j) in valid_pairs:
#    #print("x[{}, {}] = {:.3f}".format(i, j, x_vars[i, j].X))
#    cout_tot += c[i,j]*x_affect[i, j].X
#print("Cout total :", cout_tot)


# In[78]:


#####ratio_o = val_approx_o/val_exact_o
#####print("O-MIP, Ratio de :", ratio_o)
#####ratio_c = mod_chassein_it/val_exact_c
#####print("C-MIP, Ratio de :", ratio_c)
#####ratio_3e = val_new_mod_v2_2/val_exact_o # O-MIP donne la meme solution exacte que le 3eme modèle
#####print("3e mod, Ratio de :", ratio_3e)
print("------")


# In[80]:


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
    # Paramètres du modèle
    n = 10   # Nombre de lignes
    m = 10  # Nombre de colonnes (avec n = m)
    k = 5  # k <= n = m
    nb_ressources = k

    # Cas n = m (voir bloc-notes reunion 23_05)
    lower_agent = [1] * n # li
    upper_agent = [1] * n # ui
    lower_item = [1] * m  # lj'
    upper_item = [1] * m  # uj'

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

def run_one_test(c, w_prime, valid_pairs):
    c, w_prime, valid_pairs = init_prob()
    model, x_affect, x_obj, z, l, sync_constr = constr_model_dynamic(c, w_prime, valid_pairs)
    return iterative_rounding_dynamic_sorted1(model, x_affect, x_obj, z, l, sync_constr, valid_pairs)[-1]  # ne retourne que c_saturees_test


# In[ ]:




