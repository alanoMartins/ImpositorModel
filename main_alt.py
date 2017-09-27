
# Observações
#EMM deve ser interativo, comparando com P(x|lamba), com termino numa quantidade de loops e não modificados



import numpy as np
import functools
import math
from sklearn.datasets import load_iris
import random
import decimal
from numpy import linalg as LA



Ng = 3
#Nv = len(data[0])

def kmeans_gauss(X):
    return ([2,4,1], [5,1,2], [2,1 ,2])

def gauss(x, u, e):
    D = len(x)
    det = np.linalg.det(e)
    p1 = 1 / (2 * math.pi)**(D/2) * (np.sqrt(det))

    auxE = np.asmatrix(e)
    invE = np.linalg.inv(e)
    x = x.reshape(4, 1)
    aux1 = np.dot(x.T, invE)
    inv_aux = np.dot(aux1, x)
    p2 = (-1/2) * inv_aux.reshape(1,1)
    return p1 * math.exp(p2)

def weighted_gauss(x, m, u, e):
    return m * gauss(x, u, e)

def sum_weighted_gauss(x, Lam):
    acc = []
    for g in range(0, Ng):
        m_arr = Lam[0]
        u_arr = Lam[1]
        e_arr = Lam[2]
        acc.append(weighted_gauss(x, m_arr[g], u_arr[g], e_arr[g]))
    return sum(acc)

def log_sum_p(X, lam):
    acc = []
    for i in range(0, len(X)):
        s = sum_weighted_gauss(X[i], lam)
        slog = math.log(s)
        acc.append(math.log(sum_weighted_gauss(X[i], lam)))
    return sum(acc)


def GMM(X):
    l = kmeans_manual(X)
    lam = max_EMM(X, l)
    return (1 / len(X) * math.log(log_sum_p(X, lam)))

def get_prob(X, lamb):
    acc = []
    for i in range(0, len(X)):
        acc.append(sum_weighted_gauss(X[i], lamb))
    return sum(acc)

def max_EMM(X, lamb):
    max_interation = 10

    for i in range(0, max_interation):
        lamb_new = EMM(X, lamb)
        if get_prob(X, lamb_new) > get_prob(X, lamb):
            lamb = lamb_new


def EMM(X, lam):
    new_lam_m = []
    new_lam_u = []
    new_lam_e = []
    Lgi = []
    for g in range(0, Ng):
        m_arr = lam[0]
        u_arr = lam[1]
        e_arr = lam[2]

        lgg = []
        for i in range(0, len(X)):
            aux1 = weighted_gauss(X[i], m_arr[g], u_arr[g], e_arr[g])
            aux2 = sum_weighted_gauss(X[i], lam)
            lgg.append(aux1 / aux2)
        Lgi.append(lgg)

    for g in range(0, Ng):
        Lg = sum(Lgi[g])
        mg = Lg / len(X)

        ug_arr = []
        for i in range(0, len(X)):
            ug_arr.append(X[i] * Lgi[g][i])

        ug = (1/Lg * sum(ug_arr))

        sigG_arr = []
        for i in range(0, len(X)):
            outer_x = np.outer(X[i], X[i])
            sigG_arr.append(outer_x * Lgi[g][i])
        outer_u = np.outer(ug, ug)
        sigG =  (1 / Lg) * np.dot(sum(sigG_arr), outer_u)
        sigG = np.asmatrix(sigG)

        new_lam_m.append(mg)
        new_lam_u.append(ug)
        new_lam_e.append(sigG)

    return (new_lam_m, new_lam_u, new_lam_e)

# def kmeans(X):
#     kmeans = KMeans(n_clusters=Ng, random_state=0).fit(X)
#     print(kmeans.cluster_centers_)



def kmeans_manual(X):
    Nv = len(X)
    Ug = []
    ng = []
    Yi = []
    for g in range(0, Ng):
        ix = random.randint(0, Nv)
        Ug.append(X[ix])
    loop = 1
    endloop = 10
    finished = False
    while not finished:
        Yi.clear()
        ng.clear()
        for i in range(0, Nv):
            ming = Ng + 1
            minValue = 100
            for g in range(0, Ng):
                vectAux = Ug[g] - X[i]
                aux = LA.norm(vectAux)
                if (minValue > aux):
                    minValue = aux
                    ming = g
            Yi.append(ming)
        ug = []
        ug.clear()
        for g in range(0, Ng):
            gf = []
            ugarr = []
            for i in range(0, Nv):
                gf.append(Yi[i] == g)
                ugarr.append(X[i] * (Yi[i] == g))
            ng.append(sum(gf))
            ug.append(1 / Ng * sum(ugarr))

        same = True
        for g in range(0, Ng):
            if np.array_equal(Ug[g], ug[g]):
                same = False

        loop = loop + 1

        if same or loop > endloop:
            finished = True

        for g in range(0, Ng):
            Ug[g] = ug[g]

    mg = []
    eg = []
    for g in range(0, Ng):
        mg.append(ng[g] / Nv)
        egtemp = []
        for i in range(0, Nv):
            aux = X[i] - Ug[g]
            outer_prod = np.outer(aux, aux)
            outer_prod = np.asmatrix(outer_prod)
            egtemp.append(outer_prod * Yi[i])
        eg.append((1 / Ng) * sum(egtemp))



    return mg, Ug, eg


def exec():
    data, target = load_iris(True)
    c0 = data[0:50]
    c1 = data[50:100]
    c2 = data[100:150]

    d1 = np.array(c0)
    print(c0)
    r = GMM(d1)


exec()
