
# Observações
#EMM deve ser interativo, comparando com P(x|lamba), com termino numa quantidade de loops e não modificados



import numpy as np
import functools
import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import random
import decimal
from numpy import linalg as LA



Ng = 3


def gauss(x, u, e):
    D = len(x)
    det = abs(np.linalg.det(e))
    det = round(det, 3)
    p1 = 1 / (2 * math.pi)**(D/2) * (np.sqrt(det))

    sub = x - u
    m = 10 ^ -6
    e_aux = e + np.eye(e.shape[1])*m
    invE = np.linalg.inv(e)
    sub = sub.reshape(4, 1)
    aux1 = np.dot(sub.T, invE)
    inv_aux = np.dot(aux1, sub)
    p2 = (-1/2) * inv_aux.reshape(1,1)
    p2 = p2.item(0)
    p2 = round(p2, 3)
    return p1 * math.e ** p2


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
    return max_EMM(X, l)


def get_prob(X, lamb):
    acc = []
    for i in range(0, len(X)):
        acc.append(sum_weighted_gauss(X[i], lamb))
    return sum(acc)


def max_EMM(X, lamb):
    max_interation = 10

    for i in range(0, max_interation):
        p_old = get_prob(X, lamb)
        lamb_new = EMM(X, lamb)
        p_new = get_prob(X, lamb_new)
        if p_new > p_old:
            lamb = lamb_new
    return lamb


def EMM(X, lam):
    m_arr = lam[0]
    u_arr = lam[1]
    e_arr = lam[2]

    Lgi = []
    for g in range(0, Ng):

        lgg = []
        for i in range(0, len(X)):
            aux1 = weighted_gauss(X[i], m_arr[g], u_arr[g], e_arr[g])
            aux2 = sum_weighted_gauss(X[i], lam)
            lgg.append(aux1 / aux2)
        Lgi.append(lgg)


    Lg = []
    lg = []
    for g in range(0, Ng):
        for i in range(0, len(X)):
            lg.append(Lgi[g][i])
        Lg.append(sum(lg))

    Mg = []
    for g in range(0, Ng):
        mg = Lg[g] / len(X)
        Mg.append(mg)

    Ug = []
    for g in range(0, Ng):
        ug_arr = []
        for i in range(0, len(X)):
            ug_arr.append(X[i] * Lgi[g][i])

        ug = (1/Lg[g] * np.sum(ug_arr))
        Ug.append(ug)

    SigG = []
    for g in range(0, Ng):
        sigG_arr = []
        for i in range(0, len(X)):
            arr = X[i] - Ug[g]
            outer_x = np.outer(arr, arr)
            sigG_arr.append(outer_x * Lgi[g][i])
        sigG = (1 / Lg[g]) * sum(sigG_arr)
        sigG = np.asmatrix(sigG)
        SigG.append(sigG)

    return Mg, Ug, SigG


def kmeans_manual(X):
    Nv = len(X)
    Ug = []
    ng = []
    Yi = []
    for g in range(0, Ng):
        ix = random.randint(0, Nv - 1)
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

def compare(X_test, y, models):
    accepts = []
    for ix in range(0, len(X_test)):
        X = X_test[ix]
        best_p = 0
        best_i = -1
        for m in range(0, len(models)):
            p = sum_weighted_gauss(X, models[m])
            if p > best_p:
                best_p = p
                best_i = m
        res = y[ix]
        accepts.append(best_i == res)

    print(sum(accepts) / len(X_test))


def cross_validation(dataset, target, test_size):
    c0 = dataset[0:50]
    c1 = dataset[50:100]
    c2 = dataset[100:150]

    t0 = target[0:50]
    t1 = target[50:100]
    t2 = target[100:150]

    n_interations = round(len(c0) * test_size)
    test_arr = []
    test_tar = []

    for i in range(0, n_interations):
        rand_i = random.randint(0, len(c0) - 1)
        test_arr.append(c0[rand_i])
        test_tar.append(t0[rand_i])
        c0 = np.delete(c0, rand_i, axis=0)
        t0 = np.delete(t0, rand_i, axis=0)

    for i in range(0, n_interations):
        rand_i = random.randint(0, len(c1) - 1)
        test_arr.append(c1[rand_i])
        test_tar.append(t1[rand_i])
        c1 = np.delete(c1, rand_i, axis=0)
        t1 = np.delete(t1, rand_i, axis=0)

    for i in range(0, n_interations):
        rand_i = random.randint(0, len(c2) - 1)
        test_arr.append(c2[rand_i])
        test_tar.append(t2[rand_i])
        c2 = np.delete(c2, rand_i, axis=0)
        t2 = np.delete(t2, rand_i, axis=0)

    return np.array([c0, c1, c2]), np.array([t0, t1, t2]), test_arr, test_tar

def exec():

    data, target = load_iris(True)

    X_train, y_train, X_test, y_test = cross_validation(data, target, 0.4)


    c0, c1, c2 = X_train

    orq1 = np.array(c0)
    orq2 = np.array(c1)
    orq3 = np.array(c2)

    modelOrq1 = GMM(orq1)
    modelOrq2 = GMM(orq2)
    modelOrq3 = GMM(orq3)

    models = [modelOrq1, modelOrq2, modelOrq3]
    compare(X_test, y_test, models)



exec()
