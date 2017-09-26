import numpy as np
import functools
import  math
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import random
import decimal


# data = [[2, 51, 24, 25, 2, 41], #Cada array, uma classe
#         [24, 45, 42, 62, 86, 22],
#         [91, 15, 19, 79, 46, 18, 71]]

Ng = 4
#Nv = len(data[0])

def kmeans_gauss(X):
    return ([2,4,1], [5,1,2], [2,1 ,2])

def gauss(x, u, e):
    D = 2
    p1 = 1 / (2 * math.pi)**(D/2) * (abs(e) ** 1/2)
    p2 = (-1/2)*(((x-u) / e) ** 2)
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
    lam = EMM(X, l)
    return (1 / len(X) * math.log(log_sum_p(X, lam)))

def EMM(X, lam):
    new_lam_m = []
    new_lam_u = []
    new_lam_e = []
    for g in range(0, Ng):
        m_arr = lam[0]
        u_arr = lam[1]
        e_arr = lam[2]

        Lgi = []
        for i in range(0, len(X)):
            aux1 = weighted_gauss(X[i], m_arr[g], u_arr[g], e_arr[g])
            aux2 = sum_weighted_gauss(X[i], lam)
            Lgi.append(aux1 / aux2)

    for g in range(0, Ng):
        Lg = sum(Lgi)
        mg = Lg / len(X)

        ug_arr = []
        for i in range(0, len(X)):
            ug_arr.append(X[i] * Lgi[i])

        ug = (1/Lg * sum(ug_arr))

        sigG_arr = []
        for i in range(0, len(X)):
            sigG_arr.append((X[i] - ug) * Lgi[i])
        sigG = sum(sigG_arr)

        new_lam_m.append(mg)
        new_lam_u.append(ug)
        new_lam_e.append(sigG)

    return (new_lam_m, new_lam_u, new_lam_e)

def kmeans(X):
    kmeans = KMeans(n_clusters=Ng, random_state=0).fit(X)
    print(kmeans.cluster_centers_)



def kmeans_manual(X):
    Nv = len(X)
    Ug = []
    ng = []
    Yi = []
    for g in range(0, Ng):
        ix = random.randint(0, Ng)
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
                aux = abs(Ug[g] - X[i])
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
            if Ug[g] != ug[g]:
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
            egtemp.append(Yi[i] * (X[i] - ug[g] * (X[i] - ug[g])))
        eg.append((1 / Ng) * sum(egtemp))

    return (Ug, mg, eg)














def exec():
    data, target = load_iris(True)
    c0 = data[0:50]
    c1 = data[50:100]
    c2 = data[100:150]

    c0at0 = c0[:, [0]]
    c0at1 = c0[:, [1]]
    c0at2 = c0[:, [2]]
    c0at3 = c0[:, [3]]

    c0_arr = [c0at0, c0at1, c0at2 ,c0at3]

    c1at0 = c1[:, [0]]
    c1at1 = c1[:, [1]]
    c1at2 = c1[:, [2]]
    c1at3 = c1[:, [3]]

    c1_arr = [c1at0, c1at1, c1at2, c1at3]

    c2at0 = c2[:, [0]]
    c2at1 = c2[:, [1]]
    c2at2 = c2[:, [2]]
    c2at3 = c2[:, [3]]

    c2_arr = [c2at0, c2at1, c2at2, c2at3]

    model0_arr = [GMM(np.array(c_aux)) for c_aux in c1_arr]
    model0 = sum(model0_arr)

    model1_arr = [GMM(np.array(c_aux)) for c_aux in c1_arr]
    model1 = sum(model1_arr)

    model2_arr = [GMM(np.array(c_aux)) for c_aux in c1_arr]
    model2 = sum(model2_arr)

    print(model0)
    print(model1)
    print(model2)

    d1 = np.array(c0at0).flatten()
    print(d1)
    r = GMM(d1)
    print(r)

exec()
