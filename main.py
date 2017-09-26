import numpy as np
import functools
import  math
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import random

#data = [[[2,51,24,25],[12,52,63],[15,346,45],[],63,52], [1,12,51,2,62,23], [12,3,41,7,23,2]]
data = [[2, 51, 24, 25, 2, 41],
        [24, 45, 42, 62, 86, 22],
        [91, 15, 19, 79, 46, 18, 71]]

Ng = 3
Nv = len(data[0])


def baysian_rule(likelihood):
    P = 1 / 3
    return likelihood * P


def join_likelihood(data_li):
    data_prod = [d * 1/3 for d in data_li]
    return functools.reduce(lambda x, y: x * y, data_prod)


def gauss(x, u, e):
    D = 10
    p1 = 1 / (2 * math.pi)**(D/2) * (math.modf(e) ** 1/2)
    p2 = (-1/2)*(((x-u) / e) ** 2)
    return p1 * math.exp(p2)


def PdexDadoLambda(x, Lams):
    for ix in range(0, Ng):
        yield GMMPartial(x, Lams[ix])


def GMMPartial(x, lam):
    m = lam[0]
    u = lam[1]
    e = lam[2]
    return m * gauss(x, u, e)

# X é a lista dos arrays de cada gaussiana
def EEM(X):
    for d in data:
        yield kmeans(d)
    # kmeans = KMeans(n_clusters=Ng, random_state=0)
    # kmeans.fit(X)
    # y_pred = kmeans.predict([41,21,5, 1, 24,2])
    # print(y_pred)
    # means = kmeans.cluster_centers_
    # print(means)

def kmeans(X):
    Ug = []
    ng = []
    Yi = []
    for g in range(0, Ng):
      Ug.append(random.randint(1, Nv))
    loop = 1
    endloop = 10
    finished = False
    while not finished:
        Yi.clear()
        ng.clear()
        for i in range(0, Nv):
            ming = []
            for g in range(0, Ng):
                ming.append(abs(Ug[g] - X[i]))
            Yi.append(min(ming))
        ug = []
        for g in range(0, Ng):
            gf = []
            ugarr = []
            for i in range(0, Nv):
                gf.append(Yi[i] == g)
                ugarr.append(X[i] * (Yi[i] == g))
            ng.append(functools.reduce(lambda x, y: x + y, gf))
            ug.append(1/Ng * functools.reduce(lambda x, y: x + y, ugarr))

        same = True
        for g in range(0, Ng):
            if Ug[g] != ug[g]:
                same = False

        loop = loop + 1

        if same or loop > endloop:
            finished = True

        for g in range(0, Ng):
            Ug = ug

    mg = []
    eg = []
    for g in range(0, Ng):
        mg.append(ng[g] / Nv)
        egtemp = []
        for i in range(0, Nv):
            egtemp.append(Yi[i] * (X[i] - ug[g] * (X[i] - ug[g])))
        eg.append( (1/Ng) * functools.reduce(lambda x,y: x + y, egtemp))


    return (Ug, mg, eg)

def EEM(X):
    for d in data:
        mean = kmeans(d)
        print(mean)


def exec():
    EEM(data)


exec()