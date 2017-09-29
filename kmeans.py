import numpy as np
import functools
import math
from sklearn.datasets import load_iris
import random
from numpy import linalg as LA

class KMeans():

    def __init__(self, number_of_gaussians):
        self.Ng = number_of_gaussians

    def min_norm_gauss(self, Ug, Xi):
        norms = [LA.norm(Ug[g] - Xi) for g in range(0, self.Ng)]
        return norms.index(min(norms))

    def min_index_gauss(self, Ug, X):
        return [self.min_norm_gauss(Ug, X[i]) for i in range(0, len(X))]

    def new_mu(self, X, delta_kronecker):
        gf = [delta_kronecker(i) for i in range(0, len(X))]
        ugarr = (1 / sum(gf)) * sum([X[ix] * y for ix, y in enumerate(gf)])
        return sum(gf), ugarr

    def new_ng(self, X, delta_kronecker):
        gf = [delta_kronecker(i) for i in range(0, len(X))]
        return sum(gf)

    def delta_kronecker(self, X, Y):
        return X == Y

    def means(self, X):
        Ug = [random.choice(X) for i in range(0, self.Ng)]

        loop = 1
        endloop = 10
        finished = False
        while not finished:
            Yi = self.min_index_gauss(Ug, X)
            res = [self.new_mu(X, lambda ix: Yi[ix] == g) for g in range(0, self.Ng)]
            ng, ug = zip(*res)
            ng = list(ng)
            ug = list(ug)

            same = all([np.array_equal(Ug[g], ug[g]) for g in range(0, self.Ng)])
            loop = loop + 1
            finished = same or loop > endloop

            Ug = ug
        return ng, Ug, Yi

    def exec(self, X):
        ng, Mu_arr, y = self.means(X)

        mg = [ng[g] / len(X) for g in range(0, self.Ng)]

        Sigmas = []
        for g in range(0, self.Ng):
            sigma_temp = []
            for i in range(0, len(X)):
                aux_re = (X[i] - Mu_arr[g]).reshape(4, 1)
                aux_outer = aux_re * aux_re.T
                sigma_temp.append(aux_outer * y[i])
            Sigmas.append((1 / self.Ng) * sum(sigma_temp))

        return mg, Mu_arr, Sigmas