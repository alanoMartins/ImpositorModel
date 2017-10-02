import numpy as np
import random
from numpy import linalg as LA


class KMeans:

    def __init__(self, number_of_gaussians, max_interation=10):
        self.Ng = number_of_gaussians
        self.max_interation = max_interation

    def __min_norm_gauss(self, mu, Xi):
        norms = [LA.norm(mu[g] - Xi) for g in range(0, self.Ng)]
        return norms.index(min(norms))

    def __min_index_gauss(self, mu, data):
        return [self.__min_norm_gauss(mu, data[i]) for i in range(0, len(data))]

    def __means(self, X):
        def __new_mu(data, delta_kronecker):
            gf = [delta_kronecker(i) for i in range(0, len(data))]
            if sum(gf) == 0:
                return 0, np.zeros(len(data[0]))
            mu_updated = (1 / sum(gf)) * sum([data[ix] * y for ix, y in enumerate(gf)])
            return sum(gf), mu_updated

        mu = [random.choice(X) for i in range(0, self.Ng)]

        loop = 1
        finished = False
        yi = []
        ng = []
        while not finished:
            yi = self.__min_index_gauss(mu, X)
            res = [__new_mu(X, lambda ix: yi[ix] == g) for g in range(0, self.Ng)]
            ng, mu_new = zip(*res)
            ng = list(ng)
            mu_new = list(mu_new)

            same = all([np.array_equal(mu[g], mu_new[g]) for g in range(0, self.Ng)])
            loop = loop + 1
            finished = same or loop > self.max_interation

            mu = mu_new
        return ng, mu, yi

    def exec(self, data):
        ng, mu, y = self.__means(data)
        mg = [ng[g] / len(data) for g in range(0, self.Ng)]
        sigmas = [np.cov((data - mu[g]).T) for g in range(0, self.Ng)]

        return mg, mu, sigmas
