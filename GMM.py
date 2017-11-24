import numpy as np
from gaussian import Util
from kmeans import KMeans


class GMM:

    def __init__(self, number_of_gaussian, max_interation=10):
        self.Ng = number_of_gaussian
        self.max_interation = max_interation
        self.utils = Util(number_of_gaussian)

    def model(self, data):
        kmeans = KMeans(self.Ng)
        lamb = kmeans.exec(data)
        return lamb
        #return self.__maximum_likelihood(data, lamb)

    def __maximum_likelihood(self, X, lamb):
        for i in range(0, self.max_interation):
            p_old = self.utils.get_prob_log(X, lamb)
            lamb_new = self.__expectation_maximisation(X, lamb)
            p_new = self.utils.get_prob_log(X, lamb_new)
            if p_new > p_old:
                lamb = lamb_new
        return lamb

    def calc_lg(self, data, lamb, g):
        for i in range(0, len(data)):
            aux1 = self.utils.weighted_gauss(data[i], lamb[0][g], lamb[1][g], lamb[2][g])
            aux2 = self.utils.sum_weighted_gauss(data[i], lamb)
            yield aux1 / aux2

    def __expectation_maximisation(self, X, lam):

        def ug_concat(x, lg_aux): return [x[i] * lg_aux[i] for i in range(0, len(x))]

        def safe_div(x, y):
            if y == 0:
                return 0
            return x / y

        lgi = [list(self.calc_lg(X, lam, g)) for g in range(0, self.Ng)]
        lg = [sum(lgi[g]) for g in range(0, self.Ng)]

        weight = [lg[g] / len(X) for g in range(0, self.Ng)]
        mu = [(safe_div(1, lg[g]) * sum(ug_concat(X, lgi[g]))) for g in range(0, self.Ng)]
        sigmas = []
        for g in range(0, self.Ng):
            acc = []
            for i in range(0, len(X)):
                s = np.outer(X[i], X[i]) * lgi[g][i]
                acc.append(s)
            r = safe_div(1, lg[g]) * sum(acc) - np.outer(mu[g], mu[g])
            sigmas.append(r)


        #sigma = [np.cov((X - mu[g]).T) for g in range(0, self.Ng)]

        return weight, mu, sigmas
