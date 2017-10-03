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
        return self.__maximum_likelihood(data, lamb)

    def model_with_impostor(self, data, universal_lamb):
        kmeans = KMeans(self.Ng)
        lamb = kmeans.exec(data)
        return self.__maximum_a_posteriori(data, lamb, universal_lamb)

    def a_posteriori(self, data, lam_client, lam_universal):
        weight_client = lam_client[0]
        mu_client = lam_client[1]
        sigma_client = lam_client[2]

        weight_universal = lam_universal[0]
        mu_universal = lam_universal[1]
        sigma_universal = lam_universal[2]

        lgi = [list(self.calc_lg(data, lam_client, g)) for g in range(0, self.Ng)]
        lg = [sum(lgi[g]) for g in range(0, self.Ng)]

        scale = 8
        alpha = [lg[g] / (lg[g] + scale) for g in range(0, self.Ng)]

        weight_updater = lambda alpha, w_cli, w_uni: ((alpha*w_cli) + (1-alpha)*w_uni ) * scale
        mu_updater = lambda alpha, mu_cli, mu_uni: ((alpha*mu_cli) + ((1-alpha) * mu_uni))
        sigma_updater = lambda alpha, sig_cli, sig_uni, mu_cli, mu_uni, new_mu: \
            (alpha * (sig_cli + np.outer(mu_cli, mu_cli)) + (1-alpha) * (sig_uni + np.outer(mu_uni, mu_uni))) - np.outer(new_mu, new_mu)

        weight_new = [weight_updater(alpha[g], weight_client[g], weight_universal[g]) for g in range(0, self.Ng)]
        mu_new = [mu_updater(alpha[g], mu_client[g], mu_universal[g]) for g in range(0, self.Ng)]
        sigma_new = [sigma_updater(alpha[g], sigma_client[g], sigma_universal[g], mu_client[g], mu_universal[g], mu_new[g]) for g in range(0, self.Ng)]

        return weight_new, mu_new, sigma_new

    def __maximum_a_posteriori(self, X, lamb, lamb_universal):
        for i in range(0, self.max_interation):
            p_old = self.utils.get_prob_log(X, lamb)
            lamb_new = self.a_posteriori(X, lamb, lamb_universal)
            p_new = self.utils.get_prob_log(X, lamb_new)
            if p_new > p_old:
                lamb = lamb_new
        return lamb

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
        sigma = [np.cov((X - mu[g]).T) for g in range(0, self.Ng)]

        return weight, mu, sigma
