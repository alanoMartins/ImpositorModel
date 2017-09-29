from scipy.stats import multivariate_normal
import math

class Util():

    def __init__(self, number_of_gaussians):
        self.Ng = number_of_gaussians

    def gauss(self, x, u, e):
        res = multivariate_normal(u, e).pdf(x)
        return res

    def weighted_gauss(self, x, m, u, e):
        return m * self.gauss(x, u, e)

    def log_sum_p(self, X, lam):
        acc = []
        for i in range(0, len(X)):
            s = self.sum_weighted_gauss(X[i], lam)
            slog = math.log(s)
            acc.append(math.log(self.sum_weighted_gauss(X[i], lam)))
        return sum(acc)

    def sum_weighted_gauss(self, x, Lam):
        acc = []
        for g in range(0, self.Ng):
            m_arr = Lam[0]
            u_arr = Lam[1]
            e_arr = Lam[2]
            acc.append(self.weighted_gauss(x, m_arr[g], u_arr[g], e_arr[g]))
        return sum(acc)

    def get_prob(self, X, lamb):
        acc = []
        for i in range(0, len(X)):
            acc.append(self.sum_weighted_gauss(X[i], lamb))
        return sum(acc)