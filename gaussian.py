from scipy.stats import multivariate_normal
import numpy as np
import math


class Util:

    def __init__(self, number_of_gaussians):
        self.Ng = number_of_gaussians

    @staticmethod
    def gauss(x, u, e):
        det = np.linalg.det(e)
        det = round(det, 2)
        r = np.linalg.cond(e)
        print('Determinante %d' % det)
        print('Condition %d' % r)
        res = multivariate_normal(u, e)
        return res.pdf(x)

    def weighted_gauss(self, x, m, u, e):
        return m * self.gauss(x, u, e)

    def sum_weighted_gauss(self, x, lamb):
        acc = []
        for g in range(0, self.Ng):
            m_arr = lamb[0]
            u_arr = lamb[1]
            e_arr = lamb[2]
            acc.append(self.weighted_gauss(x, m_arr[g], u_arr[g], e_arr[g]))
        return sum(acc)

    def get_prob(self, data, lamb):
        acc = []
        for i in range(0, len(data)):
            acc.append(self.sum_weighted_gauss(data[i], lamb))
        return sum(acc)

    def get_prob_log(self, data, lamb):
        acc = []
        for i in range(0, len(data)):
            acc.append(math.log(self.sum_weighted_gauss(data[i], lamb)))
        return sum(acc)
