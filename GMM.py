import numpy as np
from gaussian import Util
from kmeans import KMeans

class GMM():

    def __init__(self, number_of_gaussian):
        self.Ng = number_of_gaussian
        self.utils = Util(number_of_gaussian)

    def malicious(self, X, lamb):
        kmeans = KMeans(self.Ng)
        max_interation = 10

        for i in range(0, max_interation):
            p_old = self.utils.get_prob(X, lamb)
            lamb_new = kmeans.exec(X)
            p_new = self.utils.get_prob(X, lamb_new)
            if p_new > p_old:
                lamb = lamb_new
        return lamb

    def model(self, X):
        kmeans = KMeans(self.Ng)
        lamb = kmeans.exec(X)
        return self.malicious(X, lamb)

    def max_EMM(self, X, lamb):
        max_interation = 10

        for i in range(0, max_interation):
            p_old = self.utils.get_prob(X, lamb)
            lamb_new = self.EM(X, lamb)
            p_new = self.utils.get_prob(X, lamb_new)
            if p_new > p_old:
                lamb = lamb_new
        return lamb

    def EM(self, X, lam):
        m_arr = lam[0]
        u_arr = lam[1]
        e_arr = lam[2]

        Lgi = []
        for g in range(0, self.Ng):
            lgg = []
            for i in range(0, len(X)):
                aux1 = self.utils.weighted_gauss(X[i], m_arr[g], u_arr[g], e_arr[g])
                det_aux = []
                for gix in range(0, self.Ng):
                    m_arr_aux = lam[0]
                    u_arr_aux = lam[1]
                    e_arr_aux = lam[2]
                    aux = self.utils.weighted_gauss(X[i], m_arr_aux[gix], u_arr_aux[gix], e_arr_aux[gix])
                    det_aux.append(aux)
                aux2 = sum(det_aux)
                lgg.append(aux1 / aux2)
            Lgi.append(lgg)

        Lg = []
        lg = []
        for g in range(0, self.Ng):
            for i in range(0, len(X)):
                lg.append(Lgi[g][i])
            Lg.append(sum(lg))

        Mg = [Lg[g] / len(X) for g in range(0, self.Ng)]

        Ug = []
        for g in range(0, self.Ng):
            ug_arr = [X[i] * Lgi[g][i] for i in range(0, len(X))]
            ug = (1 / Lg[g] * sum(ug_arr))
            Ug.append(ug)

        SigG = []
        for g in range(0, self.Ng):
            sigG_arr = []
            for i in range(0, len(X)):
                arr = X[i] - Ug[g]
                outer_x = np.outer(arr, arr)
                sigG_arr.append(outer_x * Lgi[g][i])
            sigG = (1 / Lg[g]) * sum(sigG_arr)
            sigG = np.asmatrix(sigG)
            SigG.append(sigG)

        # SigG = []
        # for g in range(0, Ng):
        #     sigG_arr = []
        #     for i in range(0, len(X)):
        #         outer_x = np.outer(X[i], X[i])
        #         sigG_arr.append(outer_x * Lgi[g][i])
        #     acc = sum(sigG_arr)
        #     outer_u = np.outer(Ug, Ug)
        #     sub = np.subtract(acc, outer_u)
        #     res = (1 / Lg[g]) * sub
        #     sigG = np.asmatrix(res)
        #     SigG.append(sigG)

        return Mg, Ug, SigG
