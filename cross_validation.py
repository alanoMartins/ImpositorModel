import numpy as np
import random
from sklearn.decomposition import PCA


class CrossValidation:

    def __init__(self, test_size):
        self.test_size = test_size

    def group_by_data(self, data, tar):

        def get_key(t):
            return t[1]

        res = list(zip(data, tar))
        res = sorted(res, key=get_key)
        data, tar = list(zip(*res))

        d0 = data[0:10]
        d1 = data[10:20]
        d2 = data[20:30]
        d3 = data[30:40]
        d4 = data[40:50]
        d5 = data[50:60]
        d6 = data[60:70]
        d7 = data[70:80]
        d8 = data[80:90]
        d9 = data[90:100]

        t0 = tar[0:10]
        t1 = tar[10:20]
        t2 = tar[20:30]
        t3 = tar[30:40]
        t4 = tar[40:50]
        t5 = tar[50:60]
        t6 = tar[60:70]
        t7 = tar[70:80]
        t8 = tar[80:90]
        t9 = tar[90:100]

        return np.array([d0, d1, d2, d3, d4, d5, d6, d7, d8, d9]), np.array([t0, t1, t2, t3, t4, t5, t6, t7, t8, t9])

    def decomposer(self, data):
        return PCA(n_components=100).fit_transform(data)

    def exec(self, data, tar):
        dataset, target = self.group_by_data(data, tar)

        filtered = []
        for data1 in dataset:
            aux = []
            for d in data1:
                d = d[d != 0]
                d = d[0:1000]
                aux.append(d)
            filtered.append(aux)
        dataset = np.array(filtered)

        dataset = [self.decomposer(d) for d in dataset]

        n_interaction = [round(len(d) * self.test_size) for d in dataset]

        data_tes = []
        target_tes = []
        data_train = []
        target_train = []
        for d in range(0, len(dataset)):
            for i in range(0, n_interaction[d]):
                rand_i = random.randint(0, len(dataset[d]) - 1)
                data_tes.append(dataset[d][rand_i])
                target_tes.append(target[d][rand_i])
                new_dataset = np.delete(dataset[d], rand_i, axis=0)
                new_target = np.delete(target[d], rand_i, axis=0)
            data_train.append(new_dataset)
            target_train.append(new_target)
        return data_train, target_train, data_tes, target_tes
