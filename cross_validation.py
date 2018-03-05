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

        d = []
        t = []
        init = 0
        step = 10

        for idx in range(0, 42):
            d.append(data[init:init+step])
            t.append(tar[init:init + step])
            init += step

        return np.array(d), np.array(t)

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

        #
        # dataset = [self.decomposer(d) for d in dataset]

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
