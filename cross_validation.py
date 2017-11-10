import numpy as np
import random


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

        t0 = tar[0:10]
        t1 = tar[10:20]
        t2 = tar[20:30]

        return np.array([d0, d1, d2]), np.array([t0, t1, t2])

    def exec(self, data, tar):
        dataset, target = self.group_by_data(data, tar)

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
