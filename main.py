import numpy as np
import datetime
from sklearn.datasets import load_iris
import pandas as pd
from GMM import GMM
import matplotlib.pyplot as plt
from comparator import ComparatorWithImpostor, Comparator
from cross_validation import CrossValidation


Ng = 3

class Tester:

    def __init__(self, number_of_gaussian):
        self.ng = number_of_gaussian
        self.gmm = GMM(number_of_gaussian)
        self.comparator_with_impostor = ComparatorWithImpostor(number_of_gaussian)
        self.comparator = Comparator(number_of_gaussian)
        self.cross_validator = CrossValidation(0.2)

    def initializer(self):
        data, target = load_iris(True)
        self.data_frame = []
        for d in data:
            df_t = pd.DataFrame(d)
            self.data_frame.append(df_t)

        self.target_frame = pd.DataFrame(target)
        self.X_train, self.y_train, self.X_test, self.y_test = self.cross_validator.exec(data, target)

    def plot(self, executions, index_models=0):
        startTime = datetime.datetime.now()

        xAxis = np.arange(executions)

        execs = [self.exec_with_time(index_models) for e in range(0, executions)]
        if index_models == 0:
            accepts, errors = zip(*execs)
        else:
            accepts, errors, impostor = zip(*execs)

        finishTime = datetime.datetime.now()
        duration = finishTime - startTime
        print('Duration Total')
        print(duration)

        fig, ax = plt.subplots()
        rects1 = ax.bar(xAxis, accepts, color='b')
        rects2 = ax.bar(xAxis, errors, color='r')

        ax.set_ylabel('Taxa de acerto')
        ax.set_title('GMM')
        ax.legend((rects1[0], rects2[0]), ('Acerto', 'Erro'))
        plt.show()


    def exec_with_time(self, index_models=0):
        startTime = datetime.datetime.now()
        if index_models == 0: result = self.exec()
        elif index_models == 1: result = self.exec_with_impostor()
        else: result = self.exec_with_impostor_universal()

        finishTime = datetime.datetime.now()

        duration = finishTime - startTime
        print('Duration parcial: ')
        print(duration)
        print('\n')

        return result

    def models_with_bms(self, X_train):
        c0, c1, c2 = X_train

        orq1 = np.array(c0)
        orq2 = np.array(c1)
        orq3 = np.array(c2)

        impostorOrq1 = np.concatenate([c1, c2])
        impostorOrq2 = np.concatenate([c0, c2])
        impostorOrq3 = np.concatenate([c0, c1])

        return [(self.gmm.model(orq1), self.gmm.model(impostorOrq1)),
                (self.gmm.model(orq2), self.gmm.model(impostorOrq2)),
                (self.gmm.model(orq3), self.gmm.model(impostorOrq3))]

    def models_with_universal(self, X_train):
        c0, c1, c2 = X_train

        orq1 = np.array(c0)
        orq2 = np.array(c1)
        orq3 = np.array(c2)

        impostorUniversal = np.concatenate([c0, c1, c2])
        universal = self.gmm.model(impostorUniversal)

        return [(self.gmm.model(orq1), universal),
                (self.gmm.model(orq2), universal),
                (self.gmm.model(orq3), universal)]

    def models(self, X_train):
        c0, c1, c2 = X_train

        orq1 = np.array(c0)
        orq2 = np.array(c1)
        orq3 = np.array(c2)

        return [self.gmm.model(orq1), self.gmm.model(orq2), self.gmm.model(orq3)]

    def exec(self):
        models = self.models(self.X_train)
        result = self.comparator.sumary(self.X_test, self.y_test, models)
        return result

    def exec_with_impostor(self):
        models = self.models_with_bms(self.X_train)
        result = self.comparator_with_impostor.sumary(self.X_test, self.y_test, models)
        return result

    def exec_with_impostor_universal(self):
        models = self.models_with_universal(self.X_train)
        result = self.comparator_with_impostor.sumary(self.X_test, self.y_test, models)
        return result


if __name__ == '__main__':
    t = Tester(Ng)
    t.initializer()
    print('------------------ Modelo sem impostors -----------------------')
    print('\n')
    t.plot(10, 0)
    print('------------------ Modelo modelo BPS -----------------------')
    print('\n')
    t.plot(3, 1)
    print('------------------ Modelo universal -----------------------')
    print('\n')
    t.plot(3, 2)

